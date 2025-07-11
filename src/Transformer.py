import math
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class TransformerDecoder:
    """Beam-search decoder that mixes SignCLIP logits with a causal LM.

        The score for a hypothesis is
            score = alpha · log P_LM(token | prefix)  +  beta · log P_SignCLIP(token)
        where log P_LM comes from a Hugging-Face causal model and log P_SignCLIP
        is the (soft-maxed) similarity of the candidate gloss in its window.

        Parameters
        ----------
        lm_name : str
            Hugging-Face repo or local path.  Defaults to a 7-B instruct model
            but any causal LM works (Gemma-2B-it, Phi-2, etc.).
        beam_size : int
            Number of parallel hypotheses (default 5).
        alpha : float
            Weight for the LM terms.
        beta : float
            Weight for SignCLIP terms.
        device : str | None
            "cuda" / "cpu" / None → auto.  If CUDA + bitsandbytes are present we
            attempt 8-bit loading; otherwise fall back to fp16 (GPU) or fp32 (CPU).
        load_in_8bit : bool
            Force or forbid 8-bit.  If None (default) we choose automatically.
        scoring_mode : str
            How to adjust the score. Options are:
            - "default": no adjustment.
            - "length_norm": divide by (len(tokens) + 1) ** len_gamma
            - "coverage": add coverage_lambda * (coverage_count + 1)
            - "uncertainty": subtract ent_lambda * entropy
        len_gamma : float
            Exponent for length normalization (default 0.7).
        coverage_lambda : float
            Weight for coverage adjustment (default 0.1).
        ent_lambda : float
            Weight for uncertainty adjustment (default 0.05).
        open_set_k : int
            If > 0, the LM will also generate open-set tokens (top-k logits).
            This is useful for models that can generate tokens not in the
            SignCLIP vocabulary, e.g., "mistralai/Mistral-7B-Instruct-v0.2".
            If 0 (default), only SignCLIP glosses are used.
        min_logp_lm_threshold : float
            Minimum log probability for a token to be considered by the LM.
            If the LM's log probability is below this threshold, the token is
            not extended in the beam search. This helps to filter out unlikely
            tokens and avoid noise in the generated sentences.
        refine : bool
            If True (default), the final sentence is rewritten by the LM for fluency.
        debug : bool
            If True, prints debug information about the beam search process.

        Notes
        -----
        - The model must have a BOS token, which is used to start decoding.
        - The model must have a PAD token, which is used to pad the input.
        - The model must be a causal LM (e.g., GPT, Mistral, etc.).
        - The model must be able to handle the BOS token as the first token.
        - The model must be able to handle the PAD token as the last token.
    """
    def __init__(
        self,
        lm_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        beam_size: int = 5,
        alpha: float = 0.7,
        beta: float = 0.3,
        device: str | None = None,
        load_in_8bit: bool | None = None,
        scoring_mode: str = "default",  # options: default, length_norm, coverage, uncertainty
        len_gamma: float = 0.7,
        coverage_lambda: float = 0.1,
        ent_lambda: float = 0.05,
        open_set_k: int = 0,
        min_logp_lm_threshold: float = -40.0,
        refine: bool = True,
        debug: bool = False,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.beam_size = beam_size
        self.scoring_mode = scoring_mode
        self.len_gamma = len_gamma
        self.coverage_lambda = coverage_lambda
        self.ent_lambda = ent_lambda
        self.open_set_k = open_set_k
        self.refine = refine
        self.debug = debug
        self.min_logp_lm_threshold = min_logp_lm_threshold

        self.max_memory = {
            0: "16GB",
            "cpu": "32GB"
        }

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        quant_ok = load_in_8bit

        if quant_ok:
            try:
                bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    lm_name,
                    device_map="auto",
                    max_memory=self.max_memory,
                    quantization_config=bnb_cfg,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    trust_remote_code=True,
                    offload_folder="offload",
                    offload_state_dict=True,
                )
            except Exception as e:
                print(f"⚠️  8-bit load failed ({e.__class__.__name__}: {e}). Falling back to fp16/fp32.")
                quant_ok = False

        if not quant_ok:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                lm_name,
                torch_dtype=dtype,
                device_map={"": self.device},
                trust_remote_code=True,
            )

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.bos_token is None and hasattr(self.tokenizer, "bos_token_id"):
            self.tokenizer.bos_token = self.tokenizer.eos_token

    @torch.no_grad()
    def _lm_logprobs(self, prefix_ids: torch.LongTensor, past=None):
        """
        Compute log-probabilities and entropy of the next token given a prefix.

        Parameters
        ----------
        prefix_ids : torch.LongTensor
            Token IDs of the prefix input sequence.
        past : optional
            Cached past key values for faster decoding (not used currently).

        Returns
        -------
        log_probs : torch.FloatTensor
            Log-probabilities for the next token over the vocabulary.
        past_key_values : tuple
            Cached past key values for future use.
        entropy : float
            Entropy of the predicted next-token distribution.
        """
        out = self.model(input_ids=prefix_ids, past_key_values=past)
        logits = out.logits[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).item()
        return log_probs, out.past_key_values, entropy

    @torch.no_grad()
    def _full_gloss_logprob(self, gloss: str, prefix_tokens: List[str]) -> Tuple[float, float]:
        """
        Compute average log-probability and entropy for a gloss continuation.

        Parameters
        ----------
        gloss : str
            Candidate gloss to evaluate.
        prefix_tokens : List[str]
            Current token sequence as text tokens.

        Returns
        -------
        avg_lp : float
            Average log-probability of the gloss conditioned on the prefix.
        avg_entropy : float
            Average entropy over the gloss tokens.
        """
        # Build text strings
        prefix_text = " ".join(prefix_tokens)
        full_text = prefix_text + " " + gloss if prefix_text else gloss

        # Tokenize both
        prefix_ids = self.tokenizer.encode(prefix_text, return_tensors="pt").to(self.device)
        full_ids = self.tokenizer.encode(full_text, return_tensors="pt").to(self.device)

        # Forward full sequence
        outputs = self.model(full_ids)
        logits = outputs.logits[:, :-1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        
        gloss_start = prefix_ids.shape[1]
        new_token_ids = full_ids[0][gloss_start:]

        logp = 0.0
        entropy_sum = 0.0

        start_pos = prefix_ids.shape[1] - 1
        if start_pos < 0:
            start_pos = 0

        for i, tid in enumerate(new_token_ids):
            pos = start_pos + i
            if pos >= log_probs.shape[1]:
                break
            lp = log_probs[0, pos, tid].item()
            logp += lp

            token_log_probs = log_probs[0, pos, :]
            token_entropy = -(token_log_probs.exp() * token_log_probs).sum().item()
            entropy_sum += token_entropy

        avg_lp = logp / len(new_token_ids) if len(new_token_ids) > 0 else float('-inf')
        avg_entropy = entropy_sum / len(new_token_ids) if len(new_token_ids) > 0 else 0.0

        return avg_lp, avg_entropy


    def decode(self, candidate_lists: List[List[Tuple[str, float]]]) -> str:
        """
        Perform beam search to decode the most fluent and relevant sentence 
        from a list of candidate glosses at each time step.

        Parameters
        ----------
        candidate_lists : List[List[Tuple[str, float]]]
            A list of gloss candidates for each decoding step.
            Each inner list contains (gloss, SignCLIP_score) tuples.

        Returns
        -------
        str
            The final decoded and optionally refined sentence.
        """
        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        beams: list[tuple[list[str], float, int]] = [([], 0.0, 0)]

        for window_idx, window in enumerate(candidate_lists):
            next_beams: list[tuple[list[str], float, int]] = []

            if self.debug:
                print(f"\n=== Window {window_idx} ===")

            for tokens, score_so_far, cov in beams:
                # encode current prefix once
                prefix_ids = (
                    self.tokenizer(" ".join(tokens), return_tensors="pt",
                                    add_special_tokens=False).input_ids.to(self.device)
                    if tokens else torch.tensor([[bos_id]], device=self.device)
                )
                lm_next, _, _ = self._lm_logprobs(prefix_ids)

                # 1) SignCLIP glosses
                for gloss, p_s in window:
                    new_beam = self._extend_beam(gloss, p_s, tokens, score_so_far,
                                lm_next, cov)
                    if new_beam:
                        next_beams.append(new_beam)

                # 2) Open-set LM tokens (optional)
                if self.open_set_k > 0:
                    topk_logits = torch.topk(lm_next, self.open_set_k, dim=-1)
                    for tid in topk_logits.indices[0].tolist():
                        token_str = self.tokenizer.decode([tid]).strip()
                        if not token_str or token_str == self.tokenizer.eos_token:
                            continue
                        new_beam = self._extend_beam(token_str, 1.0, tokens, score_so_far,
                                          lm_next, cov,
                                          subtoken_tid=tid)
                        if new_beam:
                            next_beams.append(new_beam)

            # fallback & prune
            if not next_beams:
                best_gloss, p_s = window[0]
                for tokens, score_so_far, coverage_count in beams:
                    score = score_so_far + self.beta * math.log(p_s + 1e-12)
                    next_beams.append((tokens + [best_gloss], score, coverage_count + 1))
                    
            next_beams.sort(key=lambda x: x[1], reverse=True)

            if self.debug:
                print(f"Window {window_idx} - {len(next_beams)} candidates:")
                print(f"Showing top {self.beam_size} beams:")
                for b in next_beams[:self.beam_size]:
                    print(f"  Beam: {' '.join(b[0])}, Score: {b[1]:.2f}")

            beams = next_beams[: self.beam_size]

        best_tokens = beams[0][0]

        if self.debug:
            print("\n=== Top 5 Beams ===")
            for b in beams[:5]:
                print(f"  Beam: {' '.join(b[0])}, Score: {b[1]:.2f}")

        draft = self._post_process(best_tokens)
        return self._refine_sentence(draft) if self.refine else draft

    def _extend_beam(self, gloss, p_s, tokens, score_so_far,
                 lm_next, coverage_count, *, subtoken_tid=None) -> tuple:
        # LM score (if subtoken_tid provided use cached lm_next)
        if subtoken_tid is not None:
            logp_lm = lm_next[0, subtoken_tid].item()
            entropy = -(lm_next.exp() * lm_next).sum(dim=-1).item()
        else:
            logp_lm, entropy = self._full_gloss_logprob(gloss, tokens)

        # Threshold: skip if LLM doesn't like this gloss
        if logp_lm < self.min_logp_lm_threshold:
            if self.debug:
                print(f"Skipping gloss '{gloss}' due to low logp_lm ({logp_lm:.2f})")
            return None

        base = self.alpha * logp_lm + self.beta * math.log(p_s + 1e-12)
        new_score = score_so_far + base

        if self.debug:
            print(f"Beam extension for gloss '{gloss}':")
            print(f"    Gloss: {gloss}, p_s: {p_s:.2f}, logp_lm: {logp_lm:.2f}, base: {base:.2f}, new: {new_score:.2f}")

        # scoring-mode tweaks
        if self.scoring_mode == "length_norm":
            new_score /= (len(tokens)+1) ** self.len_gamma
        elif self.scoring_mode == "coverage":
            new_score += self.coverage_lambda * (coverage_count+1)
        elif self.scoring_mode == "uncertainty":
            new_score -= self.ent_lambda * entropy

        return (tokens + [gloss], new_score, coverage_count+1)

    def _refine_sentence(self, draft: str) -> str:
        """
        Optionally post-process the draft sentence using the language model 
        to make it more fluent or grammatically correct.

        Parameters
        ----------
        draft : str
            The raw output sequence from beam search.

        Returns
        -------
        str
            The refined sentence, if generation succeeded; otherwise the original draft.
        """
        if self.debug:
            print(f"\nDraft before refinement: '{draft}'")
        prompt = f"What does this sentence mean? Can you write this sentence in a way that it makes sense? Just write the sentence:\n\"{draft}\""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
        gen = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=False,
            num_beams=1,
            eos_token_id=self.tokenizer.eos_token_id
        )
        rewritten = self.tokenizer.decode(gen[0][input_ids.size(1):],
                                          skip_special_tokens=True).strip()
        # Sometimes model returns empty / identical:
        return rewritten if rewritten else draft
    
    @staticmethod
    def _post_process(tokens: List[str]) -> str:
        """
        Convert a list of tokens into a well-formed sentence.

        - Removes consecutive duplicate tokens.
        - Joins tokens into a string.
        - Adds terminal punctuation if missing.
        - Capitalizes the first letter.

        Parameters
        ----------
        tokens : List[str]
            Token sequence to process.

        Returns
        -------
        str
            A cleaned-up sentence.
        """
        result: list[str] = []
        for tok in tokens:
            if not result or tok != result[-1]:
                result.append(tok)
        sent = " ".join(result).strip()
        if not sent.endswith("."):
            sent += "."
        return sent.capitalize()

__all__ = ["TransformerDecoder"]