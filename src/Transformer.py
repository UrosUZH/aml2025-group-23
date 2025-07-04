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
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.beam_size = beam_size
        self.scoring_mode = scoring_mode
        self.len_gamma = len_gamma
        self.coverage_lambda = coverage_lambda
        self.ent_lambda = ent_lambda

        self.max_memory = {
            0: "12GB",
            "cpu": "32GB"
        }

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        quant_ok = load_in_8bit

        if quant_ok:
            try:
                bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
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
        out = self.model(input_ids=prefix_ids, past_key_values=past)
        logits = out.logits[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).item()
        return log_probs, out.past_key_values, entropy

    @torch.no_grad()
    def _full_gloss_logprob(self, gloss: str, past) -> Tuple[float, tuple, float]:
        toks = self.tokenizer.encode(" " + gloss, add_special_tokens=False)
        total_lp = 0.0
        entropies = []
        for tid in toks:
            input_ids = torch.tensor([[tid]], device=self.device)
            lp, past = self._lm_logprobs(input_ids, past)
            total_lp += lp[0, tid].item()
            entropies.append(-(lp.exp() * lp).sum(dim=-1).item())
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        return total_lp, past, avg_entropy

    def decode(self, candidate_lists: List[List[Tuple[str, float]]]) -> str:
        bos_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        beams: list[tuple[list[str], float, tuple | None, int]] = [([], 0.0, None, 0)]

        for window in candidate_lists:
            next_beams: list[tuple[list[str], float, tuple | None, float]] = []

            for tokens, score_so_far, past, coverage_count in beams:
                if tokens:
                    prefix_text = " ".join(tokens)
                    prefix_ids = self.tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                else:
                    prefix_ids = torch.tensor([[bos_id]], device=self.device)

                _, past_next, _ = self._lm_logprobs(prefix_ids, past)

                for gloss, p_s in window:
                    if not gloss.strip():
                        continue
                    try:
                        logp_lm, past_out, entropy = self._full_gloss_logprob(gloss, past_next)
                    except Exception:
                        continue

                    base_score = self.alpha * logp_lm + self.beta * math.log(p_s + 1e-12)
                    new_score = score_so_far + base_score

                    # Different scoring adjustments
                    if self.scoring_mode == "length_norm":
                        length_penalty = (len(tokens) + 1) ** self.len_gamma
                        adjusted_score = new_score / length_penalty
                    elif self.scoring_mode == "coverage":
                        adjusted_score = new_score + self.coverage_lambda * (coverage_count + 1)
                    elif self.scoring_mode == "uncertainty":
                        adjusted_score = new_score - self.ent_lambda * entropy
                    else:
                        adjusted_score = new_score

                    next_beams.append((tokens + [gloss], adjusted_score, past_out, coverage_count + 1))

            if not next_beams:
                best_gloss, p_s = window[0]
                for tokens, score_so_far, past, coverage_count in beams:
                    score = score_so_far + self.beta * math.log(p_s + 1e-12)
                    next_beams.append((tokens + [best_gloss], score, past, coverage_count + 1))

            next_beams.sort(key=lambda x: x[1], reverse=True)
            beams = next_beams[: self.beam_size]

        best_tokens, _, _, _ = beams[0]
        return self._post_process(best_tokens)

    @staticmethod
    def _post_process(tokens: List[str]) -> str:
        result: list[str] = []
        for tok in tokens:
            if not result or tok != result[-1]:
                result.append(tok)
        sent = " ".join(result).strip()
        if not sent.endswith("."):
            sent += "."
        return sent.capitalize()


__all__ = ["TransformerDecoder"]