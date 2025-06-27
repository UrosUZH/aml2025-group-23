import math
from typing import List, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


class TransformerDecoder:
    """Decode SignCLIP top-k gloss predictions into a sentence with a causal LM

    Parameters
    ----------
    lm_name : str
        Hugging Face model repo or local path.  Default picks a compact 7-B model
        that fits in 16 GB VRAM with 8-bit weights.
    beam_size : int, optional
        Number of active hypotheses, by default 5.
    alpha : float, optional
        Weight for LM log-probabilities, by default 0.7.
    beta : float, optional
        Weight for SignCLIP log-probabilities, by default 0.3.
    device : str | None, optional
        "cuda", "cpu", or None for auto.  If a CUDA device is available the
        model is loaded there with 8-bit quantisation via *bitsandbytes*.
    """

    def __init__(
        self,
        lm_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        beam_size: int = 5,
        alpha: float = 0.7,
        beta: float = 0.3,
        device: str | None = None,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.beam_size = beam_size

        # ------------------------- device & quant --------------------------- #
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.device == "cuda":
            # 8‑bit quantisation ⇒  ~14 GB for 7‑B params → fits in 16 GB VRAM.
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                lm_name,
                device_map="auto",
                quantization_config=bnb_cfg,
                trust_remote_code=True,
            )
        else:
            # CPU fallback (can still be quantised if you prefer GGUF/4‑bit)
            self.model = AutoModelForCausalLM.from_pretrained(
                lm_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map={"": self.device},
                trust_remote_code=True,
            )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def _lm_logprobs(
        self, prefix_ids: torch.LongTensor, past_key_values=None
    ) -> Tuple[torch.Tensor, tuple]:
        """Return log‑probs over vocab for next token and updated cache."""
        out = self.model(input_ids=prefix_ids, past_key_values=past_key_values)
        logits = out.logits[:, -1, :]  # (B, |V|)
        return torch.log_softmax(logits, dim=-1), out.past_key_values

    # --------------------------------------------------------------------- #
    def decode(
        self, candidate_lists: List[List[Tuple[str, float]]]
    ) -> str:  # noqa: D401
        """Beam‑search through windows of gloss candidates.

        Each *candidate_lists[t]* is a list of (gloss, softmax_prob_from_signclip)
        sorted by probability.  Returns the best sentence string.
        """

        # beams: (tokens, score, past_key_values)
        beams = [([], 0.0, None)]

        for window in candidate_lists:
            next_beams: list[tuple[list[str], float, tuple]] = []

            for tokens, score_so_far, past in beams:
                # Encode current prefix once for this beam
                prefix_text = " ".join(tokens) if tokens else ""
                prefix_ids = self.tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.device)

                lm_logp, past_next = self._lm_logprobs(prefix_ids, past)

                for gloss, p_s in window:
                    tid_list = self.tokenizer.encode(gloss, add_special_tokens=False)
                    if len(tid_list) != 1:
                        continue  # skip multi‑subword glosses
                    tid = tid_list[0]
                    logp_lm = lm_logp[0, tid].item()

                    new_score = score_so_far + self.alpha * logp_lm + self.beta * math.log(p_s + 1e-9)
                    next_beams.append((tokens + [gloss], new_score, past_next))

            # prune
            next_beams.sort(key=lambda x: x[1], reverse=True)
            beams = next_beams[: self.beam_size]

        best_tokens, _, _ = beams[0]
        return self._post_process(best_tokens)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _post_process(tokens: List[str]) -> str:
        """Simple cleanup: remove consecutive repeats and capitalise.«"""  # noqa: D400
        deduped = []
        for tok in tokens:
            if not deduped or tok != deduped[-1]:
                deduped.append(tok)
        sentence = " ".join(deduped)
        return sentence.capitalize() + "."


__all__ = ["TransformerDecoder"]

