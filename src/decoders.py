"""
Multi-decoder generation engine supporting:
1. Greedy decoding
2. Beam search (num_beams=5)
3. Contrastive search (penalty_alpha in [0.4, 0.6, 0.8], top_k=4)
4. Nucleus sampling (T=0.7, top_p=0.9, multi-seed)

Extracts both raw logits and decoder-processed scores for every generated token.
"""

import math
from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.metrics import (
    compute_entropy_from_logits,
    compute_token_log_prob,
    compute_token_prob,
    compute_msp
)


class DecoderEngine:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: str = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def generate(
        self,
        prompt: str,
        decoder_type: str = "greedy",
        max_new_tokens: int = 32,
        seed: Optional[int] = None,
        penalty_alpha: float = 0.6,
        top_k: int = 4,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_beams: int = 5
    ) -> Dict[str, Any]:
        """
        Executes text generation under the specified decoder configuration and
        returns generated text, raw logits distribution, and processed scores distribution.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "return_dict_in_generate": True,
            "output_logits": True,    # Raw unnormalized forward pass logits
            "output_scores": True,    # Decoder-processed scores
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        decoder_type = decoder_type.lower()
        if decoder_type == "greedy":
            gen_kwargs.update({
                "do_sample": False,
                "num_beams": 1
            })

        elif decoder_type == "beam_search":
            gen_kwargs.update({
                "do_sample": False,
                "num_beams": num_beams,
                "early_stopping": True
            })

        elif decoder_type == "nucleus_sampling":
            gen_kwargs.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p
            })

        elif decoder_type == "contrastive_search":
            gen_kwargs.update({
                "penalty_alpha": penalty_alpha,
                "top_k": top_k,
                "custom_generate": "transformers-community/contrastive-search",
                "trust_remote_code": True
            })

        else:
            raise ValueError(f"Unsupported decoder_type: '{decoder_type}'")

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # In beam search, outputs.sequences contains (batch_size * num_return_sequences, seq_len)
        # Top beam sequence is at index 0
        generated_sequence = outputs.sequences[0]
        generated_tokens = generated_sequence[prompt_len:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        raw_logits_steps = outputs.logits
        processed_scores_steps = outputs.scores
        num_steps = len(generated_tokens)

        token_records = []
        raw_probs = []
        raw_entropies = []
        raw_log_probs = []
        proc_probs = []
        proc_entropies = []
        proc_log_probs = []

        for step_idx in range(num_steps):
            tok_id = generated_tokens[step_idx].item()
            tok_str = self.tokenizer.decode([tok_id])

            # Raw logits for step
            if step_idx < len(raw_logits_steps):
                raw_l = raw_logits_steps[step_idx][0].float().cpu()
                raw_lp = compute_token_log_prob(raw_l, tok_id)
                raw_p = compute_token_prob(raw_l, tok_id)
                raw_ent = compute_entropy_from_logits(raw_l)
                raw_msp = compute_msp(raw_l)
            else:
                raw_lp, raw_p, raw_ent, raw_msp = 0.0, 1.0, 0.0, 1.0

            raw_probs.append(raw_p)
            raw_entropies.append(raw_ent)
            raw_log_probs.append(raw_lp)

            # Processed scores for step
            if step_idx < len(processed_scores_steps):
                proc_s = processed_scores_steps[step_idx][0].float().cpu()
                proc_lp = compute_token_log_prob(proc_s, tok_id)
                proc_p = compute_token_prob(proc_s, tok_id)
                proc_ent = compute_entropy_from_logits(proc_s)
                proc_msp = compute_msp(proc_s)
            else:
                proc_lp, proc_p, proc_ent, proc_msp = 0.0, 1.0, 0.0, 1.0

            proc_probs.append(proc_p)
            proc_entropies.append(proc_ent)
            proc_log_probs.append(proc_lp)

            token_records.append({
                "step": step_idx,
                "token_id": tok_id,
                "token_str": tok_str,
                "raw": {
                    "log_prob": round(raw_lp, 4),
                    "prob": round(raw_p, 4),
                    "entropy": round(raw_ent, 4),
                    "msp": round(raw_msp, 4)
                },
                "processed": {
                    "log_prob": round(proc_lp, 4),
                    "prob": round(proc_p, 4),
                    "entropy": round(proc_ent, 4),
                    "msp": round(proc_msp, 4)
                }
            })

        return {
            "decoder_type": decoder_type,
            "seed": seed,
            "generated_text": generated_text,
            "num_tokens": num_steps,
            "tokens": token_records,
            "raw_summary": {
                "min_prob": min(raw_probs) if raw_probs else 1.0,
                "max_entropy": max(raw_entropies) if raw_entropies else 0.0,
                "mean_nll": -float(torch.tensor(raw_log_probs).mean().item()) if raw_log_probs else 0.0
            },
            "processed_summary": {
                "min_prob": min(proc_probs) if proc_probs else 1.0,
                "max_entropy": max(proc_entropies) if proc_entropies else 0.0,
                "mean_nll": -float(torch.tensor(proc_log_probs).mean().item()) if proc_log_probs else 0.0
            }
        }

