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
from src.scoring import (
    score_tokens,
    cut_draft_at_first_newline_or_eos,
    compute_distinct_n,
    compute_nucleus_entropy_from_logits
)
from src.contrastive import contrastive_search_generate


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
        num_beams: int = 5,
        cut_draft: bool = True
    ) -> Dict[str, Any]:
        """
        Executes text generation under the specified decoder configuration.
        Computes both exact forward-pass raw uncertainty distributions and
        decoder-processed (adjusted) scores.
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
            if decoder_type == "contrastive_search":
                try:
                    outputs = self.model.generate(**inputs, **gen_kwargs)
                except Exception as e:
                    # Fallback to local offline implementation if remote custom_generate cannot be fetched
                    outputs = contrastive_search_generate(
                        model=self.model,
                        input_ids=input_ids,
                        max_new_tokens=max_new_tokens,
                        penalty_alpha=penalty_alpha,
                        top_k=top_k,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            else:
                outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_sequence = outputs.sequences[0]
        generated_tokens = generated_sequence[prompt_len:]
        full_generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Cut draft at first newline or EOS if requested
        if cut_draft:
            effective_text = cut_draft_at_first_newline_or_eos(full_generated_text)
        else:
            effective_text = full_generated_text

        # Compute exact forward pass scoring over prompt + effective_text
        compute_nucleus = (decoder_type == "nucleus_sampling")
        scored = score_tokens(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_text=prompt,
            draft_text=effective_text,
            device=self.device,
            compute_nucleus_adjusted=compute_nucleus,
            temperature=temperature,
            top_p=top_p
        )

        # Also extract decoder-processed scores if available
        processed_scores_steps = getattr(outputs, "scores", None)
        proc_entropies = []
        proc_probs = []
        proc_log_probs = []

        if processed_scores_steps:
            cut_token_count = scored["draft_len"]
            for step_idx in range(min(cut_token_count, len(processed_scores_steps))):
                proc_s = processed_scores_steps[step_idx][0].float().cpu()
                proc_ent = compute_entropy_from_logits(proc_s)
                proc_entropies.append(proc_ent)
                tok_id = scored["tokens"][step_idx]["token_id"] if step_idx < len(scored["tokens"]) else 0
                proc_probs.append(compute_token_prob(proc_s, tok_id))
                proc_log_probs.append(compute_token_log_prob(proc_s, tok_id))

        adjusted_max_entropy = (
            max(proc_entropies) if proc_entropies
            else scored.get("adjusted_max_entropy", scored["max_entropy"])
        )

        return {
            "decoder_type": decoder_type,
            "seed": seed,
            "generated_text": effective_text,
            "full_generated_text": full_generated_text,
            "draft_len": scored["draft_len"],
            "tokens": scored["tokens"],
            "raw_summary": {
                "min_prob": scored["min_prob"],
                "max_entropy": scored["max_entropy"],
                "mean_entropy": scored["mean_entropy"],
                "mean_nll": scored["mean_nll"]
            },
            "processed_summary": {
                "min_prob": min(proc_probs) if proc_probs else scored["min_prob"],
                "max_entropy": round(adjusted_max_entropy, 4),
                "mean_entropy": round(float(torch.tensor(proc_entropies).mean().item()), 4) if proc_entropies else scored["mean_entropy"],
                "mean_nll": -float(torch.tensor(proc_log_probs).mean().item()) if proc_log_probs else scored["mean_nll"]
            },
            "distinct_1": scored["distinct_1"],
            "distinct_2": scored["distinct_2"],
            "distinct_3": scored["distinct_3"],
            "contrastive_alpha": penalty_alpha if decoder_type == "contrastive_search" else None,
            "contrastive_k": top_k if decoder_type == "contrastive_search" else None
        }

