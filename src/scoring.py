"""
Single forward-pass scoring engine and draft text processing.
Implements teacher-forced scoring over generated drafts to eliminate
step-by-step decoding artifacts for beam and contrastive search.
"""

import math
import re
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.metrics import (
    compute_entropy_from_logits,
    compute_token_log_prob,
    compute_token_prob,
    compute_msp
)


def cut_draft_at_first_newline_or_eos(text: str) -> str:
    """
    Cuts the generated draft answer at the first newline or EOS to prevent
    open-ended length artifacts and multi-turn hallucinations.
    """
    lines = text.split("\n")
    first_line = lines[0].strip() if lines else ""
    return first_line


def compute_distinct_n(text: str, n: int = 1) -> float:
    """
    Computes Distinct-n metric: |unique n-grams| / |total n-grams|.
    Tests for decoder repetition degeneration.
    """
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < n:
        return 1.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 1.0
    return len(set(ngrams)) / len(ngrams)


def compute_nucleus_entropy_from_logits(
    logits: torch.Tensor,
    temperature: float = 0.7,
    top_p: float = 0.9,
    unit: str = "nats"
) -> float:
    """
    Computes the entropy of the adjusted distribution under nucleus sampling
    (temperature scaling + top-p truncation).
    """
    scaled_logits = logits.float() / max(temperature, 1e-6)
    probs = F.softmax(scaled_logits, dim=-1)
    sorted_probs, _ = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Truncate at top_p
    mask = cumulative_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False

    nucleus_probs = sorted_probs[~mask]
    if nucleus_probs.numel() == 0:
        nucleus_probs = sorted_probs[:1]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()

    entropy = -torch.sum(nucleus_probs * torch.log(nucleus_probs + 1e-12)).item()
    if unit == "bits":
        entropy /= math.log(2.0)
    return float(entropy)


def score_tokens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_text: str,
    draft_text: str,
    device: str,
    unit: str = "nats",
    compute_nucleus_adjusted: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> Dict[str, Any]:
    """
    Performs a single teacher-forced forward pass over prompt + draft to compute
    exact per-token probabilities, log-probabilities, entropies, and MSPs.
    
    Guarantees stable, uncorrupted probability distributions across all decoders
    (Greedy, Beam Search, Contrastive Search, and Nucleus Sampling).
    """
    # Tokenize prompt alone to get exact prefix boundary
    prompt_enc = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = prompt_enc["input_ids"].to(device)
    prompt_len = prompt_ids.shape[1]

    # Combine prompt + draft
    full_text = prompt_text + draft_text
    full_enc = tokenizer(full_text, return_tensors="pt")
    full_ids = full_enc["input_ids"].to(device)
    full_len = full_ids.shape[1]

    if full_len <= prompt_len:
        # Empty draft or degenerate continuation
        return {
            "draft_text": draft_text,
            "draft_len": 0,
            "tokens": [],
            "max_entropy": 0.0,
            "mean_entropy": 0.0,
            "min_prob": 1.0,
            "mean_nll": 0.0,
            "distinct_1": 1.0,
            "distinct_2": 1.0,
            "distinct_3": 1.0,
            "adjusted_max_entropy": 0.0
        }

    draft_ids = full_ids[0, prompt_len:]
    num_draft_tokens = len(draft_ids)

    # Single forward pass
    with torch.no_grad():
        outputs = model(full_ids)
        draft_logits = outputs.logits[0, prompt_len - 1 : full_len - 1, :]

    assert draft_logits.shape[0] == num_draft_tokens, f"Logits length {draft_logits.shape[0]} != draft length {num_draft_tokens}"

    token_records = []
    probs = []
    log_probs = []
    entropies = []
    msps = []
    adjusted_entropies = []

    for step_idx in range(num_draft_tokens):
        tok_id = draft_ids[step_idx].item()
        tok_str = tokenizer.decode([tok_id])
        logits_step = draft_logits[step_idx].float().cpu()

        lp = compute_token_log_prob(logits_step, tok_id)
        p = compute_token_prob(logits_step, tok_id)
        ent = compute_entropy_from_logits(logits_step, unit=unit)
        msp = compute_msp(logits_step)

        probs.append(p)
        log_probs.append(lp)
        entropies.append(ent)
        msps.append(msp)

        rec = {
            "step": step_idx,
            "token_id": tok_id,
            "token_str": tok_str,
            "prob": round(p, 4),
            "log_prob": round(lp, 4),
            "entropy": round(ent, 4),
            "msp": round(msp, 4)
        }

        if compute_nucleus_adjusted:
            adj_ent = compute_nucleus_entropy_from_logits(logits_step, temperature=temperature, top_p=top_p, unit=unit)
            adjusted_entropies.append(adj_ent)
            rec["adjusted_entropy"] = round(adj_ent, 4)

        token_records.append(rec)

    distinct_1 = compute_distinct_n(draft_text, n=1)
    distinct_2 = compute_distinct_n(draft_text, n=2)
    distinct_3 = compute_distinct_n(draft_text, n=3)

    return {
        "draft_text": draft_text,
        "draft_len": num_draft_tokens,
        "tokens": token_records,
        "max_entropy": round(max(entropies), 4) if entropies else 0.0,
        "mean_entropy": round(float(np.mean(entropies)), 4) if entropies else 0.0,
        "min_prob": round(min(probs), 4) if probs else 1.0,
        "mean_nll": round(-float(np.mean(log_probs)), 4) if log_probs else 0.0,
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "distinct_3": round(distinct_3, 4),
        "adjusted_max_entropy": round(max(adjusted_entropies), 4) if adjusted_entropies else round(max(entropies), 4) if entropies else 0.0
    }
