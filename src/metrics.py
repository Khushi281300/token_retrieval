"""
Core uncertainty metrics and trigger calculations for uncertainty-triggered RAG.
Based on Hashimoto et al. and DRAGIN / FLARE formulations.
"""

import math
import re
import string
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Union


# ==========================================
# 1. Per-Token Uncertainty Functions
# ==========================================

def compute_entropy_from_logits(logits: torch.Tensor) -> float:
    """
    Computes Shannon entropy (base e) of next-token distribution:
    H(P) = - sum(P * log P)
    Calculated in log-space for numerical stability.
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    # Avoid 0 * log(0) NaN by masking
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.item()


def compute_token_log_prob(logits: torch.Tensor, token_id: int) -> float:
    """
    Computes log P(token_id | context) = log_softmax(logits)[token_id].
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs[0, token_id].item()


def compute_token_prob(logits: torch.Tensor, token_id: int) -> float:
    """
    Computes P(token_id | context) = softmax(logits)[token_id].
    """
    return math.exp(compute_token_log_prob(logits, token_id))


def compute_msp(logits: torch.Tensor) -> float:
    """
    Maximum Softmax Probability (MSP):
    MSP = max_v P(v)
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    probs = F.softmax(logits, dim=-1)
    max_p = torch.max(probs, dim=-1).values
    return max_p.item()


# ==========================================
# 2. Sequence-Level Uncertainty Triggers
# ==========================================

def trigger_min_prob(token_probs: List[float], tau: float) -> bool:
    """
    FLARE/DRAGIN style minimum probability trigger:
    Retrieves if ANY generated token has probability < tau.
    """
    if not token_probs:
        return False
    return min(token_probs) < tau


def trigger_max_entropy(token_entropies: List[float], tau: float) -> bool:
    """
    FLARE/DRAGIN style maximum entropy trigger (MTE):
    Retrieves if ANY generated token has entropy > tau.
    """
    if not token_entropies:
        return False
    return max(token_entropies) > tau


def trigger_mean_nll(token_log_probs: List[float], tau: float) -> bool:
    """
    Mean negative log-likelihood of the draft:
    Retrieves if mean(-log P) > tau.
    """
    if not token_log_probs:
        return False
    mean_nll = -np.mean(token_log_probs)
    return bool(mean_nll > tau)


# ==========================================
# 3. QA Evaluation Metrics (F1, EM, Copy Rate)
# ==========================================

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    norm_pred = normalize_answer(prediction)
    return float(any(norm_pred == normalize_answer(gt) for gt in ground_truths))


def compute_f1(prediction: str, ground_truths: List[str]) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    max_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue
        common = set(pred_tokens) & set(gt_tokens)
        num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common)
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > max_f1:
            max_f1 = f1

    return max_f1


def compute_copy_rate(generated_text: str, context_text: str, n: int = 3) -> float:
    """
    Measures n-gram overlap between generated text and retrieved passages.
    Tests whether contrastive search penalizes copying from context.
    """
    def get_ngrams(tokens, n_val):
        return [tuple(tokens[i:i + n_val]) for i in range(len(tokens) - n_val + 1)]

    gen_tokens = normalize_answer(generated_text).split()
    ctx_tokens = normalize_answer(context_text).split()

    if len(gen_tokens) < n or len(ctx_tokens) < n:
        return 0.0

    gen_ngrams = get_ngrams(gen_tokens, n)
    ctx_ngrams_set = set(get_ngrams(ctx_tokens, n))

    copied = sum(1 for ng in gen_ngrams if ng in ctx_ngrams_set)
    return copied / len(gen_ngrams)


# ==========================================
# 4. Calibration & UQ Metrics (AUROC, PRR)
# ==========================================

def compute_prr(uncertainty_scores: List[float], correctness: List[int], steps: int = 100) -> float:
    """
    Prediction Rejection Ratio (PRR) as used in Hashimoto et al.
    Measures how much accuracy improves as we reject questions with highest uncertainty.
    Higher is better.
    """
    scores = np.array(uncertainty_scores)
    labels = np.array(correctness)

    # Sort ascending by uncertainty (most certain first)
    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]

    retention_fractions = np.linspace(0.1, 1.0, steps)
    accuracies = []

    for frac in retention_fractions:
        k = max(1, int(len(sorted_labels) * frac))
        accuracies.append(np.mean(sorted_labels[:k]))

    # Area under the accuracy-retention curve relative to baseline
    baseline_acc = np.mean(labels)
    
    # NumPy 2.0+ compatibility (trapz was renamed to trapezoid in NumPy 2.0)
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(accuracies, retention_fractions)
    elif hasattr(np, "trapz"):
        area = np.trapz(accuracies, retention_fractions)
    else:
        area = sum((accuracies[i] + accuracies[i+1]) * (retention_fractions[i+1] - retention_fractions[i]) / 2.0 for i in range(len(accuracies)-1))

    prr = area - (baseline_acc * 0.9)
    return float(prr)

