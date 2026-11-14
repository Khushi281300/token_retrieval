"""
Core uncertainty metrics and trigger calculations for uncertainty-triggered RAG.
Based on Hashimoto et al. (EMNLP Findings 2025) and DRAGIN / FLARE formulations.
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

def compute_entropy_from_logits(logits: torch.Tensor, unit: str = "nats") -> float:
    """
    Computes Shannon entropy of next-token distribution:
    H(P) = - sum(P * log P)
    Calculated in log-space for numerical stability.
    unit: 'nats' (base e) or 'bits' (base 2).
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    # Mask to prevent 0 * -inf = NaN
    mask = probs > 0
    entropy = -torch.sum(torch.where(mask, probs * log_probs, torch.zeros_like(probs)), dim=-1).item()
    
    if unit == "bits":
        entropy = entropy / math.log(2.0)
    return entropy


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
# 3. QA Evaluation Metrics (F1, EM, Rouge-L, Copy Rate)
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


def _lcs(x: List[str], y: List[str]) -> int:
    """Longest common subsequence length between token lists x and y."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if x[i] == y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]


def compute_rouge_l(prediction: str, ground_truths: List[str]) -> float:
    """
    Computes Rouge-L F1 (LCS-based) matching Hashimoto et al. quality score.
    Returns value between 0.0 and 1.0.
    """
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0

    max_rouge = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()
        if not gt_tokens:
            continue
        lcs_len = _lcs(pred_tokens, gt_tokens)
        if lcs_len == 0:
            continue
        prec = lcs_len / len(pred_tokens)
        rec = lcs_len / len(gt_tokens)
        f1 = (2 * prec * rec) / (prec + rec)
        if f1 > max_rouge:
            max_rouge = f1

    return max_rouge


def compute_copy_rate(generated_text: str, context_text: str, n: int = 3) -> float:
    """
    Measures n-gram overlap between generated text and retrieved passages.
    Tests whether contrastive search penalizes copying from context (RQ1c).
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
# 4. Calibration & UQ Metrics (AUROC, PRR, Cohen's Kappa, Bootstrap CI)
# ==========================================

def compute_cohen_kappa(decisions_a: List[Union[bool, int]], decisions_b: List[Union[bool, int]]) -> float:
    """
    Computes Cohen's kappa agreement between two binary decision vectors.
    Used to quantify per-question retrieval trigger agreement between Greedy
    and alternative decoders at matched retrieval rates.
    """
    assert len(decisions_a) == len(decisions_b), "Decisions lists must have the same length"
    if not decisions_a:
        return 1.0

    a = np.array(decisions_a, dtype=int)
    b = np.array(decisions_b, dtype=int)
    n = len(a)

    po = float(np.mean(a == b))

    p_a1 = float(np.mean(a == 1))
    p_a0 = 1.0 - p_a1
    p_b1 = float(np.mean(b == 1))
    p_b0 = 1.0 - p_b1

    pe = (p_a1 * p_b1) + (p_a0 * p_b0)

    if abs(1.0 - pe) < 1e-12:
        return 1.0

    kappa = (po - pe) / (1.0 - pe)
    return float(kappa)


def bootstrap_threshold_ci(
    scores: List[float],
    target_rate: float,
    num_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Computes threshold tau at target retrieval rate and its paired bootstrap
    confidence interval over Dev items.
    """
    scores_arr = np.array(scores)
    n = len(scores_arr)
    if n == 0:
        return {"tau": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    pct = (1.0 - target_rate) * 100.0
    point_tau = float(np.percentile(scores_arr, pct))

    rng = np.random.RandomState(seed)
    boot_taus = []
    for _ in range(num_bootstraps):
        resampled = rng.choice(scores_arr, size=n, replace=True)
        boot_taus.append(float(np.percentile(resampled, pct)))

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_taus, alpha * 100.0))
    ci_upper = float(np.percentile(boot_taus, (1.0 - alpha) * 100.0))

    return {
        "tau": round(point_tau, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4)
    }


def compute_prr(
    uncertainty_scores: List[float],
    quality_scores: List[float],
    steps: int = 100,
    scale_100: bool = True
) -> float:
    """
    Prediction Rejection Ratio (PRR) as defined in Hashimoto et al. (EMNLP Findings 2025).
    Measures how much average quality improves as high-uncertainty answers are rejected.
    
    uncertainty_scores: Higher means MORE uncertain.
    quality_scores: Higher means BETTER quality (e.g., Rouge-L or binary accuracy in [0, 1]).
    scale_100: If True, multiplies PRR by 100 to match Hashimoto's published scale.
    """
    scores = np.array(uncertainty_scores)
    labels = np.array(quality_scores)

    # Sort ascending by uncertainty (most confident answers retained first)
    sorted_indices = np.argsort(scores)
    sorted_labels = labels[sorted_indices]

    retention_fractions = np.linspace(0.1, 1.0, steps)
    accuracies = []

    for frac in retention_fractions:
        k = max(1, int(len(sorted_labels) * frac))
        accuracies.append(np.mean(sorted_labels[:k]))

    baseline_quality = float(np.mean(labels))

    # Numerical integration across retention curve
    if hasattr(np, "trapezoid"):
        area = np.trapezoid(accuracies, retention_fractions)
    elif hasattr(np, "trapz"):
        area = np.trapz(accuracies, retention_fractions)
    else:
        area = sum(
            (accuracies[i] + accuracies[i+1]) * (retention_fractions[i+1] - retention_fractions[i]) / 2.0
            for i in range(len(accuracies)-1)
        )

    # Area under retention curve relative to baseline
    prr = area - (baseline_quality * 0.9)
    if scale_100:
        prr = prr * 100.0
    return float(prr)
