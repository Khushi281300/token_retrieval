"""
Standalone Contrastive Search generator for offline and local environments.
Implements Su et al. (2022):
x_t = argmax_{v in TopK(P)} [ (1 - alpha) * P(v | x_<t) - alpha * max_{j < t} cos_sim(h_v, h_j) ]
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from transformers.modeling_outputs import CausalLMOutputWithPast


def contrastive_search_generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    penalty_alpha: float = 0.6,
    top_k: int = 4,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None
):
    """
    Executes Contrastive Search generation step-by-step.
    Returns an object with .sequences, .logits, and .scores matching model.generate().
    """
    device = input_ids.device
    batch_size, prompt_len = input_ids.shape
    curr_ids = input_ids.clone()

    step_logits = []
    step_scores = []

    # Get initial hidden states and logits
    with torch.no_grad():
        outputs = model(curr_ids, output_hidden_states=True)
        # Logits at last prompt token
        next_token_logits = outputs.logits[:, -1, :]
        last_hidden_state = outputs.hidden_states[-1] # (batch, seq_len, hidden_dim)

    past_key_values = outputs.past_key_values if hasattr(outputs, "past_key_values") else None

    for step in range(max_new_tokens):
        # 1. Candidate selection: Top-K tokens by probability
        probs = F.softmax(next_token_logits, dim=-1)
        top_k_probs, top_k_ids = torch.topk(probs, k=top_k, dim=-1) # (batch, k)

        step_logits.append(next_token_logits)

        # 2. Re-ranking: Degeneration penalty via cosine similarity with past hidden states
        best_score = -float("inf")
        best_token_id = top_k_ids[0, 0]
        adjusted_scores_step = torch.full_like(next_token_logits, -float("inf"))

        # Normalized past representations
        past_h = last_hidden_state[0] # (seq_len, hidden_dim)
        past_h_norm = F.normalize(past_h, p=2, dim=-1)

        candidate_scores = []
        for cand_idx in range(top_k):
            cand_id = top_k_ids[0, cand_idx].view(1, 1)
            cand_prob = top_k_probs[0, cand_idx].item()

            # Forward pass candidate token to get its hidden state
            with torch.no_grad():
                cand_out = model(cand_id, past_key_values=past_key_values, output_hidden_states=True)
                cand_h = cand_out.hidden_states[-1][0, -1, :] # (hidden_dim)
                cand_h_norm = F.normalize(cand_h, p=2, dim=-1)

            # Cosine similarity against all tokens in context
            sims = torch.mv(past_h_norm, cand_h_norm) # (seq_len,)
            max_sim = float(torch.max(sims).item())

            # Contrastive objective
            score = (1.0 - penalty_alpha) * cand_prob - penalty_alpha * max_sim
            candidate_scores.append(score)
            adjusted_scores_step[0, top_k_ids[0, cand_idx]] = score

            if score > best_score:
                best_score = score
                best_token_id = top_k_ids[0, cand_idx]

        step_scores.append(adjusted_scores_step)

        # Append chosen token
        chosen_token = best_token_id.view(1, 1)
        curr_ids = torch.cat([curr_ids, chosen_token], dim=-1)

        if eos_token_id is not None and (chosen_token == eos_token_id).all():
            break

        # Advance past_key_values and hidden states for next step
        with torch.no_grad():
            outputs = model(chosen_token, past_key_values=past_key_values, output_hidden_states=True)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values if hasattr(outputs, "past_key_values") else None
            # Update last_hidden_state
            last_hidden_state = torch.cat([last_hidden_state, outputs.hidden_states[-1]], dim=1)

    class ContrastiveOutput:
        def __init__(self, sequences, logits, scores):
            self.sequences = sequences
            self.logits = logits
            self.scores = scores

    return ContrastiveOutput(
        sequences=curr_ids,
        logits=tuple(step_logits),
        scores=tuple(step_scores)
    )
