"""
Day-1 Smoke Test Script for Uncertainty-Triggered RAG Project.
Verifies that:
1. The model generates text correctly.
2. Raw logits and decoder-processed scores are simultaneously extracted for every generated token.
3. Per-token log-probability and entropy are computed and stored.
4. Offline triggers (Min Prob, Max Entropy, Mean NLL) are calculated.
5. A timestamped configuration and result JSON is saved to runs/smoke_test.json.
"""

import os
import sys
import json
import time
import argparse
import platform
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.metrics import (
    compute_entropy_from_logits,
    compute_token_log_prob,
    compute_token_prob,
    compute_msp,
    trigger_min_prob,
    trigger_max_entropy,
    trigger_mean_nll
)


def run_smoke_test(
    model_name: str = "gpt2",
    prompt: str = "The capital of France is",
    device: str = None,
    output_path: str = "runs/smoke_test.json"
):
    print(f"=== Starting Smoke Test ===")
    print(f"Transformers version: {transformers.__version__}")
    print(f"PyTorch version:      {torch.__version__}")
    print(f"Operating System:     {platform.platform()}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Target Device:        {device}")
    print(f"Loading Model:        {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in appropriate precision
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None
    )
    if device == "cpu":
        model.to(device)
    model.eval()

    print(f"Model loaded successfully.")
    print(f"Prompt: '{prompt}'")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[1]

    # Run generation requesting BOTH raw logits and processed scores
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,              # Greedy decoding for baseline
            return_dict_in_generate=True,
            output_logits=True,           # Raw distribution (Section 6.0)
            output_scores=True,           # Decoder-processed scores (Section 6.0)
            pad_token_id=tokenizer.eos_token_id
        )

    generated_sequence = outputs.sequences[0]
    generated_tokens = generated_sequence[prompt_len:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    full_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)

    print(f"Generated text: '{generated_text}'")

    raw_logits_steps = outputs.logits
    processed_scores_steps = outputs.scores
    num_steps = len(generated_tokens)

    per_token_analysis = []
    raw_probs = []
    raw_entropies = []
    raw_log_probs = []

    proc_probs = []
    proc_entropies = []
    proc_log_probs = []

    for step_idx in range(num_steps):
        tok_id = generated_tokens[step_idx].item()
        tok_str = tokenizer.decode([tok_id])

        # 1. Raw logits distribution
        raw_l = raw_logits_steps[step_idx][0].float().cpu()
        raw_lp = compute_token_log_prob(raw_l, tok_id)
        raw_p = compute_token_prob(raw_l, tok_id)
        raw_ent = compute_entropy_from_logits(raw_l)
        raw_msp = compute_msp(raw_l)

        raw_probs.append(raw_p)
        raw_entropies.append(raw_ent)
        raw_log_probs.append(raw_lp)

        # 2. Processed scores distribution
        proc_s = processed_scores_steps[step_idx][0].float().cpu()
        proc_lp = compute_token_log_prob(proc_s, tok_id)
        proc_p = compute_token_prob(proc_s, tok_id)
        proc_ent = compute_entropy_from_logits(proc_s)
        proc_msp = compute_msp(proc_s)

        proc_probs.append(proc_p)
        proc_entropies.append(proc_ent)
        proc_log_probs.append(proc_lp)

        per_token_analysis.append({
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

    # Test Offline Triggers with example threshold
    sample_tau_prob = 0.5
    sample_tau_ent = 1.5

    triggers_result = {
        "raw": {
            "trigger_min_prob_tau_0.5": trigger_min_prob(raw_probs, sample_tau_prob),
            "trigger_max_entropy_tau_1.5": trigger_max_entropy(raw_entropies, sample_tau_ent),
            "trigger_mean_nll_tau_1.0": trigger_mean_nll(raw_log_probs, 1.0)
        },
        "processed": {
            "trigger_min_prob_tau_0.5": trigger_min_prob(proc_probs, sample_tau_prob),
            "trigger_max_entropy_tau_1.5": trigger_max_entropy(proc_entropies, sample_tau_ent),
            "trigger_mean_nll_tau_1.0": trigger_mean_nll(proc_log_probs, 1.0)
        }
    }

    result = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "device": device,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "platform": platform.platform()
        },
        "prompt": prompt,
        "generated_text": generated_text,
        "full_text": full_text,
        "num_generated_tokens": num_steps,
        "per_token_analysis": per_token_analysis,
        "triggers_test": triggers_result
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\n[SUCCESS] Smoke test completed!")
    print(f"Results written to: {output_path}")
    print(f"First 3 tokens preview:")
    for t in per_token_analysis[:3]:
        print(f"  Step {t['step']}: token='{t['token_str']}' | raw_p={t['raw']['prob']} | raw_H={t['raw']['entropy']}")
    print(f"Trigger check (raw max-entropy > 1.5): {triggers_result['raw']['trigger_max_entropy_tau_1.5']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run uncertainty RAG smoke test.")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name or HuggingFace ID")
    parser.add_argument("--device", type=str, default=None, help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--output", type=str, default="runs/smoke_test.json", help="Path to save output JSON")
    args = parser.parse_args()

    run_smoke_test(
        model_name=args.model,
        device=args.device,
        output_path=args.output
    )
