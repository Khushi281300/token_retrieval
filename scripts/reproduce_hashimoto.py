"""
Hashimoto et al. Baseline Evaluation Runner (EMNLP Findings 2025).
Evaluates greedy decoding on TriviaQA (rc.nocontext) to compute:
- Mean Rouge-L (primary quality metric in Hashimoto et al.)
- Exact Match (EM)
- MSP (Maximum Softmax Probability)
- MTE (Maximum Token Entropy)
- PRR (Prediction Rejection Ratio, scaled x100)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.metrics import (
    compute_entropy_from_logits,
    compute_token_prob,
    compute_msp,
    compute_exact_match,
    compute_f1,
    compute_rouge_l,
    compute_prr
)


def run_hashimoto_evaluation(
    model_name: str = "allenai/OLMo-2-1124-7B-Instruct",
    num_samples: int = 100,
    output_path: str = "runs/hashimoto_evaluation.json",
    entropy_unit: str = "nats"
):
    print("="*65)
    print("Hashimoto et al. Greedy Uncertainty Baseline (TriviaQA)")
    print(f"Model:        {model_name}")
    print(f"Samples:      {num_samples}")
    print(f"Entropy Unit: {entropy_unit}")
    print("="*65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model.to(device)
    model.eval()

    questions = []
    try:
        from datasets import load_dataset
        print(f"Loading TriviaQA (rc.nocontext validation split, {num_samples} items)...")
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True)
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            aliases = item.get("answer", {}).get("aliases", [])
            val = item.get("answer", {}).get("value", "")
            gts = list(set(aliases + ([val] if val else [])))
            questions.append({
                "id": item.get("question_id", str(i)),
                "question": item["question"],
                "answers": gts
            })
    except Exception as e:
        print(f"Warning: streaming TriviaQA failed ({e}). Loading local fallback...")
        with open("data/sample_qa.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))

    print(f"Starting inference on {len(questions)} questions...")

    msp_scores = []
    mte_scores = []
    rouge_l_scores = []
    em_scores = []
    detailed_records = []

    for idx, q in enumerate(questions):
        prompt = f"Question: {q['question']}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id
            )

        gen_tokens = outputs.sequences[0][prompt_len:]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Extract per-token entropy and MSP from raw forward pass
        token_entropies = []
        token_msps = []
        for step_idx in range(len(gen_tokens)):
            logits = outputs.logits[step_idx][0].float().cpu()
            token_entropies.append(compute_entropy_from_logits(logits, unit=entropy_unit))
            token_msps.append(compute_msp(logits))

        sample_mte = max(token_entropies) if token_entropies else 0.0
        sample_min_msp = min(token_msps) if token_msps else 1.0

        em = compute_exact_match(pred_text, q["answers"])
        rouge_l = compute_rouge_l(pred_text, q["answers"])
        f1 = compute_f1(pred_text, q["answers"])

        msp_scores.append(sample_min_msp)
        mte_scores.append(sample_mte)
        rouge_l_scores.append(rouge_l)
        em_scores.append(em)

        detailed_records.append({
            "id": q["id"],
            "question": q["question"],
            "prediction": pred_text,
            "ground_truths": q["answers"],
            "exact_match": em,
            "rouge_l": round(rouge_l, 4),
            "f1": round(f1, 4),
            "mte": round(sample_mte, 4),
            "min_msp": round(sample_min_msp, 4)
        })

    # Compute PRR using Rouge-L (Hashimoto's primary metric) and scale by 100
    prr_mte_rouge = compute_prr(uncertainty_scores=mte_scores, quality_scores=rouge_l_scores, scale_100=True)
    # For MSP, lower probability = more uncertain, so negate score
    prr_msp_rouge = compute_prr(uncertainty_scores=[-s for s in msp_scores], quality_scores=rouge_l_scores, scale_100=True)

    # Secondary PRR using exact match
    prr_mte_em = compute_prr(uncertainty_scores=mte_scores, quality_scores=em_scores, scale_100=True)
    prr_msp_em = compute_prr(uncertainty_scores=[-s for s in msp_scores], quality_scores=em_scores, scale_100=True)

    mean_rouge_l = float(np.mean(rouge_l_scores))
    mean_em = float(np.mean(em_scores))

    summary = {
        "metadata": {
            "model_name": model_name,
            "num_questions": len(questions),
            "entropy_unit": entropy_unit,
            "quality_metric": "Rouge-L (matching Hashimoto et al.)"
        },
        "mean_rouge_l": round(mean_rouge_l, 4),
        "mean_exact_match": round(mean_em, 4),
        "prr_x100": {
            "mte_rouge_l": round(prr_mte_rouge, 2),
            "msp_rouge_l": round(prr_msp_rouge, 2),
            "mte_exact_match": round(prr_mte_em, 2),
            "msp_exact_match": round(prr_msp_em, 2)
        },
        "records": detailed_records
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*65)
    print("HASHIMOTO BASELINE EVALUATION SUMMARY")
    print("="*65)
    print(f"Total Questions Evaluated:       {len(questions)}")
    print(f"Mean Rouge-L:                    {mean_rouge_l:.4f}")
    print(f"Exact Match (EM):                {mean_em:.2%}")
    print(f"PRR x100 (MTE with Rouge-L):     {prr_mte_rouge:.2f}")
    print(f"PRR x100 (MSP with Rouge-L):     {prr_msp_rouge:.2f}")
    print("="*65)
    print(f"Full results saved to: {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-1124-7B-Instruct")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--unit", type=str, default="nats", choices=["nats", "bits"])
    parser.add_argument("--output", type=str, default="runs/hashimoto_evaluation.json")
    args = parser.parse_args()

    run_hashimoto_evaluation(
        model_name=args.model,
        num_samples=args.samples,
        entropy_unit=args.unit,
        output_path=args.output
    )
