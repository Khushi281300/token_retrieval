"""
Step 2: Reproduction of Hashimoto et al. (Uncertainty Quantification for LLMs)
Evaluates greedy decoding on TriviaQA (rc.nocontext) to compute:
- MSP (Maximum Softmax Probability)
- MTE (Maximum Token Entropy)
- PRR (Prediction Rejection Ratio)
- Accuracy vs Rejection curves
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
    compute_prr
)


def run_hashimoto_reproduction(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_samples: int = 50,
    output_path: str = "runs/hashimoto_reproduction.json"
):
    print("="*60)
    print("Step 2: Hashimoto et al. TriviaQA Reproduction Baseline")
    print(f"Model:       {model_name}")
    print(f"Samples:     {num_samples}")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device if device == "cuda" else None
    )
    if device == "cpu":
        model.to(device)
    model.eval()

    # Try loading trivia_qa from huggingface datasets or fallback to sample
    questions = []
    try:
        from datasets import load_dataset
        print("Fetching TriviaQA (rc.nocontext)...")
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation", streaming=True)
        for i, item in enumerate(ds):
            if i >= num_samples:
                break
            questions.append({
                "id": item["question_id"],
                "question": item["question"],
                "answers": item["answer"]["aliases"] + [item["answer"]["value"]]
            })
    except Exception as e:
        print(f"Streaming TriviaQA failed ({e}). Loading fallback samples...")
        with open("data/sample_qa.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    questions.append(item)

    print(f"Loaded {len(questions)} evaluation questions.")

    results = []
    msp_scores = []
    mte_scores = []
    correctness_labels = []

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
        
        # Uncertainty metrics
        token_entropies = []
        token_msps = []
        for step_idx in range(len(gen_tokens)):
            logits = outputs.logits[step_idx][0].float().cpu()
            token_entropies.append(compute_entropy_from_logits(logits))
            token_msps.append(compute_msp(logits))

        sample_mte = max(token_entropies) if token_entropies else 0.0
        sample_min_msp = min(token_msps) if token_msps else 1.0

        em = compute_exact_match(pred_text, q["answers"])
        f1 = compute_f1(pred_text, q["answers"])

        msp_scores.append(sample_min_msp)
        mte_scores.append(sample_mte)
        correctness_labels.append(int(em > 0.5))

        results.append({
            "id": q.get("id", str(idx)),
            "question": q["question"],
            "prediction": pred_text,
            "ground_truths": q["answers"],
            "exact_match": em,
            "f1": f1,
            "mte": round(sample_mte, 4),
            "min_msp": round(sample_min_msp, 4)
        })

    # Compute PRR for MTE and MSP
    # For PRR with MTE, uncertainty is MTE (higher is more uncertain)
    prr_mte = compute_prr(mte_scores, correctness_labels)
    # For PRR with MSP, uncertainty is -MSP (lower probability is more uncertain)
    prr_msp = compute_prr([-s for s in msp_scores], correctness_labels)

    baseline_acc = float(np.mean(correctness_labels))

    summary = {
        "metadata": {
            "model_name": model_name,
            "num_questions": len(questions),
            "device": device
        },
        "baseline_accuracy": round(baseline_acc, 4),
        "prr_mte": round(prr_mte, 4),
        "prr_msp": round(prr_msp, 4),
        "detailed_results": results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("HASHIMOTO REPRODUCTION RESULTS SUMMARY")
    print("="*60)
    print(f"Total Questions Evaluated: {len(questions)}")
    print(f"Greedy Baseline Accuracy (EM): {baseline_acc:.2%}")
    print(f"PRR (Maximum Token Entropy - MTE): {prr_mte:.4f}")
    print(f"PRR (Minimum Softmax Prob - MSP):  {prr_msp:.4f}")
    print(f"Full results written to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", type=str, default="runs/hashimoto_reproduction.json")
    args = parser.parse_args()

    run_hashimoto_reproduction(model_name=args.model, num_samples=args.samples, output_path=args.output)
