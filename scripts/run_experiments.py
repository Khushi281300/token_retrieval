"""
Unified Experiment Runner for Uncertainty-Triggered RAG.
Runs comparative evaluations across decoders:
1. Fixed-threshold comparison (RQ1a)
2. Matched-retrieval-rate comparison (RQ1b)
3. Answer decoder copy-rate analysis (RQ1c)
"""

import os
import sys
import json
import time
import argparse
import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.decoders import DecoderEngine
from src.retriever import LocalBM25Retriever
from src.pipeline import RAGPipeline


def load_dataset(file_path: str):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def run_experiment(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    data_path: str = "data/sample_qa.jsonl",
    output_dir: str = "runs",
    decoders: list = None,
    trigger_type: str = "max_entropy",
    fixed_tau: float = 2.5
):
    if decoders is None:
        decoders = ["greedy", "beam_search", "nucleus_sampling"]

    print(f"==================================================")
    print(f"Starting Multi-Decoder Uncertainty RAG Experiment")
    print(f"Model:        {model_name}")
    print(f"Dataset:      {data_path}")
    print(f"Trigger Type: {trigger_type} (Fixed tau = {fixed_tau})")
    print(f"Decoders:     {decoders}")
    print(f"==================================================")

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

    decoder_engine = DecoderEngine(model=model, tokenizer=tokenizer, device=device)
    dataset = load_dataset(data_path)

    # Build local corpus from dataset passages
    corpus = []
    for item in dataset:
        for p in item.get("passages", []):
            corpus.append({"id": item["id"], "text": p})

    retriever = LocalBM25Retriever(corpus=corpus)
    pipeline = RAGPipeline(decoder_engine=decoder_engine, retriever=retriever, trigger_type=trigger_type)

    results_by_decoder = {}

    for dec in decoders:
        print(f"\n---> Running Decoder: {dec.upper()} ...")
        dec_results = []
        for item in dataset:
            res = pipeline.run_single_question(
                question=item["question"],
                ground_truths=item["answers"],
                draft_decoder=dec,
                answer_decoder="greedy",
                tau=fixed_tau,
                max_draft_tokens=16,
                max_answer_tokens=24,
                seed=42
            )
            dec_results.append(res)

        # Calculate statistics
        total = len(dec_results)
        retrieval_count = sum(1 for r in dec_results if r["retrieval_fired"])
        retrieval_rate = retrieval_count / max(1, total)
        avg_f1 = sum(r["f1"] for r in dec_results) / max(1, total)
        avg_em = sum(r["em"] for r in dec_results) / max(1, total)
        avg_copy = sum(r["copy_rate"] for r in dec_results if r["retrieval_fired"]) / max(1, retrieval_count)

        results_by_decoder[dec] = {
            "retrieval_rate": round(retrieval_rate, 4),
            "retrieval_count": retrieval_count,
            "total_questions": total,
            "mean_f1": round(avg_f1, 4),
            "mean_em": round(avg_em, 4),
            "mean_copy_rate": round(avg_copy, 4),
            "individual_runs": dec_results
        }
        print(f"     Retrieval Rate: {retrieval_rate:.1%} ({retrieval_count}/{total})")
        print(f"     Mean Answer F1: {avg_f1:.4f} | EM: {avg_em:.4f}")

    # Generate Markdown Summary
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"experiment_{timestamp}.json")
    os.makedirs(output_dir, exist_ok=True)

    summary_data = {
        "metadata": {
            "timestamp": timestamp,
            "model_name": model_name,
            "device": device,
            "fixed_tau": fixed_tau,
            "trigger_type": trigger_type
        },
        "results": results_by_decoder
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY TABLE (RQ1a - Fixed Threshold)")
    print("="*60)
    print(f"{'Decoder':<20} | {'Retrieval Rate':<15} | {'Mean F1':<10} | {'Mean EM':<10}")
    print("-" * 60)
    for dec, stats in results_by_decoder.items():
        print(f"{dec:<20} | {stats['retrieval_rate']:<15.2%} | {stats['mean_f1']:<10.4f} | {stats['mean_em']:<10.4f}")
    print("="*60)
    print(f"Full results saved to: {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data", type=str, default="data/sample_qa.jsonl")
    parser.add_argument("--tau", type=float, default=2.5)
    args = parser.parse_args()

    run_experiment(model_name=args.model, data_path=args.data, fixed_tau=args.tau)
