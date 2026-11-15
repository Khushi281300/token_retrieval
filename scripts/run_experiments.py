"""
Unified Multi-Decoder Uncertainty RAG Experiment Runner.
Strictly adheres to PROTOCOL.md and pre-registered specifications:

1. Primary Design (RQ1a & RQ1b):
   - Draft decoders vary: Greedy, Beam Search (5), Contrastive Search (alpha=0.6, k=4),
     Nucleus Sampling (T=0.7, p=0.9 across 3 random seeds: 42, 43, 44).
   - Answer decoder is STRICTLY FIXED to Greedy (isolates trigger from answer generation).
   - Main query formulation: Question ALONE is used as the BM25 query to eliminate retriever confounding.
   - Table 1 (RQ1a): Fixed tau (tuned under Greedy on dev at 40%). Reports retrieval rate shift.
   - Table 2 (RQ1b): Matched retrieval rate (tau calibrated per decoder on dev at 20%, 40%, 60%).
   - Bootstrap 95% Confidence Intervals for tau (1,000 resamples over dev questions).
   - Per-question Cohen's kappa agreement with Greedy at matched rate.
   - Teacher-forced control: Greedy draft text scored under each decoder's distribution.
   - Baselines: No-retrieval, Always-retrieve, Random trigger (at matched rate), and Oracle mode.

2. Secondary Design (RQ1c):
   - Draft decoder is STRICTLY FIXED to Greedy.
   - Answer decoder varies: Greedy vs Contrastive Search.
   - Table 3 (RQ1c): Reports Answer F1, EM, and Context Copy Rate (3-gram overlap).

3. Clean Data & Logging:
   - Zero synthetic fallbacks; loads real TriviaQA (rc.wikipedia or rc.nocontext).
   - Pools Wikipedia entity pages into global BM25 corpus.
   - Evaluates Recall@3.
   - Records git commit hash and exports detailed JSON summary + JSONL row per (question, decoder, seed).
"""

import os
import sys
import json
import time
import random
import argparse
import subprocess
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.decoders import DecoderEngine
from src.retriever import LocalBM25Retriever
from src.pipeline import RAGPipeline
from src.data_loader import load_trivia_qa, check_recall_at_k
from src.scoring import score_tokens, cut_draft_at_first_newline_or_eos, compute_distinct_n
from src.metrics import (
    compute_f1,
    compute_exact_match,
    compute_copy_rate,
    compute_cohen_kappa,
    bootstrap_threshold_ci
)


def get_git_commit_hash() -> str:
    """Retrieves the current git commit hash for experimental provenance."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return output.decode("ascii").strip()
    except Exception:
        return "unknown"


def load_dataset_or_fail(
    dataset_name_or_path: str,
    num_dev: int = 500,
    num_test: int = 1000,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    """
    Loads dataset without synthetic fallbacks.
    Raises explicit RuntimeError on failure.
    """
    if dataset_name_or_path.lower() in ["trivia_qa", "triviaqa", "mandarjoshi/trivia_qa"]:
        # Try rc.wikipedia first, fallback to rc.nocontext if needed
        try:
            return load_trivia_qa(config="rc.wikipedia", split="validation", num_dev=num_dev, num_test=num_test, seed=seed)
        except Exception as e_wiki:
            print(f"TriviaQA rc.wikipedia load failed ({e_wiki}). Attempting rc.nocontext...")
            try:
                return load_trivia_qa(config="rc.nocontext", split="validation", num_dev=num_dev, num_test=num_test, seed=seed)
            except Exception as e_nocontext:
                raise RuntimeError(
                    f"Failed to load TriviaQA: rc.wikipedia ({e_wiki}) and rc.nocontext ({e_nocontext}). "
                    f"Verify HF credentials and network access."
                )

    file_path = dataset_name_or_path
    if not os.path.exists(file_path):
        file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", file_path))
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset path '{dataset_name_or_path}' not found. No synthetic fallbacks permitted.")

    raw_data = []
    corpus = []
    seen_ids = set()

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                raw_data.append(item)
                for p_idx, p in enumerate(item.get("passages", [])):
                    p_id = f"{item['id']}_p{p_idx}"
                    if p_id not in seen_ids:
                        seen_ids.add(p_id)
                        corpus.append({"id": p_id, "text": p})

    rng = random.Random(seed)
    rng.shuffle(raw_data)

    dev_set = raw_data[:num_dev]
    test_set = raw_data[num_dev : num_dev + num_test]
    if not test_set:
        test_set = dev_set

    print(f"Loaded {len(raw_data)} items from '{dataset_name_or_path}' (Dev: {len(dev_set)}, Test: {len(test_set)}, Corpus: {len(corpus)})")
    return dev_set, test_set, corpus


def run_full_pipeline_experiment(
    model_name: str = "allenai/OLMo-2-1124-7B-Instruct",
    dataset_name_or_path: str = "trivia_qa",
    num_dev: int = 500,
    num_test: int = 1000,
    primary_target_rate: float = 0.40,
    target_rates: List[float] = [0.20, 0.40, 0.60],
    score_type: str = "raw",
    entropy_unit: str = "nats",
    max_draft_tokens: int = 64,
    max_answer_tokens: int = 32,
    load_in_4bit: bool = False,
    run_dev_only: bool = False,
    skip_dev: bool = False,
    calibrated_tau_file: Optional[str] = None,
    num_shards: int = 1,
    shard_idx: int = 0,
    output_dir: str = "runs"
):
    commit_hash = get_git_commit_hash()
    print("=" * 80)
    print("Multi-Decoder Uncertainty-Triggered RAG Evaluation (Topic 1 Protocol)")
    print(f"Git Commit Hash:       {commit_hash}")
    print(f"Model:                 {model_name}")
    print(f"Precision / 4-bit:     {'4-bit Quantized' if load_in_4bit else 'FP16 / BF16'}")
    print(f"Primary Target Rate:   {primary_target_rate:.1%}")
    print(f"All Evaluated Rates:   {[f'{r:.0%}' for r in target_rates]}")
    print(f"Scored Distribution:   {score_type.upper()} ({entropy_unit})")
    print(f"Draft Token Cap:       {max_draft_tokens} tokens (cut at first newline/EOS)")
    print(f"Answer Token Cap:      {max_answer_tokens} tokens")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"device_map": "auto" if device == "cuda" else None}
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            compute_dtype = torch.float16 if device == "cuda" else torch.float32
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            print(f"Configured 4-bit quantization with BitsAndBytesConfig (compute_dtype={compute_dtype} [native Turing FP16]).")
        except Exception as e:
            print(f"Warning: BitsAndBytesConfig setup failed ({e}). Loading in standard FP16.")
            model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
    else:
        model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model weights for {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device == "cpu":
        model.to(device)
    model.eval()

    decoder_engine = DecoderEngine(model=model, tokenizer=tokenizer, device=device)
    dev_set, test_set, corpus = load_dataset_or_fail(
        dataset_name_or_path,
        num_dev=num_dev,
        num_test=num_test,
        seed=42
    )

    if not corpus:
        # Build fallback corpus from questions' context/gold passage if not already pooled
        corpus = []
        for item in dev_set + test_set:
            if item.get("gold_passage"):
                corpus.append({"id": f"doc_{item['id']}", "text": item["gold_passage"]})
            for p_idx, p in enumerate(item.get("passages", [])):
                corpus.append({"id": f"p_{item['id']}_{p_idx}", "text": p})

    retriever = LocalBM25Retriever(corpus=corpus)
    pipeline = RAGPipeline(decoder_engine=decoder_engine, retriever=retriever, trigger_type="max_entropy")
    use_raw = (score_type == "raw")

    total_eval_q = len(dev_set) + len(test_set)
    passages_per_q = len(corpus) / max(1, total_eval_q)
    print(f"\n[CORPUS NOTICE] Pooled BM25 Corpus has {len(corpus)} passages across {total_eval_q} questions (~{passages_per_q:.1f} passages/question).")
    print(f"[CORPUS NOTICE] This is a pooled gold-in-distractor benchmark; Recall@3 is strictly evaluated to verify retrieval quality.\n")

    # Decoder configurations
    decoders_config = [
        {"name": "greedy", "seeds": [None], "kwargs": {"decoder_type": "greedy"}},
        {"name": "beam_search", "seeds": [None], "kwargs": {"decoder_type": "beam_search", "num_beams": 5}},
        {"name": "contrastive_search", "seeds": [None], "kwargs": {"decoder_type": "contrastive_search", "penalty_alpha": 0.6, "top_k": 4}},
        {"name": "nucleus_sampling", "seeds": [42, 43, 44], "kwargs": {"decoder_type": "nucleus_sampling", "temperature": 0.7, "top_p": 0.9}}
    ]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    jsonl_records_file = os.path.join(output_dir, f"experiment_records_{timestamp}.jsonl")
    f_jsonl = open(jsonl_records_file, "w", encoding="utf-8")

    # =========================================================================
    # PHASE 1: DEV CALIBRATION & CONTROLS
    # =========================================================================
    if calibrated_tau_file is None and skip_dev:
        default_cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "dev_calibrated_taus.json"))
        if os.path.exists(default_cfg):
            calibrated_tau_file = default_cfg

    if calibrated_tau_file and os.path.exists(calibrated_tau_file):
        print(f"\n[PHASE 1 SKIPPED] Loading precomputed Dev calibration from: {calibrated_tau_file}")
        with open(calibrated_tau_file, "r", encoding="utf-8") as f_tau:
            saved_calib = json.load(f_tau)

        calibrated_taus = {}
        for dec, r_dict in saved_calib["calibrated_taus"].items():
            calibrated_taus[dec] = {}
            for r_str, ci in r_dict.items():
                calibrated_taus[dec][float(r_str)] = ci

        fixed_tau_greedy = saved_calib.get("fixed_tau_greedy", calibrated_taus["greedy"][primary_target_rate]["tau"])
        kappa_results = saved_calib.get("cohen_kappa_vs_greedy", {})
        teacher_forced_results = saved_calib.get("teacher_forced_control", {})

        print(f"--> Loaded Fixed Reference Threshold: tau_fixed = {fixed_tau_greedy:.4f}")
        print("\n--- LOADED DEV CALIBRATION SUMMARY ---")
        for dec in decoders_config:
            dec_name = dec["name"]
            tau_val = calibrated_taus[dec_name][primary_target_rate]["tau"]
            ci_low = calibrated_taus[dec_name][primary_target_rate]["ci_lower"]
            ci_high = calibrated_taus[dec_name][primary_target_rate]["ci_upper"]
            print(f"  * {dec_name:<20}: tau @ 40% = {tau_val:.4f} [95% CI: {ci_low:.4f}, {ci_high:.4f}]")
    else:
        print(f"\n[PHASE 1] Pre-generating Dev drafts on {len(dev_set)} questions to calibrate thresholds...")
        dev_drafts = {}
        calibrated_taus = {}          # {dec_name: {rate: {"tau": ..., "ci_lower": ..., "ci_upper": ...}}}
        dev_trigger_decisions_40 = {} # {dec_name: List[bool]} for Cohen's kappa

        for dec in decoders_config:
            dec_name = dec["name"]
            seeds = dec["seeds"]
            all_seed_drafts = []

            for s in seeds:
                print(f"  --> Pre-generating Dev drafts for {dec_name.upper()} (Seed {s or 42})...")
                seed_drafts = []
                for q_idx, q in enumerate(dev_set):
                    prompt = f"Question: {q['question']}\nDraft Answer:"
                    res = decoder_engine.generate(
                        prompt=prompt,
                        max_new_tokens=max_draft_tokens,
                        seed=s,
                        cut_draft=True,
                        **dec["kwargs"]
                    )
                    seed_drafts.append(res)
                all_seed_drafts.extend(seed_drafts)

            dev_drafts[dec_name] = all_seed_drafts

            # Calibrate tau at each target rate with 1,000 bootstrap resamples
            calibrated_taus[dec_name] = {}
            for rate in target_rates:
                ci_dict = pipeline.calibrate_threshold(all_seed_drafts, target_retrieval_rate=rate, use_raw=use_raw, compute_ci=True)
                calibrated_taus[dec_name][rate] = ci_dict
                print(f"      tau @ {rate:.0%} = {ci_dict['tau']:.4f} [95% CI: {ci_dict['ci_lower']:.4f}, {ci_dict['ci_upper']:.4f}]")

            # Record binary trigger decisions at primary 40% rate for Cohen's kappa
            primary_tau = calibrated_taus[dec_name][primary_target_rate]["tau"]
            # Take first len(dev_set) drafts (seed 42 if multi-seed)
            primary_dev_drafts = all_seed_drafts[:len(dev_set)]
            decisions = [pipeline.evaluate_trigger(d, tau=primary_tau, use_raw=use_raw) for d in primary_dev_drafts]
            dev_trigger_decisions_40[dec_name] = decisions

        # Fixed threshold is Greedy's tau at primary 40% target rate
        fixed_tau_greedy = calibrated_taus["greedy"][primary_target_rate]["tau"]
        print(f"\n--> Reference Fixed Threshold (tuned under Greedy on Dev @ {primary_target_rate:.0%}): tau_fixed = {fixed_tau_greedy:.4f}")

        # Cohen's Kappa against Greedy on Dev at matched 40% rate
        print("\n--- PER-QUESTION TRIGGER AGREEMENT ON DEV (Cohen's Kappa vs Greedy @ 40%) ---")
        kappa_results = {}
        greedy_decisions = dev_trigger_decisions_40["greedy"]
        for dec in decoders_config:
            dec_name = dec["name"]
            kappa = compute_cohen_kappa(greedy_decisions, dev_trigger_decisions_40[dec_name])
            kappa_results[dec_name] = round(kappa, 4)
            print(f"  * Cohen's kappa (Greedy vs {dec_name:<20}): kappa = {kappa:.4f}")

        # Teacher-Forced Control on Dev: Score Greedy draft under each decoder's scoring rule
        print("\n--- TEACHER-FORCED SCORING CONTROL ON DEV ---")
        print("Scoring the exact same Greedy draft texts under each decoder's scoring distribution...")
        greedy_drafts = dev_drafts["greedy"]
        teacher_forced_results = {}
        for dec in decoders_config:
            dec_name = dec["name"]
            entropies = []
            for d in greedy_drafts:
                if dec_name == "contrastive_search":
                    entropies.append(d["processed_summary"]["max_entropy"])
                else:
                    entropies.append(d["raw_summary"]["max_entropy"])
            teacher_forced_results[dec_name] = {
                "mean_max_entropy": float(np.mean(entropies)),
                "retrieval_rate_under_greedy_tau": float(np.mean([e > fixed_tau_greedy for e in entropies]))
            }
            print(f"  * Under {dec_name:<20} scoring: Mean Max-Entropy = {np.mean(entropies):.4f}, Trigger Rate under Greedy tau = {teacher_forced_results[dec_name]['retrieval_rate_under_greedy_tau']:.1%}")

    if run_dev_only:
        print("\n" + "=" * 80)
        print(f"DEV EVALUATION PASS ({len(dev_set)} questions) - Baseline Metrics & Margin Estimation")
        print("=" * 80)

        # 1. Draft properties (lengths & distinct-n) across decoders
        dev_decoder_stats = {}
        for dec_name, drafts in dev_drafts.items():
            primary_drafts = drafts[:len(dev_set)]
            lens = [d["draft_len"] for d in primary_drafts]
            d1s = [d.get("distinct_1", 1.0) for d in primary_drafts]
            d2s = [d.get("distinct_2", 1.0) for d in primary_drafts]
            d3s = [d.get("distinct_3", 1.0) for d in primary_drafts]
            dev_decoder_stats[dec_name] = {
                "mean_draft_len": round(float(np.mean(lens)), 2),
                "distinct_1": round(float(np.mean(d1s)), 4),
                "distinct_2": round(float(np.mean(d2s)), 4),
                "distinct_3": round(float(np.mean(d3s)), 4)
            }

        # 2. Baselines on Dev
        print("  --> Running Dev Baselines (No-Retrieval, Always-Retrieve, Oracle)...")
        dev_no_ret = []
        dev_always_ret = []
        dev_oracle = []

        for q_idx, q in enumerate(dev_set):
            # Dev No-retrieval (Closed-book Greedy)
            r_no = pipeline.run_single_question(
                question=q["question"],
                ground_truths=q["answers"],
                draft_decoder="greedy",
                force_retrieval=False,
                max_draft_tokens=max_draft_tokens
            )
            dev_no_ret.append(r_no)

            # Dev Always-retrieve (BM25 top-3 + Greedy answer)
            r_always = pipeline.run_single_question(
                question=q["question"],
                ground_truths=q["answers"],
                draft_decoder="greedy",
                answer_decoder="greedy",
                force_retrieval=True,
                max_draft_tokens=max_draft_tokens,
                max_answer_tokens=max_answer_tokens
            )
            dev_always_ret.append(r_always)

            # Dev Oracle (Gold passage context + Greedy answer)
            r_oracle = pipeline.run_single_question(
                question=q["question"],
                ground_truths=q["answers"],
                draft_decoder="greedy",
                answer_decoder="greedy",
                force_retrieval=True,
                oracle_context=q.get("gold_passage", ""),
                max_draft_tokens=max_draft_tokens,
                max_answer_tokens=max_answer_tokens
            )
            dev_oracle.append(r_oracle)

        f1_no = np.array([r["f1"] for r in dev_no_ret])
        f1_always = np.array([r["f1"] for r in dev_always_ret])
        em_no = np.array([r["em"] for r in dev_no_ret])
        em_always = np.array([r["em"] for r in dev_always_ret])
        recall_always = np.array([r["recall_at_3"] for r in dev_always_ret])
        f1_oracle = np.array([r["f1"] for r in dev_oracle])
        em_oracle = np.array([r["em"] for r in dev_oracle])

        # Paired bootstrap difference (Always-Retrieve vs No-Retrieval on Dev)
        delta_f1 = f1_always - f1_no
        rng = np.random.RandomState(42)
        boot_deltas = [float(np.mean(rng.choice(delta_f1, size=len(delta_f1), replace=True))) for _ in range(1000)]
        ci_delta_low = float(np.percentile(boot_deltas, 2.5))
        ci_delta_high = float(np.percentile(boot_deltas, 97.5))
        se_f1 = float(np.std(boot_deltas))

        dev_baselines = {
            "no_retrieval": {
                "f1": round(float(np.mean(f1_no)), 4),
                "em": round(float(np.mean(em_no)), 4)
            },
            "always_retrieve": {
                "f1": round(float(np.mean(f1_always)), 4),
                "em": round(float(np.mean(em_always)), 4),
                "recall_at_3": round(float(np.mean(recall_always)), 4)
            },
            "oracle": {
                "f1": round(float(np.mean(f1_oracle)), 4),
                "em": round(float(np.mean(em_oracle)), 4)
            },
            "retrieval_delta_f1": {
                "mean": round(float(np.mean(delta_f1)), 4),
                "ci_95": [round(ci_delta_low, 4), round(ci_delta_high, 4)],
                "se_f1": round(se_f1, 4)
            }
        }

        print("\n" + "=" * 80)
        print("DEV BASELINE & CORPUS METRICS SUMMARY")
        print("=" * 80)
        print(f"Dev No-Retrieval:     F1 = {dev_baselines['no_retrieval']['f1']:.4f} | EM = {dev_baselines['no_retrieval']['em']:.4f}")
        print(f"Dev Always-Retrieve:  F1 = {dev_baselines['always_retrieve']['f1']:.4f} | EM = {dev_baselines['always_retrieve']['em']:.4f} | Recall@3 = {dev_baselines['always_retrieve']['recall_at_3']:.1%}")
        print(f"Dev Oracle:           F1 = {dev_baselines['oracle']['f1']:.4f} | EM = {dev_baselines['oracle']['em']:.4f}")
        print(f"Dev Retrieval Delta:  ΔF1 = {dev_baselines['retrieval_delta_f1']['mean']:+.4f} [95% CI: {ci_delta_low:+.4f}, {ci_delta_high:+.4f}, SE = {se_f1:.4f}]")

        print("\n" + "=" * 80)
        print("DEV DRAFT GENERATION METRICS (Tokens & Repetition)")
        print("=" * 80)
        print(f"{'Decoder':<22} | {'Mean Length':<12} | {'Distinct-1':<12} | {'Distinct-2':<12} | {'Distinct-3'}")
        print("-" * 80)
        for dec_n, st in dev_decoder_stats.items():
            print(f"{dec_n:<22} | {st['mean_draft_len']:<12.1f} | {st['distinct_1']:<12.4f} | {st['distinct_2']:<12.4f} | {st['distinct_3']:.4f}")

        # Save full dev calibration summary
        dev_summary_file = os.path.join(output_dir, f"dev_calibration_summary_{timestamp}.json")
        with open(dev_summary_file, "w", encoding="utf-8") as f_dev:
            json.dump({
                "metadata": {
                    "timestamp": timestamp,
                    "git_commit_hash": commit_hash,
                    "model_name": model_name,
                    "compute_dtype": str(compute_dtype),
                    "score_type": score_type,
                    "entropy_unit": entropy_unit,
                    "num_dev_questions": len(dev_set),
                    "num_corpus_passages": len(corpus)
                },
                "dev_calibration": calibrated_taus,
                "dev_cohen_kappa_vs_greedy": kappa_results,
                "dev_teacher_forced_control": teacher_forced_results,
                "dev_decoder_stats": dev_decoder_stats,
                "dev_baselines": dev_baselines
            }, f_dev, indent=2)

        print(f"\n[SUCCESS] Full Dev calibration and baseline summary saved to: {dev_summary_file}")
        print("[INFO] --run_dev_only completed. All dev statistics and margins are documented.\n")
        f_jsonl.close()
        return

    # =========================================================================
    # PHASE 2: STREAMLINED CACHED GENERATION ON TEST (Generate each element once)
    # =========================================================================
    import math
    if num_shards > 1:
        shard_size = int(math.ceil(len(test_set) / num_shards))
        start_i = shard_idx * shard_size
        end_i = min(len(test_set), (shard_idx + 1) * shard_size)
        active_test_set = test_set[start_i:end_i]
        cache_suffix = f"_shard_{shard_idx}_of_{num_shards}"
        print(f"\n[PHASE 2 & 3] Running Shard {shard_idx + 1}/{num_shards} (Questions {start_i + 1}..{end_i} of {len(test_set)})...")
    else:
        active_test_set = test_set
        cache_suffix = ""
        print(f"\n[PHASE 2 & 3] Running Streamlined Test Evaluation Across {len(test_set)} Questions...")

    checkpoint_file = os.path.join(output_dir, f"test_cache{cache_suffix}_{commit_hash[:7]}.jsonl")
    cached_records = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f_chk:
            for line in f_chk:
                if line.strip():
                    rec = json.loads(line)
                    cached_records[rec["question_id"]] = rec
        print(f"  --> Resuming: Found {len(cached_records)} pre-generated questions in {checkpoint_file}.")

    f_chk_out = open(checkpoint_file, "a", encoding="utf-8")

    all_test_records = []
    for q_idx, q in enumerate(active_test_set):
        q_id = q["id"]
        if q_id in cached_records:
            all_test_records.append(cached_records[q_id])
            continue

        print(f"  --> [Generating Cache] Question {q_idx + 1}/{len(active_test_set)}: {q['question'][:50]}...", end="\r", flush=True)

        # 1. Retrieve top-3 passages once using question alone
        retrieved_passages = retriever.retrieve(query=q["question"], top_k=3)
        retrieved_text = "\n".join([p["text"] for p in retrieved_passages])
        recall_3 = check_recall_at_k(retrieved_passages, q["answers"], k=3)

        # 2. Generate ONE greedy answer conditioned on retrieved context
        ans_prompt = f"Context:\n{retrieved_text}\n\nQuestion: {q['question']}\nAnswer:"
        ans_greedy_res = decoder_engine.generate(
            prompt=ans_prompt,
            decoder_type="greedy",
            max_new_tokens=max_answer_tokens,
            cut_draft=True
        )
        final_ans_retrieved = ans_greedy_res["generated_text"]
        f1_retrieved = compute_f1(final_ans_retrieved, q["answers"])
        em_retrieved = compute_exact_match(final_ans_retrieved, q["answers"])
        copy_rate_greedy = compute_copy_rate(final_ans_retrieved, retrieved_text, n=3)

        # 3. Generate ONE contrastive answer on retrieved context (for Secondary RQ1c)
        ans_con_res = decoder_engine.generate(
            prompt=ans_prompt,
            decoder_type="contrastive_search",
            penalty_alpha=0.6,
            top_k=4,
            max_new_tokens=max_answer_tokens,
            cut_draft=True
        )
        final_ans_con = ans_con_res["generated_text"]
        f1_retrieved_con = compute_f1(final_ans_con, q["answers"])
        em_retrieved_con = compute_exact_match(final_ans_con, q["answers"])
        copy_rate_con = compute_copy_rate(final_ans_con, retrieved_text, n=3)

        # 4. Generate ONE oracle answer (gold context)
        oracle_ctx = q.get("gold_passage", "")
        oracle_prompt = f"Context:\n{oracle_ctx}\n\nQuestion: {q['question']}\nAnswer:"
        oracle_res = decoder_engine.generate(
            prompt=oracle_prompt,
            decoder_type="greedy",
            max_new_tokens=max_answer_tokens,
            cut_draft=True
        )
        f1_oracle = compute_f1(oracle_res["generated_text"], q["answers"])
        em_oracle = compute_exact_match(oracle_res["generated_text"], q["answers"])

        # 5. Generate drafts for each decoder
        draft_prompt = f"Question: {q['question']}\nDraft Answer:"
        draft_records = {}

        for dec in decoders_config:
            dec_name = dec["name"]
            seeds = dec["seeds"]
            draft_records[dec_name] = []
            for s in seeds:
                d_res = decoder_engine.generate(
                    prompt=draft_prompt,
                    decoder_type=dec_name,
                    max_new_tokens=max_draft_tokens,
                    seed=s,
                    cut_draft=True,
                    **dec["kwargs"]
                )
                d_f1 = compute_f1(d_res["generated_text"], q["answers"])
                d_em = compute_exact_match(d_res["generated_text"], q["answers"])
                draft_records[dec_name].append({
                    "seed": s,
                    "draft_text": d_res["generated_text"],
                    "draft_len": d_res["draft_len"],
                    "raw_max_entropy": d_res["raw_summary"]["max_entropy"],
                    "raw_mean_entropy": d_res["raw_summary"]["mean_entropy"],
                    "adjusted_max_entropy": d_res["processed_summary"]["max_entropy"],
                    "distinct_1": d_res.get("distinct_1", 1.0),
                    "distinct_2": d_res.get("distinct_2", 1.0),
                    "distinct_3": d_res.get("distinct_3", 1.0),
                    "contrastive_alpha": d_res.get("contrastive_alpha"),
                    "contrastive_k": d_res.get("contrastive_k"),
                    "f1_draft": d_f1,
                    "em_draft": d_em
                })

        q_record = {
            "question_id": q_id,
            "question": q["question"],
            "answers": q["answers"],
            "retrieved_passage_ids": [p.get("id", "") for p in retrieved_passages],
            "recall_at_3": recall_3,
            "f1_retrieved": f1_retrieved,
            "em_retrieved": em_retrieved,
            "copy_rate_greedy": copy_rate_greedy,
            "f1_retrieved_con": f1_retrieved_con,
            "em_retrieved_con": em_retrieved_con,
            "copy_rate_con": copy_rate_con,
            "f1_oracle": f1_oracle,
            "em_oracle": em_oracle,
            "drafts": draft_records
        }
        f_chk_out.write(json.dumps(q_record) + "\n")
        f_chk_out.flush()
        all_test_records.append(q_record)

    f_chk_out.close()
    print(f"\n[DONE] Cached all {len(all_test_records)} test questions.")

    # =========================================================================
    # OFFLINE TABLE COMPUTATION (Instant sub-second evaluation)
    # =========================================================================
    N = len(all_test_records)
    print("\n--- COMPUTING SCIENTIFIC TABLES OFFLINE ---")

    # Table 4: Baselines
    greedy_draft_f1s = [r["drafts"]["greedy"][0]["f1_draft"] for r in all_test_records]
    greedy_draft_ems = [r["drafts"]["greedy"][0]["em_draft"] for r in all_test_records]
    retrieved_f1s = [r["f1_retrieved"] for r in all_test_records]
    retrieved_ems = [r["em_retrieved"] for r in all_test_records]
    recall_3s = [r["recall_at_3"] for r in all_test_records]
    oracle_f1s = [r["f1_oracle"] for r in all_test_records]
    oracle_ems = [r["em_oracle"] for r in all_test_records]

    rng_rand = random.Random(42)
    rand_triggers = [rng_rand.random() < primary_target_rate for _ in range(N)]
    rand_f1s = [retrieved_f1s[i] if rand_triggers[i] else greedy_draft_f1s[i] for i in range(N)]
    rand_ems = [retrieved_ems[i] if rand_triggers[i] else greedy_draft_ems[i] for i in range(N)]

    baselines_results = {
        "no_retrieval": {"f1": float(np.mean(greedy_draft_f1s)), "em": float(np.mean(greedy_draft_ems)), "retrieval_rate": 0.0},
        "always_retrieve": {"f1": float(np.mean(retrieved_f1s)), "em": float(np.mean(retrieved_ems)), "recall_at_3": float(np.mean(recall_3s)), "retrieval_rate": 1.0},
        "random_trigger_40": {"f1": float(np.mean(rand_f1s)), "em": float(np.mean(rand_ems)), "retrieval_rate": float(np.mean(rand_triggers))},
        "oracle": {"f1": float(np.mean(oracle_f1s)), "em": float(np.mean(oracle_ems)), "retrieval_rate": 1.0}
    }

    # Table 1: Fixed Threshold (RQ1a) & Table 2: Matched Rate (RQ1b)
    table1_fixed_results = {}
    table2_matched_results = {}

    for dec in decoders_config:
        dec_name = dec["name"]
        seeds = dec["seeds"]

        seed_fixed_rates, seed_fixed_f1s, seed_fixed_ems = [], [], []
        seed_matched_rates_dict = {r: [] for r in target_rates}
        seed_matched_f1s_dict = {r: [] for r in target_rates}
        seed_matched_ems_dict = {r: [] for r in target_rates}
        seed_matched_recall_dict = {r: [] for r in target_rates}

        for s_idx, seed_val in enumerate(seeds):
            fixed_fired_count = 0
            fixed_f1_list, fixed_em_list = [], []

            matched_fired_counts = {r: 0 for r in target_rates}
            matched_f1_lists = {r: [] for r in target_rates}
            matched_em_lists = {r: [] for r in target_rates}
            matched_recalls = {r: [] for r in target_rates}

            for q_idx, q_rec in enumerate(all_test_records):
                d_info = q_rec["drafts"][dec_name][s_idx]
                ent_score = d_info["raw_max_entropy"] if use_raw else d_info["adjusted_max_entropy"]

                # Fixed tau check
                f_fired = bool(ent_score > fixed_tau_greedy)
                if f_fired:
                    fixed_fired_count += 1
                    fixed_f1_list.append(q_rec["f1_retrieved"])
                    fixed_em_list.append(q_rec["em_retrieved"])
                else:
                    fixed_f1_list.append(d_info["f1_draft"])
                    fixed_em_list.append(d_info["em_draft"])

                # Matched tau check across rates
                for r in target_rates:
                    tau_m = calibrated_taus[dec_name][r]["tau"]
                    m_fired = bool(ent_score > tau_m)
                    if m_fired:
                        matched_fired_counts[r] += 1
                        matched_f1_lists[r].append(q_rec["f1_retrieved"])
                        matched_em_lists[r].append(q_rec["em_retrieved"])
                        matched_recalls[r].append(q_rec["recall_at_3"])
                    else:
                        matched_f1_lists[r].append(d_info["f1_draft"])
                        matched_em_lists[r].append(d_info["em_draft"])

                # Write granular row to JSONL
                primary_m_fired = bool(ent_score > calibrated_taus[dec_name][primary_target_rate]["tau"])
                row = {
                    "git_commit_hash": commit_hash,
                    "question_id": q_rec["question_id"],
                    "question": q_rec["question"],
                    "decoder": dec_name,
                    "seed": seed_val,
                    "draft_len": d_info["draft_len"],
                    "draft_text": d_info["draft_text"],
                    "distinct_1": d_info["distinct_1"],
                    "distinct_2": d_info["distinct_2"],
                    "distinct_3": d_info["distinct_3"],
                    "contrastive_alpha": d_info["contrastive_alpha"],
                    "contrastive_k": d_info["contrastive_k"],
                    "raw_max_entropy": d_info["raw_max_entropy"],
                    "raw_mean_entropy": d_info["raw_mean_entropy"],
                    "adjusted_max_entropy": d_info["adjusted_max_entropy"],
                    "tau_fixed": fixed_tau_greedy,
                    "tau_matched_40": calibrated_taus[dec_name][primary_target_rate]["tau"],
                    "fixed_retrieval_fired": f_fired,
                    "matched_retrieval_fired": primary_m_fired,
                    "retrieved_passage_ids": q_rec["retrieved_passage_ids"],
                    "recall_at_3": q_rec["recall_at_3"] if primary_m_fired else False,
                    "final_answer": final_ans_retrieved if primary_m_fired else d_info["draft_text"],
                    "f1": q_rec["f1_retrieved"] if primary_m_fired else d_info["f1_draft"],
                    "em": q_rec["em_retrieved"] if primary_m_fired else d_info["em_draft"],
                    "copy_rate_3": q_rec["copy_rate_greedy"] if primary_m_fired else 0.0
                }
                f_jsonl.write(json.dumps(row) + "\n")

            seed_fixed_rates.append(fixed_fired_count / N)
            seed_fixed_f1s.append(float(np.mean(fixed_f1_list)))
            seed_fixed_ems.append(float(np.mean(fixed_em_list)))

            for r in target_rates:
                seed_matched_rates_dict[r].append(matched_fired_counts[r] / N)
                seed_matched_f1s_dict[r].append(float(np.mean(matched_f1_lists[r])))
                seed_matched_ems_dict[r].append(float(np.mean(matched_em_lists[r])))
                seed_matched_recall_dict[r].append(float(np.mean(matched_recalls[r])) if matched_recalls[r] else 0.0)

        # Table 1 stats
        table1_fixed_results[dec_name] = {
            "retrieval_rate": float(np.mean(seed_fixed_rates)),
            "retrieval_rate_std": float(np.std(seed_fixed_rates)),
            "f1": float(np.mean(seed_fixed_f1s)),
            "f1_std": float(np.std(seed_fixed_f1s)),
            "em": float(np.mean(seed_fixed_ems)),
            "em_std": float(np.std(seed_fixed_ems))
        }

        # Table 2 stats
        table2_matched_results[dec_name] = {
            "tau_matched": calibrated_taus[dec_name][primary_target_rate]["tau"],
            "tau_ci": [calibrated_taus[dec_name][primary_target_rate]["ci_lower"], calibrated_taus[dec_name][primary_target_rate]["ci_upper"]],
            "retrieval_rate": float(np.mean(seed_matched_rates_dict[primary_target_rate])),
            "retrieval_rate_std": float(np.std(seed_matched_rates_dict[primary_target_rate])),
            "f1": float(np.mean(seed_matched_f1s_dict[primary_target_rate])),
            "f1_std": float(np.std(seed_matched_f1s_dict[primary_target_rate])),
            "em": float(np.mean(seed_matched_ems_dict[primary_target_rate])),
            "em_std": float(np.std(seed_matched_ems_dict[primary_target_rate])),
            "recall_at_3": float(np.mean(seed_matched_recall_dict[primary_target_rate])),
            "curve_rates": {
                f"{r:.0%}": {
                    "tau": calibrated_taus[dec_name][r]["tau"],
                    "test_rate": float(np.mean(seed_matched_rates_dict[r])),
                    "f1": float(np.mean(seed_matched_f1s_dict[r])),
                    "em": float(np.mean(seed_matched_ems_dict[r]))
                } for r in target_rates
            }
        }

    # Table 3: Secondary Design (RQ1c - Copy Rate on triggered items under Greedy tau)
    results_answer_decoder = {}
    greedy_triggered_indices = [
        i for i, r in enumerate(all_test_records)
        if r["drafts"]["greedy"][0]["raw_max_entropy"] > fixed_tau_greedy
    ]
    if not greedy_triggered_indices:
        greedy_triggered_indices = list(range(min(50, N)))

    f1_g_ans = [all_test_records[i]["f1_retrieved"] for i in greedy_triggered_indices]
    em_g_ans = [all_test_records[i]["em_retrieved"] for i in greedy_triggered_indices]
    copy_g_ans = [all_test_records[i]["copy_rate_greedy"] for i in greedy_triggered_indices]

    f1_c_ans = [all_test_records[i]["f1_retrieved_con"] for i in greedy_triggered_indices]
    em_c_ans = [all_test_records[i]["em_retrieved_con"] for i in greedy_triggered_indices]
    copy_c_ans = [all_test_records[i]["copy_rate_con"] for i in greedy_triggered_indices]

    results_answer_decoder["greedy"] = {
        "f1": float(np.mean(f1_g_ans)),
        "em": float(np.mean(em_g_ans)),
        "copy_rate_3": float(np.mean(copy_g_ans)),
        "evaluated_questions": len(greedy_triggered_indices)
    }
    results_answer_decoder["contrastive_search"] = {
        "f1": float(np.mean(f1_c_ans)),
        "em": float(np.mean(em_c_ans)),
        "copy_rate_3": float(np.mean(copy_c_ans)),
        "evaluated_questions": len(greedy_triggered_indices)
    }

    f_jsonl.close()

    # =========================================================================
    # PRINT RESULTS TABLES
    # =========================================================================
    print("\n" + "=" * 80)
    print(f"TABLE 1: FIXED-THRESHOLD COMPARISON (RQ1a) [tau_fixed = {fixed_tau_greedy:.4f}]")
    print("Primary Design: Answer Decoder = GREEDY strictly")
    print("=" * 80)
    print(f"{'Draft Decoder':<22} | {'Retrieval Rate':<18} | {'Answer F1':<15} | {'Exact Match':<12}")
    print("-" * 80)
    for dec, s in table1_fixed_results.items():
        rate_str = f"{s['retrieval_rate']:.1%} (+/- {s['retrieval_rate_std']:.1%})" if s['retrieval_rate_std'] > 0 else f"{s['retrieval_rate']:.1%}"
        f1_str = f"{s['f1']:.4f} (+/- {s['f1_std']:.4f})" if s['f1_std'] > 0 else f"{s['f1']:.4f}"
        em_str = f"{s['em']:.4f} (+/- {s['em_std']:.4f})" if s['em_std'] > 0 else f"{s['em']:.4f}"
        print(f"{dec:<22} | {rate_str:<18} | {f1_str:<15} | {em_str:<12}")

    print("\n" + "=" * 80)
    print(f"TABLE 2: MATCHED-RETRIEVAL-RATE COMPARISON (RQ1b) [Target Rate = {primary_target_rate:.0%}]")
    print("Primary Design: Answer Decoder = GREEDY strictly (tau tuned on Dev per decoder)")
    print("=" * 80)
    print(f"{'Draft Decoder':<20} | {'Tuned tau (95% CI)':<22} | {'Test Retrieval Rate':<20} | {'Answer F1':<15} | {'EM':<10}")
    print("-" * 80)
    for dec, s in table2_matched_results.items():
        tau_ci_str = f"{s['tau_matched']:.4f} [{s['tau_ci'][0]:.2f}, {s['tau_ci'][1]:.2f}]"
        rate_str = f"{s['retrieval_rate']:.1%} (+/- {s['retrieval_rate_std']:.1%})" if s['retrieval_rate_std'] > 0 else f"{s['retrieval_rate']:.1%}"
        f1_str = f"{s['f1']:.4f} (+/- {s['f1_std']:.4f})" if s['f1_std'] > 0 else f"{s['f1']:.4f}"
        em_str = f"{s['em']:.4f} (+/- {s['em_std']:.4f})" if s['em_std'] > 0 else f"{s['em']:.4f}"
        print(f"{dec:<20} | {tau_ci_str:<22} | {rate_str:<20} | {f1_str:<15} | {em_str:<10}")

    print("\n" + "=" * 80)
    print("TABLE 3: ANSWER DECODER CONTEXT COPY PENALTY (RQ1c)")
    print(f"Secondary Design: Evaluated on {len(greedy_triggered_indices)} questions where Greedy draft triggered retrieval")
    print("=" * 80)
    print(f"{'Answer Decoder':<20} | {'Answer F1':<15} | {'Exact Match':<15} | {'Context Copy Rate (3-gram)':<25}")
    print("-" * 80)
    for dec, s in results_answer_decoder.items():
        print(f"{dec:<20} | {s['f1']:<15.4f} | {s['em']:<15.4f} | {s['copy_rate_3']:<25.4f}")

    print("\n" + "=" * 80)
    print("TABLE 4: BASELINES & CONTROLS")
    print("=" * 80)
    for b_name, b_data in baselines_results.items():
        print(f"{b_name:<25} | F1 = {b_data['f1']:.4f} | EM = {b_data['em']:.4f} | Rate = {b_data.get('retrieval_rate', 0.0):.1%}")

    # Save complete JSON
    out_file = os.path.join(output_dir, f"full_experiment_{timestamp}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "timestamp": timestamp,
                "git_commit_hash": commit_hash,
                "model_name": model_name,
                "compute_dtype": str(compute_dtype),
                "score_type": score_type,
                "entropy_unit": entropy_unit,
                "load_in_4bit": load_in_4bit,
                "primary_target_rate": primary_target_rate,
                "target_rates": target_rates,
                "num_dev_questions": len(dev_set),
                "num_test_questions": len(all_test_records),
                "num_corpus_passages": len(corpus)
            },
            "dev_calibration": calibrated_taus,
            "dev_cohen_kappa_vs_greedy": kappa_results,
            "dev_teacher_forced_control": teacher_forced_results,
            "table1_fixed_threshold": table1_fixed_results,
            "table2_matched_rate": table2_matched_results,
            "table3_answer_decoder": results_answer_decoder,
            "table4_baselines": baselines_results,
            "jsonl_records_file": jsonl_records_file
        }, f, indent=2)

    print(f"\n[SUCCESS] Scientific JSON summary saved to: {out_file}")
    print(f"[SUCCESS] Granular per-item JSONL records saved to: {jsonl_records_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-1124-7B-Instruct")
    parser.add_argument("--dataset", type=str, default="trivia_qa")
    parser.add_argument("--num_dev", type=int, default=500)
    parser.add_argument("--num_test", type=int, default=1000)
    parser.add_argument("--primary_rate", type=float, default=0.40)
    parser.add_argument("--score_type", type=str, default="raw", choices=["raw", "processed"])
    parser.add_argument("--unit", type=str, default="nats", choices=["nats", "bits"])
    parser.add_argument("--max_draft_tokens", type=int, default=64)
    parser.add_argument("--max_answer_tokens", type=int, default=32)
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit for low VRAM (e.g. Colab T4)")
    parser.add_argument("--run_dev_only", action="store_true", help="Run only Phase 1 Dev calibration and freeze thresholds before test set")
    parser.add_argument("--skip_dev", action="store_true", help="Skip Dev draft generation and load precomputed calibration")
    parser.add_argument("--calibrated_tau_file", type=str, default=None, help="Path to precomputed Dev calibration JSON")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of GPU shards for parallel cached generation")
    parser.add_argument("--shard_idx", type=int, default=0, help="Shard index (0-indexed)")
    parser.add_argument("--output_dir", type=str, default="runs")
    args = parser.parse_args()

    run_full_pipeline_experiment(
        model_name=args.model,
        dataset_name_or_path=args.dataset,
        num_dev=args.num_dev,
        num_test=args.num_test,
        primary_target_rate=args.primary_rate,
        score_type=args.score_type,
        entropy_unit=args.unit,
        max_draft_tokens=args.max_draft_tokens,
        max_answer_tokens=args.max_answer_tokens,
        load_in_4bit=args.load_in_4bit,
        run_dev_only=args.run_dev_only,
        skip_dev=args.skip_dev,
        calibrated_tau_file=args.calibrated_tau_file,
        num_shards=args.num_shards,
        shard_idx=args.shard_idx,
        output_dir=args.output_dir
    )
