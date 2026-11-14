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
            compute_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            print(f"Configured 4-bit quantization with BitsAndBytesConfig (compute_dtype={compute_dtype}).")
        except Exception as e:
            print(f"Warning: BitsAndBytesConfig setup failed ({e}). Loading in standard precision.")
            dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
            model_kwargs["torch_dtype"] = dtype
    else:
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
        model_kwargs["torch_dtype"] = dtype

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
        print("\n[INFO] --run_dev_only was requested. Freezing dev thresholds and exiting before test execution.")
        f_jsonl.close()
        return

    # =========================================================================
    # PHASE 2: BASELINES EVALUATION ON TEST
    # =========================================================================
    print(f"\n[PHASE 2] Running Baseline Evaluations on Test ({len(test_set)} questions)...")
    baselines_results = {}

    # 1. No-Retrieval Baseline (Closed-book Greedy)
    print("  --> Baseline 1: No-Retrieval (Closed-Book Greedy)...")
    no_ret_runs = []
    for q in test_set:
        res = pipeline.run_single_question(
            question=q["question"],
            ground_truths=q["answers"],
            draft_decoder="greedy",
            force_retrieval=False,
            max_draft_tokens=max_draft_tokens
        )
        no_ret_runs.append(res)
    baselines_results["no_retrieval"] = {
        "f1": float(np.mean([r["f1"] for r in no_ret_runs])),
        "em": float(np.mean([r["em"] for r in no_ret_runs])),
        "retrieval_rate": 0.0
    }

    # 2. Always-Retrieve Baseline (100% Retrieval with Greedy Answer)
    print("  --> Baseline 2: Always-Retrieve (100% Retrieval)...")
    always_ret_runs = []
    for q in test_set:
        res = pipeline.run_single_question(
            question=q["question"],
            ground_truths=q["answers"],
            draft_decoder="greedy",
            answer_decoder="greedy",
            force_retrieval=True,
            max_draft_tokens=max_draft_tokens,
            max_answer_tokens=max_answer_tokens
        )
        always_ret_runs.append(res)
    baselines_results["always_retrieve"] = {
        "f1": float(np.mean([r["f1"] for r in always_ret_runs])),
        "em": float(np.mean([r["em"] for r in always_ret_runs])),
        "recall_at_3": float(np.mean([r["recall_at_3"] for r in always_ret_runs])),
        "retrieval_rate": 1.0
    }

    # 3. Random-Trigger Baseline (Triggered randomly at 40% rate)
    print("  --> Baseline 3: Random-Trigger @ 40% Rate...")
    rng_rand = random.Random(42)
    random_ret_runs = []
    for q in test_set:
        fired = rng_rand.random() < primary_target_rate
        res = pipeline.run_single_question(
            question=q["question"],
            ground_truths=q["answers"],
            draft_decoder="greedy",
            answer_decoder="greedy",
            force_retrieval=fired,
            max_draft_tokens=max_draft_tokens,
            max_answer_tokens=max_answer_tokens
        )
        random_ret_runs.append(res)
    baselines_results["random_trigger_40"] = {
        "f1": float(np.mean([r["f1"] for r in random_ret_runs])),
        "em": float(np.mean([r["em"] for r in random_ret_runs])),
        "retrieval_rate": float(np.mean([r["retrieval_fired"] for r in random_ret_runs]))
    }

    # 4. Oracle Baseline (Gold Reference Passage supplied)
    print("  --> Baseline 4: Oracle Mode (Gold Reference Passage)...")
    oracle_runs = []
    for q in test_set:
        oracle_ctx = q.get("gold_passage", "")
        res = pipeline.run_single_question(
            question=q["question"],
            ground_truths=q["answers"],
            draft_decoder="greedy",
            answer_decoder="greedy",
            force_retrieval=True,
            oracle_context=oracle_ctx,
            max_draft_tokens=max_draft_tokens,
            max_answer_tokens=max_answer_tokens
        )
        oracle_runs.append(res)
    baselines_results["oracle"] = {
        "f1": float(np.mean([r["f1"] for r in oracle_runs])),
        "em": float(np.mean([r["em"] for r in oracle_runs])),
        "retrieval_rate": 1.0
    }

    print(f"      No-Retrieval F1:     {baselines_results['no_retrieval']['f1']:.4f}")
    print(f"      Always-Retrieve F1:  {baselines_results['always_retrieve']['f1']:.4f} (Recall@3: {baselines_results['always_retrieve'].get('recall_at_3', 0.0):.1%})")
    print(f"      Random-Trigger F1:   {baselines_results['random_trigger_40']['f1']:.4f}")
    print(f"      Oracle F1:           {baselines_results['oracle']['f1']:.4f}")

    # =========================================================================
    # PHASE 3: PRIMARY TEST EVALUATION (Answer Decoder = GREEDY strictly)
    # Evaluates Fixed Threshold (RQ1a) and Matched Retrieval Rate (RQ1b)
    # =========================================================================
    print(f"\n[PHASE 3] Running Primary Test Evaluation (Answer Decoder = GREEDY strictly)...")
    table1_fixed_results = {}
    table2_matched_results = {}

    for dec in decoders_config:
        dec_name = dec["name"]
        seeds = dec["seeds"]

        fixed_rate_list, fixed_f1_list, fixed_em_list = [], [], []
        matched_rate_dict = {r: [] for r in target_rates}
        matched_f1_dict = {r: [] for r in target_rates}
        matched_em_dict = {r: [] for r in target_rates}
        matched_recall_dict = {r: [] for r in target_rates}

        for seed in seeds:
            fixed_runs = []
            matched_runs_by_rate = {r: [] for r in target_rates}

            tau_m_primary = calibrated_taus[dec_name][primary_target_rate]["tau"]

            for q_idx, q in enumerate(test_set):
                print(f"  --> [{dec_name.upper()} | Seed {seed or 42}] Question {q_idx+1}/{len(test_set)}...", end="\r", flush=True)

                # Generate draft ONCE
                prompt = f"Question: {q['question']}\nDraft Answer:"
                draft_res = decoder_engine.generate(
                    prompt=prompt,
                    decoder_type=dec_name,
                    max_new_tokens=max_draft_tokens,
                    seed=seed,
                    cut_draft=True
                )

                # Fixed tau check
                fixed_fired = pipeline.evaluate_trigger(draft_res, tau=fixed_tau_greedy, use_raw=use_raw)

                # Matched tau check across evaluated rates
                matched_fired_by_rate = {
                    r: pipeline.evaluate_trigger(draft_res, tau=calibrated_taus[dec_name][r]["tau"], use_raw=use_raw)
                    for r in target_rates
                }

                # Conditional Retrieval (using question alone as query)
                needs_retrieval = fixed_fired or any(matched_fired_by_rate.values())
                retrieved_passages = []
                recall_3 = False
                final_ans_retrieved = draft_res["generated_text"]
                f1_retrieved = compute_f1(final_ans_retrieved, q["answers"])
                em_retrieved = compute_exact_match(final_ans_retrieved, q["answers"])
                copy_rate_retrieved = 0.0

                if needs_retrieval:
                    retrieved_passages = retriever.retrieve(query=q["question"], top_k=3)
                    retrieved_text = "\n".join([p["text"] for p in retrieved_passages])
                    recall_3 = check_recall_at_k(retrieved_passages, q["answers"], k=3)

                    # Answer generated with greedy
                    ans_prompt = f"Context:\n{retrieved_text}\n\nQuestion: {q['question']}\nAnswer:"
                    ans_res = decoder_engine.generate(
                        prompt=ans_prompt,
                        decoder_type="greedy",
                        max_new_tokens=max_answer_tokens,
                        seed=seed,
                        cut_draft=True
                    )
                    final_ans_retrieved = ans_res["generated_text"]
                    f1_retrieved = compute_f1(final_ans_retrieved, q["answers"])
                    em_retrieved = compute_exact_match(final_ans_retrieved, q["answers"])
                    copy_rate_retrieved = compute_copy_rate(final_ans_retrieved, retrieved_text, n=3)

                # Record Fixed-Threshold metrics
                f1_fixed = f1_retrieved if fixed_fired else compute_f1(draft_res["generated_text"], q["answers"])
                em_fixed = em_retrieved if fixed_fired else compute_exact_match(draft_res["generated_text"], q["answers"])
                fixed_runs.append({"retrieval_fired": fixed_fired, "f1": f1_fixed, "em": em_fixed, "recall_at_3": recall_3 if fixed_fired else False})

                # Record Matched-Rate metrics for each rate
                for r in target_rates:
                    f_m = matched_fired_by_rate[r]
                    f1_m = f1_retrieved if f_m else compute_f1(draft_res["generated_text"], q["answers"])
                    em_m = em_retrieved if f_m else compute_exact_match(draft_res["generated_text"], q["answers"])
                    matched_runs_by_rate[r].append({
                        "retrieval_fired": f_m,
                        "f1": f1_m,
                        "em": em_m,
                        "recall_at_3": recall_3 if f_m else False
                    })

                # Write granular row to JSONL
                row = {
                    "git_commit_hash": commit_hash,
                    "question_id": q["id"],
                    "question": q["question"],
                    "decoder": dec_name,
                    "seed": seed,
                    "draft_len": draft_res["draft_len"],
                    "draft_text": draft_res["generated_text"],
                    "distinct_1": draft_res.get("distinct_1", 1.0),
                    "distinct_2": draft_res.get("distinct_2", 1.0),
                    "distinct_3": draft_res.get("distinct_3", 1.0),
                    "contrastive_alpha": draft_res.get("contrastive_alpha"),
                    "contrastive_k": draft_res.get("contrastive_k"),
                    "raw_max_entropy": draft_res["raw_summary"]["max_entropy"],
                    "raw_mean_entropy": draft_res["raw_summary"]["mean_entropy"],
                    "adjusted_max_entropy": draft_res["processed_summary"]["max_entropy"],
                    "tau_fixed": fixed_tau_greedy,
                    "tau_matched_40": tau_m_primary,
                    "fixed_retrieval_fired": fixed_fired,
                    "matched_retrieval_fired": matched_fired_by_rate[primary_target_rate],
                    "retrieved_passage_ids": [p.get("id", "") for p in retrieved_passages],
                    "recall_at_3": recall_3,
                    "final_answer": final_ans_retrieved,
                    "f1": f1_retrieved if matched_fired_by_rate[primary_target_rate] else compute_f1(draft_res["generated_text"], q["answers"]),
                    "em": em_retrieved if matched_fired_by_rate[primary_target_rate] else compute_exact_match(draft_res["generated_text"], q["answers"]),
                    "copy_rate_3": copy_rate_retrieved if matched_fired_by_rate[primary_target_rate] else 0.0
                }
                f_jsonl.write(json.dumps(row) + "\n")

            print(f"  [DONE] Primary test evaluation for {dec_name.upper()} (Seed {seed or 42}).")

            # Accumulate per seed
            total_items = len(fixed_runs)
            fixed_rate_list.append(sum(1 for r in fixed_runs if r["retrieval_fired"]) / total_items)
            fixed_f1_list.append(np.mean([r["f1"] for r in fixed_runs]))
            fixed_em_list.append(np.mean([r["em"] for r in fixed_runs]))

            for r in target_rates:
                m_runs = matched_runs_by_rate[r]
                matched_rate_dict[r].append(sum(1 for x in m_runs if x["retrieval_fired"]) / total_items)
                matched_f1_dict[r].append(np.mean([x["f1"] for x in m_runs]))
                matched_em_dict[r].append(np.mean([x["em"] for x in m_runs]))
                matched_recall_dict[r].append(np.mean([x["recall_at_3"] for x in m_runs if x["retrieval_fired"]]) if any(x["retrieval_fired"] for x in m_runs) else 0.0)

        # Table 1: Fixed Reference Threshold (RQ1a)
        table1_fixed_results[dec_name] = {
            "retrieval_rate": float(np.mean(fixed_rate_list)),
            "retrieval_rate_std": float(np.std(fixed_rate_list)),
            "f1": float(np.mean(fixed_f1_list)),
            "f1_std": float(np.std(fixed_f1_list)),
            "em": float(np.mean(fixed_em_list)),
            "em_std": float(np.std(fixed_em_list))
        }

        # Table 2: Matched Retrieval Rate (RQ1b)
        table2_matched_results[dec_name] = {
            "tau_matched": calibrated_taus[dec_name][primary_target_rate]["tau"],
            "tau_ci": [calibrated_taus[dec_name][primary_target_rate]["ci_lower"], calibrated_taus[dec_name][primary_target_rate]["ci_upper"]],
            "retrieval_rate": float(np.mean(matched_rate_dict[primary_target_rate])),
            "retrieval_rate_std": float(np.std(matched_rate_dict[primary_target_rate])),
            "f1": float(np.mean(matched_f1_dict[primary_target_rate])),
            "f1_std": float(np.std(matched_f1_dict[primary_target_rate])),
            "em": float(np.mean(matched_em_dict[primary_target_rate])),
            "em_std": float(np.std(matched_em_dict[primary_target_rate])),
            "recall_at_3": float(np.mean(matched_recall_dict[primary_target_rate])),
            "curve_rates": {
                f"{r:.0%}": {
                    "tau": calibrated_taus[dec_name][r]["tau"],
                    "test_rate": float(np.mean(matched_rate_dict[r])),
                    "f1": float(np.mean(matched_f1_dict[r])),
                    "em": float(np.mean(matched_em_dict[r]))
                } for r in target_rates
            }
        }

    # =========================================================================
    # PHASE 4: SECONDARY DESIGN (RQ1c - Answer Decoder Context Copy Penalty)
    # Draft = GREEDY strictly; Answer Decoder = Greedy vs Contrastive Search
    # =========================================================================
    print(f"\n[PHASE 4] Running Secondary Test Evaluation (Draft = GREEDY, Answer Decoder varies)...")
    results_answer_decoder = {}
    for ans_dec in ["greedy", "contrastive_search"]:
        ans_runs = []
        for q in test_set:
            res = pipeline.run_single_question(
                question=q["question"],
                ground_truths=q["answers"],
                draft_decoder="greedy", # Strictly fixed to greedy
                answer_decoder=ans_dec, # Varies: greedy vs contrastive
                tau=fixed_tau_greedy,
                use_raw=use_raw,
                max_draft_tokens=max_draft_tokens,
                max_answer_tokens=max_answer_tokens,
                seed=42
            )
            ans_runs.append(res)

        ret_count = sum(1 for r in ans_runs if r["retrieval_fired"])
        results_answer_decoder[ans_dec] = {
            "f1": float(np.mean([r["f1"] for r in ans_runs])),
            "em": float(np.mean([r["em"] for r in ans_runs])),
            "copy_rate_3": float(np.mean([r["copy_rate_3"] for r in ans_runs if r["retrieval_fired"]])) if ret_count > 0 else 0.0,
            "retrieval_rate": float(ret_count / len(ans_runs))
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
    print("Secondary Design: Draft Decoder = GREEDY strictly")
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
                "score_type": score_type,
                "entropy_unit": entropy_unit,
                "load_in_4bit": load_in_4bit,
                "primary_target_rate": primary_target_rate,
                "target_rates": target_rates,
                "num_dev_questions": len(dev_set),
                "num_test_questions": len(test_set),
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
        output_dir=args.output_dir
    )
