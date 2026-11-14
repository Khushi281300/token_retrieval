"""
Uncertainty-Triggered RAG Pipeline.
Implements:
1. Dynamic retrieval triggering based on per-token uncertainty.
2. Calibration of threshold tau on dev set (Fixed threshold vs Matched retrieval rate).
3. Draft vs Answer decoder decoupling (RQ1a, RQ1b, RQ1c).
4. Oracle mode, Always-retrieve, Never-retrieve, and Random baselines.
5. Strict Recall@3 evaluation and clean question-only retrieval query formulation.
"""

import random
from typing import List, Dict, Any, Optional
import numpy as np

from src.decoders import DecoderEngine
from src.retriever import BaseRetriever
from src.data_loader import check_recall_at_k
from src.metrics import (
    trigger_min_prob,
    trigger_max_entropy,
    trigger_mean_nll,
    compute_f1,
    compute_exact_match,
    compute_copy_rate,
    bootstrap_threshold_ci
)


class RAGPipeline:
    def __init__(
        self,
        decoder_engine: DecoderEngine,
        retriever: Optional[BaseRetriever] = None,
        trigger_type: str = "max_entropy"
    ):
        """
        trigger_type: 'max_entropy', 'min_prob', or 'mean_nll'
        """
        self.decoder_engine = decoder_engine
        self.retriever = retriever
        self.trigger_type = trigger_type

    def evaluate_trigger(self, generation_result: Dict[str, Any], tau: float, use_raw: bool = True) -> bool:
        """
        Evaluates whether uncertainty crossed threshold tau.
        use_raw: whether to evaluate on raw logits distribution or processed scores.
        """
        summary_key = "raw_summary" if use_raw else "processed_summary"
        summary = generation_result[summary_key]

        if self.trigger_type == "max_entropy":
            return bool(summary["max_entropy"] > tau)
        elif self.trigger_type == "min_prob":
            return bool(summary["min_prob"] < tau)
        elif self.trigger_type == "mean_nll":
            return bool(summary["mean_nll"] > tau)
        else:
            raise ValueError(f"Unknown trigger_type: {self.trigger_type}")

    def run_single_question(
        self,
        question: str,
        ground_truths: List[str],
        draft_decoder: str = "greedy",
        answer_decoder: str = "greedy",
        tau: float = 1.5,
        use_raw: bool = True,
        max_draft_tokens: int = 32,
        max_answer_tokens: int = 32,
        seed: Optional[int] = None,
        query_mode: str = "question",
        force_retrieval: Optional[bool] = None,
        oracle_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes a single uncertainty-triggered RAG cycle:
        1. Draft Generation using draft_decoder (cut at first newline/EOS).
        2. Uncertainty computation on draft tokens.
        3. Trigger check (or forced retrieval for baselines).
        4. Conditional Retrieval (or Oracle context) with Recall@3 validation.
        5. Final Answer generation using answer_decoder.
        6. Compute F1, EM, copy_rate, and distinct-n.
        """
        # Step 1: Draft generation
        draft_prompt = f"Question: {question}\nDraft Answer:"
        draft_res = self.decoder_engine.generate(
            prompt=draft_prompt,
            decoder_type=draft_decoder,
            max_new_tokens=max_draft_tokens,
            seed=seed,
            cut_draft=True
        )

        # Step 2: Trigger decision
        if force_retrieval is not None:
            retrieval_fired = force_retrieval
        else:
            retrieval_fired = self.evaluate_trigger(draft_res, tau=tau, use_raw=use_raw)

        retrieved_passages = []
        retrieved_text = ""
        final_answer = draft_res["generated_text"]
        recall_at_3 = False

        # Step 3: Retrieval if triggered
        if retrieval_fired:
            if oracle_context:
                retrieved_text = oracle_context
                retrieved_passages = [{"id": "oracle_gold", "text": oracle_context}]
                recall_at_3 = True
            elif self.retriever is not None:
                # Main run uses question alone to avoid retriever confounding across decoders
                query = question if query_mode == "question" else f"{question} {draft_res['generated_text']}"
                retrieved_passages = self.retriever.retrieve(query=query, top_k=3)
                retrieved_text = "\n".join([p["text"] for p in retrieved_passages])
                recall_at_3 = check_recall_at_k(retrieved_passages, ground_truths, k=3)

            # Step 4: Write answer conditioned on context
            answer_prompt = (
                f"Context:\n{retrieved_text}\n\n"
                f"Question: {question}\n"
                f"Answer:"
            )
            answer_res = self.decoder_engine.generate(
                prompt=answer_prompt,
                decoder_type=answer_decoder,
                max_new_tokens=max_answer_tokens,
                seed=seed,
                cut_draft=True
            )
            final_answer = answer_res["generated_text"]

        # Step 5: Metrics
        f1 = compute_f1(final_answer, ground_truths)
        em = compute_exact_match(final_answer, ground_truths)
        copy_rate = compute_copy_rate(final_answer, retrieved_text, n=3) if (retrieval_fired and retrieved_text) else 0.0

        return {
            "question": question,
            "draft_decoder": draft_decoder,
            "answer_decoder": answer_decoder,
            "retrieval_fired": retrieval_fired,
            "draft_text": draft_res["generated_text"],
            "draft_len": draft_res["draft_len"],
            "final_answer": final_answer,
            "retrieved_passage_ids": [p.get("id", "") for p in retrieved_passages],
            "retrieved_passages_count": len(retrieved_passages),
            "recall_at_3": recall_at_3,
            "f1": f1,
            "em": em,
            "copy_rate_3": copy_rate,
            "distinct_1": draft_res.get("distinct_1", 1.0),
            "distinct_2": draft_res.get("distinct_2", 1.0),
            "distinct_3": draft_res.get("distinct_3", 1.0),
            "raw_summary": draft_res["raw_summary"],
            "processed_summary": draft_res["processed_summary"],
            "contrastive_alpha": draft_res.get("contrastive_alpha"),
            "contrastive_k": draft_res.get("contrastive_k")
        }

    def calibrate_threshold(
        self,
        dev_draft_results: List[Dict[str, Any]],
        target_retrieval_rate: float,
        use_raw: bool = True,
        compute_ci: bool = False
    ) -> Dict[str, Any]:
        """
        Given precomputed draft results on dev, finds the threshold tau that
        achieves the exact target_retrieval_rate (e.g., 0.40).
        Optionally returns bootstrap 95% confidence intervals.
        """
        summary_key = "raw_summary" if use_raw else "processed_summary"
        
        if self.trigger_type == "max_entropy":
            scores = [r[summary_key]["max_entropy"] for r in dev_draft_results]
        elif self.trigger_type == "min_prob":
            scores = [r[summary_key]["min_prob"] for r in dev_draft_results]
        elif self.trigger_type == "mean_nll":
            scores = [r[summary_key]["mean_nll"] for r in dev_draft_results]
        else:
            raise ValueError(f"Unknown trigger_type: {self.trigger_type}")

        if compute_ci:
            return bootstrap_threshold_ci(scores, target_rate=target_retrieval_rate)
        else:
            pct = (1.0 - target_retrieval_rate) * 100.0 if self.trigger_type in ["max_entropy", "mean_nll"] else target_retrieval_rate * 100.0
            tau = float(np.percentile(scores, pct))
            return {"tau": round(tau, 4), "ci_lower": round(tau, 4), "ci_upper": round(tau, 4)}

