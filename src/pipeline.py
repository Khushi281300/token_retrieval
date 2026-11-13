"""
Uncertainty-Triggered RAG Pipeline.
Implements:
1. Dynamic retrieval triggering based on per-token uncertainty.
2. Calibration of threshold tau on dev set (Fixed threshold vs Matched retrieval rate).
3. Draft vs Answer decoder decoupling (RQ1a, RQ1b, RQ1c).
"""

from typing import List, Dict, Any, Optional
import numpy as np

from src.decoders import DecoderEngine
from src.retriever import BaseRetriever
from src.metrics import (
    trigger_min_prob,
    trigger_max_entropy,
    trigger_mean_nll,
    compute_f1,
    compute_exact_match,
    compute_copy_rate
)


class RAGPipeline:
    def __init__(
        self,
        decoder_engine: DecoderEngine,
        retriever: BaseRetriever,
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
            return summary["max_entropy"] > tau
        elif self.trigger_type == "min_prob":
            return summary["min_prob"] < tau
        elif self.trigger_type == "mean_nll":
            return summary["mean_nll"] > tau
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
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes a single uncertainty-triggered RAG cycle:
        1. Draft Generation using draft_decoder.
        2. Uncertainty computation on draft tokens.
        3. Trigger check: if uncertainty > tau, retrieve passages and write final answer using answer_decoder.
           Otherwise, draft is the final answer.
        4. Compute F1, EM, copy rate.
        """
        # Step 1: Draft prompt
        draft_prompt = f"Question: {question}\nDraft Answer:"
        draft_res = self.decoder_engine.generate(
            prompt=draft_prompt,
            decoder_type=draft_decoder,
            max_new_tokens=max_draft_tokens,
            seed=seed
        )

        # Step 2: Trigger decision
        retrieval_fired = self.evaluate_trigger(draft_res, tau=tau, use_raw=use_raw)
        retrieved_passages = []
        retrieved_text = ""
        final_answer = draft_res["generated_text"]

        # Step 3: Conditional Retrieval
        if retrieval_fired:
            # Query retriever using question + draft
            query = f"{question} {draft_res['generated_text']}"
            retrieved_passages = self.retriever.retrieve(query=query, top_k=3)
            retrieved_text = "\n".join([p["text"] for p in retrieved_passages])

            # Generate final answer with retrieved passages
            answer_prompt = (
                f"Context:\n{retrieved_text}\n\n"
                f"Question: {question}\n"
                f"Answer:"
            )
            answer_res = self.decoder_engine.generate(
                prompt=answer_prompt,
                decoder_type=answer_decoder,
                max_new_tokens=max_answer_tokens,
                seed=seed
            )
            final_answer = answer_res["generated_text"]

        # Step 4: Metrics
        f1 = compute_f1(final_answer, ground_truths)
        em = compute_exact_match(final_answer, ground_truths)
        copy_rate = compute_copy_rate(final_answer, retrieved_text) if retrieval_fired else 0.0

        return {
            "question": question,
            "draft_decoder": draft_decoder,
            "answer_decoder": answer_decoder,
            "retrieval_fired": retrieval_fired,
            "draft_text": draft_res["generated_text"],
            "final_answer": final_answer,
            "retrieved_passages_count": len(retrieved_passages),
            "f1": f1,
            "em": em,
            "copy_rate": copy_rate,
            "draft_summary": draft_res["raw_summary"]
        }

    def calibrate_threshold(
        self,
        dev_draft_results: List[Dict[str, Any]],
        target_retrieval_rate: float,
        use_raw: bool = True
    ) -> float:
        """
        Given precomputed draft results on dev, finds the threshold tau that
        achieves the exact target_retrieval_rate (e.g., 0.35).
        """
        summary_key = "raw_summary" if use_raw else "processed_summary"
        
        if self.trigger_type == "max_entropy":
            scores = [r[summary_key]["max_entropy"] for r in dev_draft_results]
            # Higher entropy means retrieve: percentile (1 - target)
            tau = float(np.percentile(scores, (1.0 - target_retrieval_rate) * 100))
        elif self.trigger_type == "min_prob":
            scores = [r[summary_key]["min_prob"] for r in dev_draft_results]
            # Lower probability means retrieve: percentile target
            tau = float(np.percentile(scores, target_retrieval_rate * 100))
        elif self.trigger_type == "mean_nll":
            scores = [r[summary_key]["mean_nll"] for r in dev_draft_results]
            tau = float(np.percentile(scores, (1.0 - target_retrieval_rate) * 100))
        else:
            raise ValueError(f"Unknown trigger_type: {self.trigger_type}")

        return round(tau, 4)
