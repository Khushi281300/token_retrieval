# Pre-Registration: Decoders and Uncertainty-Triggered RAG

**Title:** Does the decoding method change when an uncertainty-triggered RAG system decides to retrieve, and does it change answer quality at the same retrieval rate?  
**Author:** Khushi Shukla  
**Date:** 13 November 2026  
**Target Venue:** ARR (4 January 2027)  
**Status:** Pre-registered before test-set evaluation.

---

## 1. Hypotheses & Primary Research Questions

### Research Questions
- **RQ1a (Fixed-Threshold Shift):** With threshold $\tau$ calibrated under greedy decoding on dev, does changing the draft decoder significantly change retrieval frequency (retrieval rate shift) and answer quality?
- **RQ1b (Matched Retrieval Rate):** When $\tau$ is independently re-tuned for each decoder to achieve the exact same retrieval rate as greedy, does any difference in answer quality persist? Or was the effect purely a retrieval-rate shift?
- **RQ1c (Answer Decoder Interaction):** Does changing the answer decoder after retrieval (specifically Contrastive Search) degrade answer quality by penalizing verbatim copying from retrieved context?

### Hypotheses
- **H1:** Significant quality differences at matched retrieval rate appear primarily for **Contrastive Search** and for **SFT-only models**. Nucleus sampling ($T=0.7, p=0.9$) and beam search will fall within the pre-established equivalence margin of greedy decoding.
- **Mechanism Hypothesis:** Contrastive search penalizes tokens semantically close to previous context. Because retrieved passages are injected directly into context, contrastive search will exhibit a lower *Copy Rate* from retrieved passages compared to greedy decoding.
- **Null Result Validity:** If all decoders sit within the equivalence margin at matched retrieval rate, this constitutes a clean, publishable confirmation that greedy-calibrated uncertainty thresholds reliably transfer across decoders.

---

## 2. Locked Decisions (No Deviations Permitted)

| Dimension | Specification | Rationale |
| :--- | :--- | :--- |
| **Primary Model** | `allenai/OLMo-2-1124-7B-Instruct` | Fully open pretraining corpus (Dolma) enabling contamination and QuCo-RAG comparisons. |
| **Alignment Ladder** | `OLMo-2-1124-7B-SFT`, `-DPO`, `-Instruct` | Tests H1 regarding SFT vs aligned models on probability calibration. |
| **Decoders Tested** | 1. Greedy (`num_beams=1`, `do_sample=False`)<br>2. Beam Search (`num_beams=5`, `early_stopping=True`)<br>3. Contrastive Search (`penalty_alpha` $\in \{0.4, 0.6, 0.8\}$, `top_k=4`)<br>4. Nucleus Sampling ($T=0.7$, `top_p=0.9`, seeds: 42, 43, 44) | Matches Hashimoto et al. and TARG standard grids. |
| **Quantization** | **None (FP16/BF16 only)** | Quantization flattens and distorts output logits and entropy distributions. |
| **Benchmark Datasets** | 1. `TriviaQA` (`rc.nocontext`) for Step 2 reproduction.<br>2. `2WikiMultihopQA` & `HotpotQA` for main dynamic RAG experiments. | Multi-hop reasoning where dynamic retrieval intervention is critical. |
| **Splits** | Dev: $\ge 500$ questions (used for $\tau$ calibration).<br>Test: $\ge 1000$ questions (evaluated strictly once). | Prevents data leakage and overfitting of uncertainty thresholds. |
| **Retriever & Index** | DPR Wikipedia 21M passage dump (~11.2 GB index), BM25 scoring. | DRAGIN reference setup. |

---

## 3. Metrics & Calibration Protocol

1. **Uncertainty Signals Saved:**
   - Both raw forward-pass logits (`output_logits`) and decoder-processed scores (`output_scores`) are stored per token to avoid scale artifacts.
2. **Triggers:**
   - Maximum Token Entropy (MTE): fires if $\max_t H(P_t) > \tau$
   - Minimum Token Probability (MSP): fires if $\min_t P(w_t) < \tau$
   - Mean Negative Log-Likelihood: fires if $-\frac{1}{T}\sum \log P(w_t) > \tau$
3. **Evaluation Metrics:**
   - Token-level Answer F1 and Exact Match (EM).
   - Retrieval Rate ($\%$) and calls per question.
   - Copy Rate: 3-gram overlap between generated answer and retrieved context.
   - Prediction Rejection Ratio (PRR) for Hashimoto reproduction.

---

## 4. Statistical Testing & Decision Criteria

- **Seeds:** 3 seeds for stochastic decoders (seeds 42, 43, 44). Seed variance reported separately from question variance.
- **Statistical Significance:** Paired bootstrap sampling over questions ($B=1000$, 95% bootstrap confidence intervals) with Holm-Bonferroni correction.
- **Equivalence Margin:** Defined from dev-set seed-to-seed variability in pilot runs. A decoder is declared to "have an effect" if and only if its 95% confidence interval strictly excludes zero and exceeds the equivalence margin at matched retrieval rate.

