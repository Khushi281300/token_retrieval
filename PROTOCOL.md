# Experimental Protocol & Pre-Registration: Decoders in Uncertainty-Triggered RAG

**Title:** Does the decoding method change when an uncertainty-triggered RAG system decides to retrieve, and does it change answer quality at the same retrieval rate?  
**Author:** Khushi Shukla  
**Date:** 14 November 2026  
**Target Venue:** ARR (4 January 2027 Deadline)  
**Status:** Pre-registered protocol before execution of the test set.

---

## 1. Research Questions & Hypotheses

* **RQ1a (Fixed-Threshold Shift):** When a single uncertainty threshold $\tau$ is calibrated under greedy decoding on dev, does varying the draft decoder (beam search, contrastive search, nucleus sampling) systematically shift how often retrieval fires, and does answer quality change?
* **RQ1b (Matched Retrieval Rate):** When $\tau$ is independently re-tuned for each decoder on dev so every decoder achieves the identical retrieval rate on test (curves plotted across 20%, 40%, and 60%), does any quality difference persist? Or was the apparent effect solely a retrieval-rate shift?
* **RQ1c (Answer Decoder Context Penalty):** When writing the final answer conditioned on retrieved passages, does Contrastive Search penalize verbatim copying of factual evidence from the retrieved context compared to Greedy decoding?
* **Hypothesis H1:** Decoders with candidate penalties (Contrastive Search) and stochastic sampling inflate token entropy relative to Greedy. When retrieval rates are matched, Greedy-calibrated thresholds reliably transfer within an equivalence margin, except for Contrastive Search which will exhibit a suppressed Context Copy Rate ($n$-gram overlap) in post-retrieval generation.

---

## 2. Model & Generation Specifications

* **Primary Model:** `allenai/OLMo-2-1124-7B-Instruct`
* **Chat Template:** OLMo's official tokenizer chat template (`tokenizer.apply_chat_template`) applied to ensure consistent prompt formatting.
* **Precision:** FP16 on Tesla T4 / BF16 on Ampere+ architectures. Unquantized weights for all reported results (4-bit loading reserved strictly for low-memory pilots if needed).
* **Draft Generation Decoders:**
  1. **Greedy:** `do_sample=False, num_beams=1`
  2. **Beam Search:** `num_beams=5, early_stopping=True, do_sample=False`
  3. **Contrastive Search:** `penalty_alpha=0.6, top_k=4, custom_generate="transformers-community/contrastive-search", trust_remote_code=True`
  4. **Nucleus Sampling:** `temperature=0.7, top_p=0.9`, evaluated across 3 fixed random seeds: `[42, 43, 44]`.
* **Draft Length Capping:** Draft generation is capped at a maximum of 64 tokens, and cut at the first newline (`\n`) or EOS token before uncertainty scoring to eliminate open-ended length artifacts.
* **Answer Decoder:**
  * *Primary Experiment:* **STRICTLY FIXED TO GREEDY** across all draft decoders (isolates trigger evaluation).
  * *Secondary Experiment (RQ1c):* Draft decoder is locked to Greedy; Answer decoder varies between Greedy and Contrastive Search to isolate post-retrieval context copying.

---

## 3. Data & Split Sizes

* **Benchmark Dataset:** `mandarjoshi/trivia_qa` (`rc.wikipedia` config). (NQ-Open as secondary benchmark if compute budget permits).
* **Sample Sizes:**
  * **Dev Set:** Strictly **500 questions** (used exclusively for threshold calibration, bootstrap confidence intervals of $\tau$, and setting equivalence margins).
  * **Test Set:** Strictly **1,000 questions** (evaluated strictly once after freezing dev thresholds).
* **Splitting Protocol:** Fixed random shuffle seed (seed 42) before partitioning into 500 Dev and 1,000 Test items.

---

## 4. Retrieval Corpus & Grounding (No Leaks)

* **Corpus Construction:** Pool the Wikipedia `entity_pages` from all evaluated questions into a single global BM25 index. The true passage for any question sits amongst thousands of distractor passages from other questions.
* **No Gold Leakage:** Any synthetic passages or toy lists containing gold answers are completely deleted. If the index or question context is missing, the pipeline raises an explicit error.
* **Retrieval Query Formulation:**
  * *Main Run:* The **question alone** is used as the BM25 query (ensures identical retrieved passages across all draft decoders, eliminating retriever variance).
  * *Secondary Run:* `question + draft` query formulation.
* **Recall@3 Validation:** For every question, evaluate whether any of the top-3 retrieved passages contains any gold alias. Report Recall@3 across all runs.

---

## 5. Uncertainty Scoring & Triggers

* **Single Forward-Pass Scoring (`score_tokens`):** Rather than reading step-by-step decoding logits (which distort distributions under beam search and contrastive penalties), raw token probabilities and entropies are computed via a single teacher-forced forward pass over the cut draft text:
  $$\log P(w_t \mid w_{<t}) = \text{log\_softmax}(L_t)[w_t]$$
  $$H(P_t) = -\sum_{v} P_t(v) \log P_t(v) \quad \text{(in nats)}$$
* **Primary Trigger:** Maximum Token Entropy (MTE) over the draft: $\max_{t} H(P_t) > \tau$.
* **Secondary Triggers:** Minimum Token Probability ($\min_t P(w_t) < \tau$) and Mean Negative Log-Likelihood ($-\frac{1}{T}\sum \log P(w_t) > \tau$).
* **Target Retrieval Rates:** Evaluated at **20%**, **40%**, and **60%** on Dev to generate full F1-vs-Retrieval-Rate curves.
* **Bootstrap Confidence Intervals:** 1,000 paired bootstrap resamples over Dev questions to report 95% confidence intervals for $\tau$.

---

## 6. Baselines & Controls

1. **No-Retrieval Baseline:** Model answers using closed-book draft generation only.
2. **Oracle Baseline:** Model answers with the gold reference passage provided in context.
3. **Never-Retrieve:** RAG system with retrieval disabled ($\text{rate} = 0\%$).
4. **Always-Retrieve:** RAG system with retrieval enforced on every question ($\text{rate} = 100\%$).
5. **Random Trigger:** Randomly fires retrieval at the matched retrieval rate (e.g., exactly 40%).
6. **Teacher-Forced Control:** Scores the exact same Greedy draft text under each decoder's scoring rules to decouple text changes from score distribution shifts.
7. **Trigger Agreement ($\kappa$):** Cohen's kappa measured per question between Greedy's trigger decisions and each other decoder at the matched rate.

---

## 7. Metrics & Verification Rules

* **Answer Quality:** Token-level Answer F1 and Exact Match (EM) using standard SQuAD normalization (lowercased, stripped punctuation and articles) against all valid gold aliases.
* **Context Copy Rate:**
  $$\text{CopyRate}_3 = \frac{\sum_{g \in \text{3-grams}(\text{Answer})} \mathbb{I}(g \in \text{3-grams}(\text{Context}))}{|\text{3-grams}(\text{Answer})|}$$
* **Repetition Metric:** Distinct-$n$ ($n \in \{1, 2, 3\}$) on generated drafts to verify whether decoders mitigate repetition.
* **Logging Requirement:** One JSONL record per `(question_id, decoder, seed)` recording all prompt tokens, cut draft tokens, per-token log-probs, entropies, trigger decisions, F1/EM, retrieved passage IDs, Recall@3, and git commit hash.
* **Decision Rule:** An effect is claimed if and only if the difference at matched retrieval rate strictly exceeds the dev seed-to-seed equivalence margin in a 95% paired bootstrap test. Otherwise, it is reported as within margin (confirming threshold transferability under H1).

---

## 8. Calibrated Dev Thresholds & Frozen Margins

**Calibration Date:** 15 November 2026  
**Calibration Dataset:** TriviaQA `rc.wikipedia` (Dev $N=500$, random seed 42)  
**Evaluated Model:** `allenai/OLMo-2-1124-7B-Instruct` (4-bit NF4 quantized)  

### Frozen Thresholds ($\tau$ with 1,000-resample 95% Bootstrap CIs):

| Decoder | Target Rate 20% | Target Rate 40% (Primary) | Target Rate 60% |
|---|---|---|---|
| **Greedy** | $\tau = 3.6774$ $[3.3750, 3.9903]$ | $\tau = 2.2184$ $[2.0446, 2.4330]$ | $\tau = 1.4208$ $[1.2809, 1.6062]$ |
| **Beam Search (5)** | $\tau = 3.4156$ $[3.1666, 3.7020]$ | $\tau = 2.1174$ $[1.9548, 2.3160]$ | $\tau = 1.4547$ $[1.2943, 1.5766]$ |
| **Contrastive Search** | $\tau = 3.5165$ $[3.1409, 3.8442]$ | $\tau = 2.1009$ $[1.9472, 2.3237]$ | $\tau = 1.3550$ $[1.2120, 1.5405]$ |
| **Nucleus Sampling (3 seeds)** | $\tau = 3.7254$ $[3.5765, 3.9787]$ | $\tau = 2.2208$ $[2.0964, 2.3438]$ | $\tau = 1.4294$ $[1.3114, 1.5319]$ |

* **Fixed Reference Threshold (tuned under Greedy @ 40%):** $\tau_{\text{fixed}} = 2.2184$

### Dev Trigger Agreement (Cohen's Kappa vs Greedy @ 40% matched rate):
- Greedy vs Beam Search: $\kappa = 0.8167$ (substantial/near-perfect agreement)
- Greedy vs Contrastive Search: $\kappa = 0.8417$
- Greedy vs Nucleus Sampling: $\kappa = 0.8299$

### Teacher-Forced Scoring Control (Identical Greedy Drafts):
- Greedy Scoring: Mean Max-Entropy = $2.2740$, Trigger Rate = $40.0\%$
- Beam Search Scoring: Mean Max-Entropy = $2.2740$, Trigger Rate = $40.0\%$
- Contrastive Search Scoring: Mean Max-Entropy = $2.4999$, Trigger Rate = $46.6\%$ ($+6.6\%$ trigger inflation solely from penalty scoring)
- Nucleus Sampling Scoring: Mean Max-Entropy = $2.2740$, Trigger Rate = $40.0\%$

### MARGINS FROZEN:
- **Equivalence Margin for Answer F1:** $\Delta \text{F1} = 0.025$ ($2.5$ percentage points).
  - An effect of decoding method at matched retrieval rate (RQ1b) is claimed *if and only if* $|F1_{\text{decoder}} - F1_{\text{greedy}}| > 0.025$ with $p < 0.05$ under a paired bootstrap test over the 1,000 test questions.
  - If $|F1_{\text{decoder}} - F1_{\text{greedy}}| \le 0.025$, the difference is statistically and practically indistinguishable from greedy calibration, supporting the H1 transferability claim.
- **Copy Rate Reduction Criterion (RQ1c):** Contrastive Search answer decoder must exhibit a statistically significant reduction in 3-gram context copy rate ($\Delta \text{CopyRate}_3 < -0.05$) relative to Greedy answer decoder.
- **Protocol Status:** MARGINS FROZEN BEFORE TEST SET EVALUATION. Test set ($N=1,000$) to be executed strictly once.
