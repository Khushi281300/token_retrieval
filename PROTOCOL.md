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
