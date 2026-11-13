# Uncertainty-Triggered RAG: Decoding Methods & Retrieval Calibration

> **Research Question:** Does the decoding method change when an uncertainty-triggered RAG system decides to retrieve, and does it change answer quality at the same retrieval rate?
>
  


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Khushi281300/token_retrieval/blob/main/notebooks/01_colab_reproduce_and_eval.ipynb)

---

## 1. Directory Structure

```text
token_retrieval/
├── configs/           # Run configs with hyperparameters and seed definitions
├── data/              # Frozen dev & test splits (TriviaQA, 2WikiMultihopQA, HotpotQA)
├── runs/              # Saved model outputs with raw logits & processed scores
├── src/
│   ├── metrics.py     # Uncertainty metrics (MSP, MTE, log-prob, entropy) & QA eval (F1, EM, copy rate)
│   ├── decoders.py    # Generation runners for greedy, beam, contrastive, and nucleus sampling
│   └── retriever.py   # BM25 / Elasticsearch interface
├── scripts/           # Execution scripts
├── analysis/          # Notebooks, statistical bootstrap tests & plotting
└── smoke_test.py      # Day-1 verification of logits/scores extraction
```

---

## 2. Quickstart:Smoke Test

Run the verification test:
```bash
python smoke_test.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```
This produces `runs/smoke_test.json`, confirming:
1. Model generates tokens properly.
2. Raw logits and decoder-processed scores are simultaneously tracked per token.
3. Next-token entropy $H(P_t)$ and log-probability $\log P(w_t)$ are calculated in log-space.
4. Offline triggers (Min-Prob, Max-Entropy, Mean-NLL) are validated.

