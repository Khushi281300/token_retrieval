# Day-1 Verification & Grounding Report

**Project:** Decoders and Uncertainty-Triggered RAG  
**Author:** Khushi Shukla  
**Date:** 13 November 2026  

---

## 1. Third-Party Codebase Grounding

### A. DRAGIN Repository ([oneal2000/DRAGIN](https://github.com/oneal2000/DRAGIN))
- **Cloned to:** `repos/DRAGIN`
- **Core Findings:**
  1. **Elasticsearch Index:** Exactly `21,015,324` DPR Wikipedia passages (~11.2 GB store size). Elasticsearch silently aborts indexing if free disk space falls below 10%.
  2. **Multi-GPU Setup:** The original DRAGIN code was run across two GPUs. Our plan's decision to report our own single-GPU baseline is fully justified.
  3. **Step 3 Answer (When is post-retrieval signal computed?):**
     > *In DRAGIN (`generate.py`, lines 619–705), retrieval immediately appends only the top sentence generated conditioned on retrieved context without computing uncertainty on it. The post-retrieval uncertainty signal is computed on the subsequent generation step when continuing the answer in the next iteration of the while loop.*

---

### B. Hashimoto et al. Repository ([wataruhashimoto52/decoding_uncertainty](https://github.com/wataruhashimoto52/decoding_uncertainty))
- **Paper:** *"Decoding Uncertainty: The Impact of Decoding Strategies for Uncertainty Estimation in Large Language Models"* (EMNLP Findings 2025).
- **Cloned to:** `repos/decoding_uncertainty`
- **Checkpoints & Tasks:** Evaluates LLaMA-2/3 and Qwen models on TriviaQA, CoQA, and GSM8K. Uses MSP (Maximum Sequence/Softmax Probability), MTE (Mean/Max Token Entropy), and PRR (Prediction Rejection Ratio).

---

## 2. Checkpoint & Library Verifications

1. **Hugging Face Checkpoints (Confirmed):**
   - `allenai/OLMo-2-1124-7B-Instruct` (Public, ~42k downloads)
   - `allenai/OLMo-2-1124-7B-SFT` (Public)
   - `allenai/OLMo-2-1124-7B-DPO` (Public)
2. **Contrastive Search Requirement (Confirmed):**
   - In modern Transformers (v5.6.2+), contrastive search requires:
     `custom_generate="transformers-community/contrastive-search", trust_remote_code=True`
3. **Dual Scoring Mechanism (Confirmed):**
   - Hugging Face `output_logits=True` returns raw forward-pass distributions.
   - `output_scores=True` returns decoder-modified scores. Both are successfully extracted by `DecoderEngine`.
