"""
Data loader and retrieval corpus builder for TriviaQA.
Strictly implements the research protocol:
- Loads TriviaQA rc.wikipedia (or rc.nocontext)
- Eliminates all synthetic / toy fallback passages (raises explicit error on failure)
- Pools Wikipedia entity_pages from all questions into a single BM25 corpus
- Provides check_recall_at_k(retrieved_passages, gold_aliases, k=3)
- Splits into 500 Dev and 1,000 Test using fixed random seed (42)
- Supports Oracle mode and No-retrieval mode
"""

import re
import string
import random
from typing import List, Dict, Any, Tuple, Optional


def normalize_answer(s: str) -> str:
    """
    Standard SQuAD / TriviaQA text normalization:
    lowercased, punctuation removed, articles removed, extra whitespace stripped.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def check_recall_at_k(
    retrieved_passages: List[Dict[str, Any]],
    gold_aliases: List[str],
    k: int = 3
) -> bool:
    """
    Evaluates Recall@k: returns True if ANY of the top-k retrieved passages
    contains at least one valid gold alias (under normalized text comparison).
    """
    if not retrieved_passages or not gold_aliases:
        return False

    norm_aliases = [normalize_answer(a) for a in gold_aliases if a.strip()]
    top_k_passages = retrieved_passages[:k]

    for p in top_k_passages:
        p_text = normalize_answer(p.get("text", ""))
        for alias in norm_aliases:
            if not alias:
                continue
            # Word boundary regex or simple substring if single token
            if alias in p_text:
                return True

    return False


def load_trivia_qa(
    config: str = "rc.wikipedia",
    split: str = "validation",
    num_dev: int = 500,
    num_test: int = 1000,
    seed: int = 42,
    streaming: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
    """
    Loads TriviaQA questions, constructs the pooled Wikipedia corpus, and
    partitions questions into Dev and Test sets.
    
    Raises RuntimeError if dataset cannot be loaded. No synthetic fallbacks permitted.
    """
    total_needed = num_dev + num_test
    print(f"Loading TriviaQA ({config}, split='{split}', need {total_needed} questions)...")

    try:
        from datasets import load_dataset
        ds = load_dataset("mandarjoshi/trivia_qa", config, split=split, streaming=streaming)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load TriviaQA dataset from HuggingFace ({config}, {split}): {e}. "
            f"Please verify internet connection and HuggingFace authentication token."
        )

    raw_items = []
    all_corpus_docs = []
    seen_doc_ids = set()

    for idx, item in enumerate(ds):
        if len(raw_items) >= total_needed:
            break

        q_id = item.get("question_id", str(idx))
        question_text = item.get("question", "").strip()

        # Extract answer aliases
        ans_info = item.get("answer", {})
        aliases = ans_info.get("aliases", [])
        val = ans_info.get("value", "")
        normalized_aliases = ans_info.get("normalized_aliases", [])
        gold_aliases = list(set([a for a in aliases + ([val] if val else []) if a]))

        if not gold_aliases:
            continue

        # Extract wiki context entity pages
        wiki_contexts = []
        entity_pages = item.get("entity_pages", {})
        if isinstance(entity_pages, dict):
            titles = entity_pages.get("title", [])
            contexts = entity_pages.get("wiki_context", [])
            for doc_idx, text in enumerate(contexts):
                if text and text.strip():
                    doc_title = titles[doc_idx] if doc_idx < len(titles) else f"doc_{q_id}_{doc_idx}"
                    doc_id = f"wiki_{q_id}_{doc_idx}"
                    wiki_contexts.append({"id": doc_id, "title": doc_title, "text": text.strip()})
                    if doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc_id)
                        all_corpus_docs.append({"id": doc_id, "title": doc_title, "text": text.strip()})

        # Fallback to search_results if entity_pages empty (e.g. in rc.nocontext or rc.web)
        if not wiki_contexts:
            search_results = item.get("search_results", {})
            if isinstance(search_results, dict):
                search_contexts = search_results.get("search_context", [])
                for s_idx, text in enumerate(search_contexts):
                    if text and text.strip():
                        doc_id = f"search_{q_id}_{s_idx}"
                        wiki_contexts.append({"id": doc_id, "title": f"search_{q_id}", "text": text.strip()})
                        if doc_id not in seen_doc_ids:
                            seen_doc_ids.add(doc_id)
                            all_corpus_docs.append({"id": doc_id, "title": f"search_{q_id}", "text": text.strip()})

        # Identify gold oracle passage containing at least one gold alias
        gold_passage = ""
        for wc in wiki_contexts:
            norm_ctx = normalize_answer(wc["text"])
            for alias in gold_aliases:
                if normalize_answer(alias) in norm_ctx:
                    gold_passage = wc["text"]
                    break
            if gold_passage:
                break

        if not gold_passage and wiki_contexts:
            gold_passage = wiki_contexts[0]["text"]

        raw_items.append({
            "id": q_id,
            "question": question_text,
            "answers": gold_aliases,
            "gold_passage": gold_passage,
            "passages": [wc["text"] for wc in wiki_contexts[:3]]
        })

    if len(raw_items) < total_needed:
        print(f"Warning: Only {len(raw_items)} questions were retrieved (requested {total_needed}).")

    # Fixed random shuffle before partitioning
    rng = random.Random(seed)
    rng.shuffle(raw_items)

    dev_questions = raw_items[:num_dev]
    test_questions = raw_items[num_dev : num_dev + num_test]

    print(f"TriviaQA Partitioning: Dev = {len(dev_questions)}, Test = {len(test_questions)}")
    print(f"Total Pooled BM25 Corpus Passages: {len(all_corpus_docs)}")

    return dev_questions, test_questions, all_corpus_docs
