"""
Retriever modules supporting:
1. Local BM25 Retriever (for local offline testing and pilot sets).
2. Elasticsearch DPR Retriever (for connecting to the 21M Wikipedia passage index as used in DRAGIN).
"""

import math
import re
from typing import List, Dict, Any, Optional


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class BaseRetriever:
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        raise NotImplementedError


class LocalBM25Retriever(BaseRetriever):
    """
    In-memory BM25 retriever for local evaluation, unit tests, and pilots.
    Does not require external services or Elasticsearch.
    """
    def __init__(self, corpus: List[Dict[str, str]], k1: float = 1.5, b: float = 0.75):
        """
        corpus: list of dicts with keys 'id' and 'text'.
        """
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(tokenize(doc["text"])) for doc in corpus]
        self.avg_doc_len = sum(self.doc_len) / max(1, len(corpus))
        self.df = {}
        for doc in corpus:
            unique_terms = set(tokenize(doc["text"]))
            for term in unique_terms:
                self.df[term] = self.df.get(term, 0) + 1
        self.num_docs = len(corpus)

    def _idf(self, term: str) -> float:
        n_q = self.df.get(term, 0)
        return math.log(1.0 + (self.num_docs - n_q + 0.5) / (n_q + 0.5))

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        q_terms = tokenize(query)
        scores = []

        for idx, doc in enumerate(self.corpus):
            doc_terms = tokenize(doc["text"])
            score = 0.0
            doc_len = self.doc_len[idx]

            for term in q_terms:
                if term not in self.df:
                    continue
                tf = doc_terms.count(term)
                if tf == 0:
                    continue
                idf = self._idf(term)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)

            scores.append((idx, score))

        # Rank descending
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, s in scores[:top_k]:
            results.append({
                "id": self.corpus[idx].get("id", str(idx)),
                "text": self.corpus[idx]["text"],
                "score": round(s, 4)
            })
        return results


class ElasticsearchDPRRetriever(BaseRetriever):
    """
    Elasticsearch retriever interface matching DRAGIN's 21M DPR Wikipedia index.
    Connects to an Elasticsearch cluster (default localhost:9200).
    """
    def __init__(self, host: str = "http://localhost:9200", index_name: str = "dpr_wikipedia"):
        self.host = host
        self.index_name = index_name
        self.client = None

    def _connect(self):
        try:
            from elasticsearch import Elasticsearch
            self.client = Elasticsearch([self.host])
        except ImportError:
            raise ImportError("Please install elasticsearch: pip install elasticsearch")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if self.client is None:
            self._connect()

        es_query = {
            "query": {
                "match": {
                    "text": query
                }
            },
            "size": top_k
        }

        res = self.client.search(index=self.index_name, body=es_query)
        hits = res.get("hits", {}).get("hits", [])
        results = []
        for hit in hits:
            source = hit.get("_source", {})
            results.append({
                "id": hit.get("_id"),
                "text": source.get("text", ""),
                "score": round(hit.get("_score", 0.0), 4)
            })
        return results
