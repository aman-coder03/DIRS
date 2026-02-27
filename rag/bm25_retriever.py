"""
bm25_retriever.py
-----------------
Keyword-based retrieval using the BM25Okapi algorithm.
Used as the sparse retrieval component in the hybrid retrieval pipeline.
"""

import re
import logging
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def tokenize(text: str) -> list[str]:
    """
    Lowercase, strip punctuation, and split text into tokens.

    Args:
        text: Raw input string.

    Returns:
        List of lowercase word tokens.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()


class BM25Retriever:
    """
    Wraps BM25Okapi for document retrieval over a fixed corpus.

    Args:
        documents:      List of raw text chunks.
        tokenized_docs: Pre-tokenized version of documents. If provided,
                        tokenization is skipped (useful when loading from cache).
    """

    def __init__(self, documents: list[str], tokenized_docs: list[list[str]] | None = None):
        self.documents = documents
        self.tokenized_docs = tokenized_docs or [tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        logger.info("BM25Retriever initialized with %d documents.", len(documents))

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Retrieve the top-k most relevant documents for a query.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.

        Returns:
            List of (document_text, bm25_score) tuples, sorted by score descending.
        """
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [(self.documents[i], float(scores[i])) for i in top_indices]
        logger.debug("BM25 top-%d retrieved for query: '%s'", top_k, query)
        return results