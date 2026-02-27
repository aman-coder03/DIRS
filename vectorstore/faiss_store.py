"""
faiss_store.py
--------------
In-memory FAISS vector store used by the CLI pipeline (main.py).
For the Streamlit app, FAISS indices are persisted to disk via rag_engine.py.
"""

import logging
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSStore:
    """
    Wraps a FAISS IndexFlatIP (inner product / cosine) index with
    associated text storage for simple in-memory retrieval.

    Args:
        dim: Dimensionality of the embedding vectors.
    """

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.texts: list[str] = []
        logger.info("FAISSStore initialized with dim=%d.", dim)

    def add(self, embeddings: list, texts: list[str]) -> None:
        """
        Add embeddings and their corresponding text chunks to the store.

        Args:
            embeddings: List or array of embedding vectors.
            texts:      Corresponding list of text strings.
        """
        self.index.add(np.array(embeddings).astype("float32"))
        self.texts.extend(texts)
        logger.info("Added %d vectors. Total stored: %d.", len(texts), len(self.texts))

    def search(self, query_embedding: list, k: int = 3) -> list[str]:
        """
        Search for the top-k most similar chunks to a query embedding.

        Args:
            query_embedding: Single embedding vector for the query.
            k:               Number of results to return.

        Returns:
            List of top-k text chunk strings.
        """
        _, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), k
        )
        results = [self.texts[i] for i in indices[0] if i < len(self.texts)]
        logger.debug("FAISS search returned %d results.", len(results))
        return results