"""
rag_engine.py
-------------
Core RAG pipeline: document indexing (Admin) and query execution (User).

build_index() — ingests a PDF, creates embeddings, and persists the index.
query_index() — loads a stored index and answers a natural language question.
"""

import os
import time
import json
import logging
from typing import Callable, Optional

import faiss
import chromadb
import numpy as np

from rag.pdf_loader import load_pdf
from rag.chunker import chunk_text
from rag.bm25_retriever import BM25Retriever, tokenize
from models.embedding import embed
from models.llm import generate_answer, stream_answer
from config import (
    STORAGE_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
)

logger = logging.getLogger(__name__)

# ── In-process caches (avoid reloading indices on every query) ────────────────
_bm25_cache:   dict = {}
_faiss_cache:  dict = {}
_chunks_cache: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# ADMIN: Build Index
# ─────────────────────────────────────────────────────────────────────────────

def build_index(
    pdf_path: str,
    embedding_model: str,
    vector_db: str = "FAISS",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    on_step: Optional[Callable[[int], None]] = None,
) -> None:
    """
    Ingest a PDF document, generate embeddings, and persist the index to disk.

    Steps reported via on_step(step_index):
        0 — Extracting text
        1 — Chunking text
        2 — Generating embeddings
        3 — Saving index to disk

    Args:
        pdf_path:        Path to the source PDF file.
        embedding_model: Embedding model key (e.g. 'BGE-small').
        vector_db:       Vector database to use: 'FAISS' or 'Chroma'.
        chunk_size:      Number of characters per text chunk.
        overlap:         Character overlap between consecutive chunks.
        on_step:         Optional callback called with the current step index (0-3).

    Raises:
        FileExistsError: If an index already exists for this document.
        ValueError:      If an unsupported vector_db is specified.
    """
    def _step(n):
        if on_step:
            on_step(n)

    os.makedirs(STORAGE_PATH, exist_ok=True)

    document_name = os.path.splitext(os.path.basename(pdf_path))[0]
    document_folder = os.path.join(STORAGE_PATH, document_name)

    if os.path.exists(document_folder):
        raise FileExistsError(
            f"Index already exists for '{document_name}'. "
            f"Enable 'Force Rebuild' to overwrite."
        )

    os.makedirs(document_folder)
    logger.info(
        "Building index for '%s' | embedding=%s | db=%s",
        document_name, embedding_model, vector_db
    )

    # ── Step 0: Load & extract text ───────────────────────────────────────────
    _step(0)
    text = load_pdf(pdf_path)

    # ── Step 1: Chunk ─────────────────────────────────────────────────────────
    _step(1)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    # Persist tokenized chunks for BM25
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
    _write_json(os.path.join(document_folder, "tokenized_chunks.json"), tokenized_chunks)

    # ── Step 2: Embed ─────────────────────────────────────────────────────────
    _step(2)
    embeddings = np.array(embed(chunks, model_name=embedding_model)).astype("float32")

    # ── Step 3: Persist vector index ──────────────────────────────────────────
    _step(3)

    if vector_db == "FAISS":
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, os.path.join(document_folder, "index.faiss"))

    elif vector_db == "Chroma":
        client = chromadb.PersistentClient(path=document_folder)
        collection = client.get_or_create_collection("rag_collection")
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                embeddings=[embeddings[i].tolist()],
                ids=[str(i)],
            )

    else:
        raise ValueError(
            f"Unsupported vector database: '{vector_db}'. Choose 'FAISS' or 'Chroma'."
        )

    # ── Persist chunks & metadata ─────────────────────────────────────────────
    _write_json(os.path.join(document_folder, "chunks.json"), chunks)
    _write_json(
        os.path.join(document_folder, "metadata.json"),
        {
            "embedding_model": embedding_model,
            "vector_db":       vector_db,
            "chunk_size":      chunk_size,
            "overlap":         overlap,
            "chunk_count":     len(chunks),
        },
    )

    logger.info(
        "Index built successfully for '%s' (%d chunks).",
        document_name, len(chunks)
    )


# ─────────────────────────────────────────────────────────────────────────────
# USER: Query Index
# ─────────────────────────────────────────────────────────────────────────────

def query_index(
    document_name: str,
    question: str,
    llm_model: str = "llama3:latest",
    top_k: int = TOP_K,
) -> dict:
    """
    Load a stored index and answer a natural language question.

    Uses a hybrid retrieval strategy (semantic + BM25) for FAISS indices.

    Args:
        document_name: Name of the indexed document (without extension).
        question:      Natural language question string.
        llm_model:     Ollama model identifier for answer generation.
        top_k:         Number of chunks to retrieve.

    Returns:
        Dictionary with keys:
            'answer'           — Generated answer string.
            'retrieved_chunks' — List of source text chunks used as context.
            'metrics'          — Performance and configuration metadata.
        On failure, returns {'error': <message>}.
    """
    document_folder = os.path.join(STORAGE_PATH, document_name)

    if not os.path.exists(document_folder):
        logger.warning("Document not found: '%s'", document_name)
        return {"error": f"No index found for document '{document_name}'."}

    total_start = time.time()

    # ── Load metadata ─────────────────────────────────────────────────────────
    metadata = _read_json(os.path.join(document_folder, "metadata.json"))
    embedding_model = metadata["embedding_model"]
    vector_db = metadata["vector_db"]

    # ── Embed query ───────────────────────────────────────────────────────────
    t0 = time.time()
    query_vec = np.array(
        [embed([question], model_name=embedding_model)[0]]
    ).astype("float32")
    embedding_time = time.time() - t0

    # ── Retrieve ──────────────────────────────────────────────────────────────
    t0 = time.time()

    if vector_db == "FAISS":
        retrieved_chunks = _hybrid_retrieve(
            document_name, document_folder, query_vec, question, top_k
        )

    elif vector_db == "Chroma":
        client = chromadb.PersistentClient(path=document_folder)
        collection = client.get_collection("rag_collection")
        results = collection.query(
            query_embeddings=[query_vec[0].tolist()],
            n_results=top_k,
        )
        retrieved_chunks = results["documents"][0]

    else:
        return {"error": f"Unsupported vector database: '{vector_db}'."}

    retrieval_time = time.time() - t0

    # ── Generate answer ───────────────────────────────────────────────────────
    context = "\n".join(retrieved_chunks)
    prompt = (
        "Answer concisely in 4-5 lines using only the context below.\n"
        "If the answer is not present in the context, say: "
        "'I don't have enough information in this document to answer that.'\n"
        "Do not add information outside the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )

    t0 = time.time()
    answer = generate_answer(prompt, model_name=llm_model)
    generation_time = time.time() - t0

    total_time = time.time() - total_start
    approx_tokens = len(answer.split())

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "metrics": {
            "embedding_model":     embedding_model,
            "vector_db":           vector_db,
            "embedding_time":      round(embedding_time, 4),
            "retrieval_time":      round(retrieval_time, 4),
            "generation_time":     round(generation_time, 4),
            "total_time":          round(total_time, 4),
            "tokens_per_second":   round(
                approx_tokens / generation_time, 2
            ) if generation_time > 0 else 0,
            "prompt_length_chars": len(prompt),
            "answer_length_chars": len(answer),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# USER: Stream Query Index (for typing animation)
# ─────────────────────────────────────────────────────────────────────────────

def stream_query_index(
    document_name: str,
    question: str,
    llm_model: str = "llama3:latest",
    top_k: int = TOP_K,
) -> dict:
    """
    Same as query_index but returns the prompt and retrieved chunks so the
    caller can stream tokens directly via stream_answer().

    Returns:
        {
          'prompt':           str   — ready-to-send prompt,
          'retrieved_chunks': list  — source chunks used,
          'metrics':          dict  — embedding + retrieval timing (no generation time),
        }
        On failure: {'error': str}
    """
    document_folder = os.path.join(STORAGE_PATH, document_name)

    if not os.path.exists(document_folder):
        return {"error": f"No index found for document '{document_name}'."}

    total_start = time.time()

    metadata = _read_json(os.path.join(document_folder, "metadata.json"))
    embedding_model = metadata["embedding_model"]
    vector_db = metadata["vector_db"]

    # Embed query
    t0 = time.time()
    query_vec = np.array(
        [embed([question], model_name=embedding_model)[0]]
    ).astype("float32")
    embedding_time = time.time() - t0

    # Retrieve
    t0 = time.time()
    if vector_db == "FAISS":
        retrieved_chunks = _hybrid_retrieve(
            document_name, document_folder, query_vec, question, top_k
        )
    elif vector_db == "Chroma":
        client = chromadb.PersistentClient(path=document_folder)
        collection = client.get_collection("rag_collection")
        results = collection.query(
            query_embeddings=[query_vec[0].tolist()],
            n_results=top_k,
        )
        retrieved_chunks = results["documents"][0]
    else:
        return {"error": f"Unsupported vector database: '{vector_db}'."}
    retrieval_time = time.time() - t0

    context = "\n".join(retrieved_chunks)
    prompt = (
        "Answer concisely in 4-5 lines using only the context below.\n"
        "If the answer is not present in the context, say: "
        "'I don't have enough information in this document to answer that.'\n"
        "Do not add information outside the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )

    return {
        "prompt":           prompt,
        "retrieved_chunks": retrieved_chunks,
        "metrics": {
            "embedding_model": embedding_model,
            "vector_db":       vector_db,
            "embedding_time":  round(embedding_time, 4),
            "retrieval_time":  round(retrieval_time, 4),
            "prompt_length_chars": len(prompt),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hybrid_retrieve(
    document_name: str,
    document_folder: str,
    query_vec: np.ndarray,
    question: str,
    top_k: int,
) -> list[str]:
    """
    Combine FAISS vector search and BM25 keyword search using weighted score fusion.
    Weights are controlled by VECTOR_WEIGHT and BM25_WEIGHT in config.py.
    """
    # ── FAISS (cached) ────────────────────────────────────────────────────────
    if document_name not in _faiss_cache:
        _faiss_cache[document_name] = faiss.read_index(
            os.path.join(document_folder, "index.faiss")
        )
    if document_name not in _chunks_cache:
        _chunks_cache[document_name] = _read_json(
            os.path.join(document_folder, "chunks.json")
        )

    index  = _faiss_cache[document_name]
    chunks = _chunks_cache[document_name]

    distances, indices = index.search(query_vec, top_k)
    vector_scores = {
        chunks[idx]: 1 / (1 + dist)
        for idx, dist in zip(indices[0], distances[0])
    }

    # ── BM25 (cached) ─────────────────────────────────────────────────────────
    if document_name not in _bm25_cache:
        tokenized_chunks = _read_json(
            os.path.join(document_folder, "tokenized_chunks.json")
        )
        _bm25_cache[document_name] = BM25Retriever(
            documents=chunks,
            tokenized_docs=tokenized_chunks,
        )

    bm25_results = _bm25_cache[document_name].retrieve(question, top_k=top_k)
    max_bm25 = max((score for _, score in bm25_results), default=1) or 1
    bm25_scores = {chunk: score / max_bm25 for chunk, score in bm25_results}

    # ── Weighted fusion ───────────────────────────────────────────────────────
    all_chunks = set(vector_scores) | set(bm25_scores)
    combined = {
        chunk: VECTOR_WEIGHT * vector_scores.get(chunk, 0)
              + BM25_WEIGHT  * bm25_scores.get(chunk, 0)
        for chunk in all_chunks
    }

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]


def _write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)