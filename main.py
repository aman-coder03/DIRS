"""
main.py
-------
Command-line interface for the DIRS RAG pipeline.
Loads all PDFs from the configured data folder, builds an in-memory FAISS index,
and enters an interactive query loop.

Usage:
    python main.py
"""

import os
import time
import csv
import logging
from datetime import datetime

from rag.pdf_loader import load_pdf
from rag.chunker import chunk_text
from models.embedding import embed
from models.llm import generate_answer
from vectorstore.faiss_store import FAISSStore
from config import (
    PDF_FOLDER,
    EMBEDDING_MODEL,
    LLM_MODEL,
    VECTOR_DB,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K,
    RESULTS_FILE,
)


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def save_experiment(row: list) -> None:
    """Append a result row to the experiments CSV, writing a header if needed."""
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    file_exists = os.path.isfile(RESULTS_FILE)

    with open(RESULTS_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp", "embedding_model", "llm_model", "vector_db",
                "chunk_size", "chunk_overlap", "chunk_count", "top_k",
                "embedding_time", "retrieval_time", "generation_time",
                "total_time", "prompt_length_chars", "answer_length_chars",
                "tokens_per_second",
            ])
        writer.writerow(row)


def main() -> None:
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("Initializing DIRS CLI pipeline...")

    total_start = time.time()

    # ── Load PDFs ─────────────────────────────────────────────────────────────
    if not os.path.isdir(PDF_FOLDER):
        logger.error("PDF folder not found: %s", PDF_FOLDER)
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        logger.error("No PDF files found in: %s", PDF_FOLDER)
        return

    combined_text = ""
    for pdf_file in pdf_files:
        full_path = os.path.join(PDF_FOLDER, pdf_file)
        logger.info("Reading: %s", pdf_file)
        combined_text += "\n\n" + load_pdf(full_path)

    logger.info("Loaded %d PDF(s).", len(pdf_files))

    # ── Chunk ─────────────────────────────────────────────────────────────────
    chunks = chunk_text(combined_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    chunk_count = len(chunks)
    logger.info("Created %d chunks.", chunk_count)

    # ── Embed ─────────────────────────────────────────────────────────────────
    logger.info("Embedding with model: %s", EMBEDDING_MODEL)
    t0 = time.time()
    embeddings = embed(chunks, model_name=EMBEDDING_MODEL)
    embedding_time = time.time() - t0
    logger.info("Embedding completed in %.2f sec.", embedding_time)

    # ── Build FAISS index ─────────────────────────────────────────────────────
    store = FAISSStore(dim=len(embeddings[0]))
    store.add(embeddings, chunks)
    logger.info("FAISS index ready.")

    # ── Interactive query loop ─────────────────────────────────────────────────
    while True:
        query = input("\nAsk a question (or type 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a question.")
            continue

        t0 = time.time()
        query_embedding = embed([query], model_name=EMBEDDING_MODEL)[0]
        retrieved = store.search(query_embedding, k=TOP_K)
        retrieval_time = time.time() - t0

        context = "\n".join(retrieved)
        prompt = (
            "Use the context below to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}"
        )

        logger.info("Generating answer with model: %s", LLM_MODEL)
        t0 = time.time()
        answer = generate_answer(prompt, model_name=LLM_MODEL)
        generation_time = time.time() - t0

        approx_tokens = len(answer.split())
        tokens_per_second = approx_tokens / generation_time if generation_time > 0 else 0
        total_time = time.time() - total_start

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)
        print("=" * 60)

        save_experiment([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            EMBEDDING_MODEL, LLM_MODEL, VECTOR_DB,
            CHUNK_SIZE, CHUNK_OVERLAP, chunk_count, TOP_K,
            round(embedding_time, 4), round(retrieval_time, 4),
            round(generation_time, 4), round(total_time, 4),
            len(prompt), len(answer), round(tokens_per_second, 4),
        ])

        logger.info("Result logged.")

    logger.info("Session ended.")


if __name__ == "__main__":
    main()