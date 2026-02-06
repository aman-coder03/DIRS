import time
import logging
import csv
import os
from datetime import datetime

from rag.pdf_loader import load_pdf
from rag.chunker import chunk_text
from models.embedding import embed
from vectorstore.faiss_store import FAISSStore
from models.llm import generate_answer


#Configuration

PDF_FOLDER = r"C:/Users/Aman Srivastava/Desktop/Programs/Projects/RAG-Project/data"
EMBEDDING_MODEL = "BGE-small"
LLM_MODEL = "llama3"
VECTOR_DB = "FAISS"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

RESULTS_FILE = "results/experiments.csv"


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def save_experiment(row):
    os.makedirs("results", exist_ok=True)

    file_exists = os.path.isfile(RESULTS_FILE)

    with open(RESULTS_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "embedding_model",
                "llm_model",
                "vector_db",
                "chunk_size",
                "chunk_overlap",
                "chunk_count",
                "top_k",
                "embedding_time",
                "retrieval_time",
                "generation_time",
                "total_time",
                "prompt_length_chars",
                "answer_length_chars",
                "tokens_per_second"
            ])

        writer.writerow(row)


def main():
    setup_logger()
    logging.info("Initializing RAG pipeline...")

    total_start = time.time()

    # ---------------- Load All PDFs ----------------
    logging.info("Loading all PDFs from folder...")
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

    if not pdf_files:
        logging.error("No PDF files found in folder.")
        return

    combined_text = ""

    for pdf_file in pdf_files:
        full_path = os.path.join(PDF_FOLDER, pdf_file)
        logging.info(f"Reading: {pdf_file}")
        text = load_pdf(full_path)
        combined_text += "\n\n" + text

    logging.info(f"Total PDFs loaded: {len(pdf_files)}")

    # ---------------- Chunking ----------------
    logging.info("Chunking combined documents...")
    chunks = chunk_text(combined_text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    chunk_count = len(chunks)
    logging.info(f"Total chunks created: {chunk_count}")

    # ---------------- Embedding ----------------
    logging.info(f"Generating embeddings using {EMBEDDING_MODEL}...")
    embed_start = time.time()
    embeddings = embed(chunks)
    embedding_time = time.time() - embed_start
    logging.info(f"Embedding time: {embedding_time:.2f} sec")

    # ---------------- Build FAISS Index ----------------
    logging.info("Building FAISS index...")
    store = FAISSStore(dim=len(embeddings[0]))
    store.add(embeddings, chunks)
    logging.info("Vector index ready.")

    # ---------------- Query Loop ----------------
    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        retrieval_start = time.time()
        query_embedding = embed([query])[0]
        retrieved = store.search(query_embedding, k=TOP_K)
        retrieval_time = time.time() - retrieval_start

        context = "\n".join(retrieved)

        prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query}
"""

        prompt_length_chars = len(prompt)

        logging.info(f"Generating response using {LLM_MODEL}...")
        generation_start = time.time()
        answer = generate_answer(prompt, model_name=LLM_MODEL)
        generation_time = time.time() - generation_start

        answer_length_chars = len(answer)
        approx_tokens_generated = len(answer.split())
        tokens_per_second = approx_tokens_generated / generation_time if generation_time > 0 else 0

        total_time = time.time() - total_start

        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(answer)
        print("=" * 60)

        save_experiment([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            EMBEDDING_MODEL,
            LLM_MODEL,
            VECTOR_DB,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
            chunk_count,
            TOP_K,
            round(embedding_time, 4),
            round(retrieval_time, 4),
            round(generation_time, 4),
            round(total_time, 4),
            prompt_length_chars,
            answer_length_chars,
            round(tokens_per_second, 4)
        ])

        logging.info("Query logged successfully.")

    logging.info("Session ended.")


if __name__ == "__main__":
    main()
