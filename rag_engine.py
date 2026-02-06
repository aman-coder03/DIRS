import os
import time
import json
import faiss
import chromadb
import numpy as np

from chromadb.config import Settings

from rag.pdf_loader import load_pdf
from rag.chunker import chunk_text
from models.embedding import embed
from models.llm import generate_answer


# Always create storage relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_PATH = os.path.join(BASE_DIR, "storage")


# ==========================================================
# ADMIN SIDE
# ==========================================================
def build_index(
    pdf_path,
    embedding_model,
    vector_db="FAISS",
    chunk_size=500,
    overlap=50
):
    """
    Builds vector index for a given PDF and saves it locally.
    """

    os.makedirs(STORAGE_PATH, exist_ok=True)

    document_name = os.path.splitext(os.path.basename(pdf_path))[0]
    document_folder = os.path.join(STORAGE_PATH, document_name)

    # 🔥 Overwrite protection
    if os.path.exists(document_folder):
        raise Exception(
            f"Index already exists for '{document_name}'. "
            f"Delete the folder to rebuild."
        )

    os.makedirs(document_folder, exist_ok=True)

    print(f"Building index for: {document_name}")
    print(f"Using Embedding Model: {embedding_model}")
    print(f"Using Vector DB: {vector_db}")

    # 1️⃣ Load PDF
    text = load_pdf(pdf_path)

    # 2️⃣ Chunk
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    # 3️⃣ Embed
    embeddings = embed(chunks, model_name=embedding_model)
    embeddings = np.array(embeddings).astype("float32")

    # ==========================================================
    # VECTOR DATABASE SELECTION
    # ==========================================================
    if vector_db == "FAISS":

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        faiss.write_index(index, os.path.join(document_folder, "index.faiss"))

    elif vector_db == "Chroma":

        chroma_client = chromadb.PersistentClient(path=document_folder)

        collection = chroma_client.get_or_create_collection("rag_collection")

        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                embeddings=[embeddings[i].tolist()],
                ids=[str(i)]
            )

    else:
        raise ValueError("Unsupported vector database selected.")

    # 4️⃣ Save chunks (for FAISS retrieval)
    with open(os.path.join(document_folder, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    # 5️⃣ Save metadata
    metadata = {
        "embedding_model": embedding_model,
        "vector_db": vector_db,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunk_count": len(chunks)
    }

    with open(os.path.join(document_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("Index built and saved successfully.")


# ==========================================================
# USER SIDE
# ==========================================================
def query_index(document_name, question, llm_model="llama3", top_k=3):
    """
    Loads stored index and answers question.
    Returns answer + retrieved chunks + performance metrics.
    """

    document_folder = os.path.join(STORAGE_PATH, document_name)

    if not os.path.exists(document_folder):
        return {"error": "Document not found."}

    total_start = time.time()

    # Load metadata
    with open(os.path.join(document_folder, "metadata.json"), "r") as f:
        metadata = json.load(f)

    embedding_model = metadata["embedding_model"]
    vector_db = metadata["vector_db"]

    # ---------------- EMBEDDING QUERY ----------------
    embed_start = time.time()
    query_embedding = embed([question], model_name=embedding_model)[0]
    query_embedding = np.array([query_embedding]).astype("float32")
    embedding_time = time.time() - embed_start

    # ---------------- RETRIEVAL ----------------
    retrieval_start = time.time()

    if vector_db == "FAISS":

        index_path = os.path.join(document_folder, "index.faiss")
        index = faiss.read_index(index_path)

        with open(os.path.join(document_folder, "chunks.json"), "r", encoding="utf-8") as f:
            chunks = json.load(f)

        distances, indices = index.search(query_embedding, top_k)
        retrieved_chunks = [chunks[i] for i in indices[0]]

    elif vector_db == "Chroma":

        chroma_client = chromadb.PersistentClient(path=document_folder)
        collection = chroma_client.get_collection("rag_collection")

        results = collection.query(
            query_embeddings=[query_embedding[0].tolist()],
            n_results=top_k
        )

        retrieved_chunks = results["documents"][0]

    else:
        return {"error": "Unsupported vector database."}

    retrieval_time = time.time() - retrieval_start

    # ---------------- GENERATION ----------------
    context = "\n".join(retrieved_chunks)

    prompt = f"""
Answer concisely in 4-5 lines using only the context.
Do not elaborate.

Context:
{context}

Question:
{question}
"""

    prompt_length_chars = len(prompt)

    generation_start = time.time()
    answer = generate_answer(prompt, model_name=llm_model)
    generation_time = time.time() - generation_start

    answer_length_chars = len(answer)

    approx_tokens_generated = len(answer.split())
    tokens_per_second = (
        approx_tokens_generated / generation_time
        if generation_time > 0 else 0
    )

    total_time = time.time() - total_start

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "metrics": {
            "embedding_time": round(embedding_time, 4),
            "retrieval_time": round(retrieval_time, 4),
            "generation_time": round(generation_time, 4),
            "total_time": round(total_time, 4),
            "tokens_per_second": round(tokens_per_second, 2),
            "prompt_length_chars": prompt_length_chars,
            "answer_length_chars": answer_length_chars,
            "vector_db": vector_db,
            "embedding_model": embedding_model
        }
    }
