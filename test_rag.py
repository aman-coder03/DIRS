"""
test_rag.py
-----------
Unit and integration tests for the DIRS RAG pipeline.

Run all tests:
    pytest test_rag.py -v

Run a specific test:
    pytest test_rag.py::test_chunker -v
"""

import os
import shutil
import pytest

from rag.chunker import chunk_text
from rag.pdf_loader import load_pdf
from rag.bm25_retriever import BM25Retriever, tokenize
from models.embedding import embed
from rag_engine import build_index, query_index

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TEST_PDF_PATH   = os.path.join(BASE_DIR, "data", "test1.pdf")   # place a test PDF here
TEST_DOC_NAME   = "test1"
STORAGE_PATH    = os.path.join(BASE_DIR, "storage")


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChunker:
    def test_basic_chunking(self):
        text = "a" * 1000
        chunks = chunk_text(text, chunk_size=200, overlap=20)
        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)

    def test_overlap_applied(self):
        text = "abcdefghij" * 100
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        # Each chunk (except last) should share 20 chars with the next
        assert chunks[0][-20:] == chunks[1][:20]

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size=50, overlap=60)

    def test_empty_text(self):
        chunks = chunk_text("", chunk_size=100, overlap=10)
        assert chunks == []


class TestTokenizer:
    def test_lowercase(self):
        assert tokenize("Hello World") == ["hello", "world"]

    def test_punctuation_removed(self):
        assert tokenize("hello, world!") == ["hello", "world"]

    def test_empty_string(self):
        assert tokenize("") == []


class TestBM25Retriever:
    DOCS = [
        "The ramjet engine uses supersonic combustion.",
        "Neural networks are used in image classification.",
        "FAISS enables fast vector similarity search.",
    ]

    def test_retrieval_returns_correct_count(self):
        retriever = BM25Retriever(self.DOCS)
        results = retriever.retrieve("ramjet supersonic", top_k=2)
        assert len(results) == 2

    def test_top_result_is_relevant(self):
        retriever = BM25Retriever(self.DOCS)
        results = retriever.retrieve("ramjet engine", top_k=1)
        assert "ramjet" in results[0][0].lower()

    def test_scores_are_non_negative(self):
        retriever = BM25Retriever(self.DOCS)
        results = retriever.retrieve("search", top_k=3)
        assert all(score >= 0 for _, score in results)


class TestEmbedding:
    def test_embedding_shape(self):
        vectors = embed(["Hello world"], model_name="MiniLM")
        assert len(vectors) == 1
        assert len(vectors[0]) > 0

    def test_embedding_multiple_texts(self):
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        vectors = embed(texts, model_name="MiniLM")
        assert len(vectors) == 3

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError):
            embed(["test"], model_name="NonExistentModel")


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests  (require test1.pdf in data/)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(TEST_PDF_PATH),
    reason="Test PDF not found at data/test1.pdf"
)
class TestPipeline:

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Remove test index before and after each test."""
        test_folder = os.path.join(STORAGE_PATH, TEST_DOC_NAME)
        if os.path.exists(test_folder):
            shutil.rmtree(test_folder)
        yield
        if os.path.exists(test_folder):
            shutil.rmtree(test_folder)

    def test_build_index_creates_files(self):
        build_index(TEST_PDF_PATH, embedding_model="MiniLM", vector_db="FAISS")
        folder = os.path.join(STORAGE_PATH, TEST_DOC_NAME)
        assert os.path.exists(os.path.join(folder, "index.faiss"))
        assert os.path.exists(os.path.join(folder, "chunks.json"))
        assert os.path.exists(os.path.join(folder, "metadata.json"))
        assert os.path.exists(os.path.join(folder, "tokenized_chunks.json"))

    def test_build_index_duplicate_raises(self):
        build_index(TEST_PDF_PATH, embedding_model="MiniLM", vector_db="FAISS")
        with pytest.raises(FileExistsError):
            build_index(TEST_PDF_PATH, embedding_model="MiniLM", vector_db="FAISS")

    def test_query_returns_answer(self):
        build_index(TEST_PDF_PATH, embedding_model="MiniLM", vector_db="FAISS")
        result = query_index(TEST_DOC_NAME, "What is this document about?", top_k=2)
        assert "answer" in result
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_query_returns_metrics(self):
        build_index(TEST_PDF_PATH, embedding_model="MiniLM", vector_db="FAISS")
        result = query_index(TEST_DOC_NAME, "Summarize the content.", top_k=2)
        metrics = result["metrics"]
        for key in ["embedding_time", "retrieval_time", "generation_time", "total_time"]:
            assert key in metrics
            assert metrics[key] >= 0

    def test_query_nonexistent_document(self):
        result = query_index("nonexistent_doc", "Any question?")
        assert "error" in result