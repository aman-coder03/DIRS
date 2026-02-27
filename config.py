"""
config.py
---------
Central configuration for DIRS.
All tunable parameters live here — do not hardcode values in other files.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
PDF_FOLDER      = os.path.join(BASE_DIR, "data")
STORAGE_PATH    = os.path.join(BASE_DIR, "storage")
UPLOAD_PATH     = os.path.join(BASE_DIR, "uploaded_docs")
RESULTS_FILE    = os.path.join(BASE_DIR, "results", "llm_benchmark.csv")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K           = 3
VECTOR_WEIGHT   = 0.6   # weight for semantic (vector) score in hybrid fusion
BM25_WEIGHT     = 0.4   # weight for keyword (BM25) score in hybrid fusion

# ── Default Models ────────────────────────────────────────────────────────────
DEFAULT_EMBEDDING_MODEL = "BGE-small"
DEFAULT_LLM_MODEL       = "llama3:latest"
DEFAULT_VECTOR_DB       = "FAISS"

# ── LLM Generation ────────────────────────────────────────────────────────────
MAX_NEW_TOKENS  = 150