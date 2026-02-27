# DIRS — Setup & Usage Guide

This guide covers everything you need to install, configure, and run DIRS from scratch.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Setting Up Ollama & Language Models](#3-setting-up-ollama--language-models)
4. [Starting the Application](#4-starting-the-application)
5. [Using the Admin Panel](#5-using-the-admin-panel)
6. [Using the User Panel](#6-using-the-user-panel)
7. [Project Structure](#7-project-structure)
8. [Configuration](#8-configuration)
9. [Experiment Logs](#9-experiment-logs)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

Ensure the following are installed before proceeding.

| Requirement | Version | Check Command |
|---|---|---|
| Python | 3.10 or higher | `python --version` |
| pip | Latest | `pip --version` |
| Git | Any | `git --version` |
| Ollama | Latest | `ollama --version` |

### Installing Ollama

Ollama is required to run language models locally.

**Windows / macOS:**
Download and run the installer from [https://ollama.com/download](https://ollama.com/download)

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

---

## 2. Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/DIRS.git
cd DIRS
```

### Step 2 — Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt once activated.

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages including Streamlit, FAISS, ChromaDB, Sentence Transformers, Ollama client, and others.

> **Note:** The first run will also download the embedding models from Hugging Face. This requires an internet connection and may take a few minutes depending on your connection speed.

---

## 3. Setting Up Ollama & Language Models

DIRS uses Ollama to serve LLMs locally. You must pull the models before using the application.

### Start the Ollama Server

```bash
ollama serve
```

Keep this terminal open. Ollama must be running whenever you use DIRS.

### Pull Language Models

Open a **new terminal** and run the following commands to download the supported models:

```bash
# LLaMA 3 (recommended, ~4.7 GB)
ollama pull llama3:latest

# Qwen 2.5 7B (~4.4 GB)
ollama pull qwen2.5:7b

# Gemma 7B (~5.0 GB)
ollama pull gemma:7b
```

You can pull only the models you plan to use. At minimum, pull `llama3:latest`.

### Verify Models are Available

```bash
ollama list
```

You should see the pulled models listed in the output.

---

## 4. Starting the Application

Make sure:
- Your virtual environment is **activated**
- The **Ollama server is running** (`ollama serve`)

Then start the Streamlit app:

```bash
streamlit run app.py
```

The application will open automatically in your browser at:

```
http://localhost:8501
```

To stop the application, press `Ctrl + C` in the terminal.

---

## 5. Using the Admin Panel

The Admin panel is used to upload documents and build the searchable index.

1. In the sidebar, select **Admin**.
2. Click **Browse files** and upload a PDF document.
3. Select an **Embedding Model**:
   - `BGE-small` — Fast and accurate (recommended)
   - `MiniLM` — Lightest and fastest
   - `E5-small` — Strong general-purpose performance
4. Select a **Vector Database**:
   - `FAISS` — In-memory, fast retrieval with hybrid BM25 support (recommended)
   - `Chroma` — Persistent, suitable for larger collections
5. Optionally check **Force Rebuild** if you want to overwrite an existing index for the same document.
6. Click **Build Index**.

Once built, the index is saved to the `storage/` directory and is immediately available for querying.

---

## 6. Using the User Panel

The User panel is used to ask questions against indexed documents.

1. In the sidebar, select **User**.
2. Select a **Document** from the dropdown (lists all indexed documents).
3. Select an **LLM**:
   - `llama3:latest` — Best overall quality
   - `qwen2.5:7b` — Strong reasoning, concise answers
   - `gemma:7b` — Lightweight alternative
4. Type your question in the text area.
5. Click **Get Answer**.

The system will display:
- The generated **answer**
- **Performance metrics** — embedding time, retrieval time, generation time, tokens/sec
- **Retrieved source chunks** — the exact document passages used to generate the answer

---

## 7. Project Structure

```
DIRS/
│
├── app.py                  # Streamlit web interface
├── rag_engine.py           # Core RAG pipeline (build & query)
├── experiment_logger.py    # CSV experiment logging
├── main.py                 # CLI entry point (alternative to Streamlit)
├── requirements.txt        # Python dependencies
│
├── rag/
│   ├── pdf_loader.py       # PDF text extraction
│   ├── chunker.py          # Text chunking
│   └── bm25_retriever.py   # BM25 keyword retrieval
│
├── models/
│   ├── embedding.py        # Sentence Transformer embedding
│   └── llm.py              # Ollama LLM interface
│
├── vectorstore/
│   └── faiss_store.py      # FAISS vector store wrapper
│
├── storage/                # Auto-created: stores all built indices
│   └── <document_name>/
│       ├── index.faiss
│       ├── chunks.json
│       ├── tokenized_chunks.json
│       └── metadata.json
│
├── results/                # Auto-created: experiment logs
│   └── llm_benchmark.csv
│
├── uploaded_docs/          # Auto-created: stores uploaded PDFs
└── assets/                 # Screenshots for README
```

---

## 8. Configuration

The following parameters can be adjusted directly in `rag_engine.py`:

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `500` | Number of characters per chunk |
| `overlap` | `50` | Overlap between consecutive chunks |
| `top_k` | `3` | Number of chunks retrieved per query |
| Vector weight | `0.6` | Weight given to semantic (vector) score in hybrid fusion |
| BM25 weight | `0.4` | Weight given to keyword (BM25) score in hybrid fusion |

In `app.py`, `top_k` is set at line:
```python
result = query_index(..., top_k=3)
```

Change this value to retrieve more or fewer source chunks per query.

---

## 9. Experiment Logs

Every query automatically logs performance data to `results/llm_benchmark.csv`.

Logged fields include:

| Field | Description |
|---|---|
| `timestamp` | Date and time of the query |
| `document` | Document queried |
| `llm` | LLM model used |
| `embedding_model` | Embedding model used |
| `vector_db` | Vector database used |
| `embedding_time` | Time to embed the query (seconds) |
| `retrieval_time` | Time to retrieve chunks (seconds) |
| `generation_time` | Time for LLM to generate answer (seconds) |
| `total_time` | End-to-end pipeline time (seconds) |
| `tokens_per_second` | Approximate generation speed |
| `prompt_length_chars` | Total prompt size in characters |
| `answer_length_chars` | Answer length in characters |

This log is useful for benchmarking different model and database combinations.

---

## 10. Troubleshooting

**`ollama: command not found`**
Ollama is not installed or not on your PATH. Re-install from [https://ollama.com/download](https://ollama.com/download) and restart your terminal.

**`Connection refused` when querying LLM**
The Ollama server is not running. Open a terminal and run:
```bash
ollama serve
```

**`Model not found` error**
The selected LLM has not been pulled. Run the appropriate pull command:
```bash
ollama pull llama3:latest
```

**`Index already exists` error in Admin panel**
An index for this document already exists. Enable the **Force Rebuild** checkbox to overwrite it.

**Embedding model download is slow**
Embedding models are downloaded from Hugging Face on first use. This is a one-time download per model. Subsequent runs load from the local cache.

**`No documents available` in User panel**
No index has been built yet. Switch to the Admin panel and build an index first.

**`streamlit: command not found`**
Your virtual environment is not activated. Run:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Then re-run `streamlit run app.py`.

---

*For questions or contributions, contact amansri345@gmail.com*
