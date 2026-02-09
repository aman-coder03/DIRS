# DIRS - Document Intelligence & Retrieval System

DIRS is a modular Retrieval-Augmented Generation (RAG) system built in Python.  
It allows users to upload PDF documents and ask natural language questions about their content. The system retrieves relevant information from the documents and generates context-aware answers using a language model.

This project is designed as a foundation for building intelligent document understanding systems.

---

## What This Project Does

DIRS takes unstructured PDF documents and turns them into a searchable knowledge base.

When a document is uploaded:

1. The PDF is parsed and text is extracted.
2. The extracted text is divided into meaningful chunks.
3. Each chunk is converted into a numerical vector representation using an embedding model.
4. These vectors are stored in a vector database.
5. When a user asks a question, the system retrieves relevant chunks using a hybrid retrieval strategy:
   - Semantic search using vector similarity (FAISS / Chroma)
   - Keyword-based ranking using BM25
   - Weighted score fusion for improved relevance
6. The retrieved content is passed to a language model to generate a final answer.

This enables intelligent question-answering over private documents.

---

## Core Idea: Retrieval-Augmented Generation (RAG)

Traditional language models rely only on their pre-trained knowledge.  
DIRS enhances this by retrieving relevant information from user-provided documents before generating an answer.

This improves:
- Accuracy
- Context awareness
- Relevance
- Explainability

---

## User Roles

DIRS follows a role-based interaction model to simulate real-world document intelligence systems.

### 1. Admin (Document Manager)

The Admin is responsible for building and maintaining the knowledge base.

Responsibilities include:

- Uploading PDF documents
- Triggering document parsing and chunking
- Generating embeddings
- Updating the vector database
- Managing document storage

The Admin role ensures that the system’s knowledge base remains accurate, structured, and searchable.

### 2. User (Knowledge Consumer)

The Search User interacts with the system to retrieve information.

Responsibilities include:

- Asking natural language questions
- Retrieving relevant document context
- Receiving grounded, context-aware responses
- Exploring information from uploaded documents

The Search User does not modify the database but relies on the indexed knowledge prepared by the Admin.

This separation of responsibilities reflects real-world enterprise systems, where document ingestion and document consumption are handled independently.

---

## Screenshots

### Admin Interface
![Admin](assets/admin.png)

### User Interface
![User](assets/user.png)

### Analysis of Generated Answer
![Analysis](assets/analysis.png)

### Sources of Generated Answer
![Sources](assets/sources.png)

---

## Key Capabilities

- PDF ingestion and structured text extraction
- Intelligent text chunking for retrieval optimization
- Hybrid retrieval pipeline combining:
  - Semantic vector search (FAISS / Chroma)
  - Keyword-based ranking (BM25)
  - Weighted score fusion for improved relevance
- In-memory caching of FAISS indices and BM25 retrievers for reduced query latency
- Modular RAG architecture with clear separation of ingestion, retrieval, and generation layers
- Performance tracking and experiment logging for retrieval and generation benchmarking
- Extensible design supporting future upgrades such as reranking, advanced chunking strategies, and evaluation frameworks

---

## How It Works Internally

When a query is asked:

- The query is converted into an embedding vector.
- The vector database performs semantic retrieval using FAISS or Chroma.
- A BM25 retriever performs keyword-based ranking over the same chunks.
- Scores from both retrieval methods are normalized and combined using weighted fusion.
- The top-ranked chunks are selected as contextual evidence.
- These chunks are provided to the language model as context.
- The model generates a grounded response based only on retrieved information.

This reduces hallucinations and keeps answers tied to the document content.

---

## Why This Matters

DIRS demonstrates practical implementation of modern AI retrieval systems.  
It can be extended into:

- Research paper assistants  
- Internal enterprise knowledge systems  
- Document compliance tools  
- AI-powered search engines  

The modular design already supports hybrid retrieval and can be extended further with cross-encoder re-ranking, advanced chunking strategies, and retrieval evaluation benchmarking.

---

## Tech Stack

DIRS is built using the following technologies:

- **Python** — Core programming language
- **Ollama** — Local LLM serving framework
- **LLaMA (via Ollama)** — Large language model for answer generation
- **Qwen (via Ollama)** — Alternative LLM for contextual reasoning
- **Sentence Transformers** — Embedding generation
- **FAISS / ChromaDB** — Vector database for semantic retrieval
- **PyPDF / PDF Parsing Libraries** — Document text extraction
- **rank-bm25** — Keyword-based retrieval engine
- **NumPy / Pandas** — Data handling and experiment tracking
- **Git & GitHub** — Version control

---

## Author

Aman Srivastava  
amansri345@gmail.com