from rag_engine import build_index, query_index

# -------- ADMIN SIDE --------
build_index(
    pdf_path=r"C:\Users\Aman Srivastava\Desktop\Programs\Projects\RAG-Project\data\test1.pdf",   # change to your actual file
    embedding_model="MiniLM"
)

# -------- USER SIDE --------
answer = query_index(
    document_name="test1",       # filename without .pdf
    question="Which institution is working on ramjet tech?",
    llm_model="llama3",
    top_k=3
)

print("\nAnswer:\n")
print(answer)
