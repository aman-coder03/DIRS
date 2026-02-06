import os
import shutil
import streamlit as st

from experiment_logger import log_experiment
from rag_engine import build_index, query_index

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORAGE_PATH = os.path.join(BASE_DIR, "storage")
DATA_UPLOAD_PATH = os.path.join(BASE_DIR, "uploaded_docs")

os.makedirs(DATA_UPLOAD_PATH, exist_ok=True)

st.set_page_config(page_title="DIRS", layout="wide")
st.title("Document Intelligence & Retrieval System (DIRS)")

menu = st.sidebar.radio("Select Role", ["Admin", "User"])

# ADMIN PANEL
if menu == "Admin":

    st.header("Upload and Build Index")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    embedding_model = st.selectbox(
        "Select Embedding Model",
        ["BGE-small", "MiniLM", "E5-small"]
    )

    vector_db = st.selectbox(
        "Select Vector Database",
        ["FAISS", "Chroma"]
    )

    force_rebuild = st.checkbox("Force Rebuild (Overwrite Existing Index)")

    if st.button("Build Index"):

        if uploaded_file is None:
            st.error("Please upload a PDF first.")
        else:
            file_path = os.path.join(DATA_UPLOAD_PATH, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())

            document_name = os.path.splitext(uploaded_file.name)[0]
            document_folder = os.path.join(STORAGE_PATH, document_name)

            # Overwrite handling
            if os.path.exists(document_folder):
                if force_rebuild:
                    shutil.rmtree(document_folder)
                    st.warning("Existing index deleted. Rebuilding...")
                else:
                    st.error(
                        "Index already exists. Enable 'Force Rebuild' to overwrite."
                    )
                    st.stop()

            try:
                with st.spinner("Building index..."):
                    build_index(
                        pdf_path=file_path,
                        embedding_model=embedding_model,
                        vector_db=vector_db
                    )

                st.success("Index built successfully!")

            except Exception as e:
                st.error(str(e))


# USER PANEL
if menu == "User":

    st.header("Ask Questions")

    if not os.path.exists(STORAGE_PATH):
        st.warning("No documents indexed yet.")
    else:
        documents = [
            d for d in os.listdir(STORAGE_PATH)
            if os.path.isdir(os.path.join(STORAGE_PATH, d))
        ]

        if not documents:
            st.warning("No documents available.")
        else:
            selected_doc = st.selectbox("Select Document", documents)

            llm_model = st.selectbox(
                "Select LLM",
                ["llama3:latest", "qwen2.5:7b", "gemma:7b"]
            )

            question = st.text_area("Enter your question")

            if st.button("Get Answer"):

                if question.strip() == "":
                    st.error("Please enter a question.")
                else:
                    try:
                        with st.spinner("Generating answer..."):
                            result = query_index(
                                document_name=selected_doc,
                                question=question,
                                llm_model=llm_model,
                                top_k=3   #adjustable
                            )

                        if "error" in result:
                            st.error(result["error"])
                        else:
                            answer = result["answer"]
                            metrics = result["metrics"]
                            sources = result["retrieved_chunks"]
                            
                            log_experiment(
                                document=selected_doc,
                                llm=llm_model,
                                embedding=metrics["embedding_model"],
                                vector_db=metrics["vector_db"],
                                metrics=metrics
                            )

                            # CONFIGURATION HEADER
                            st.markdown(f"**Document:** {selected_doc}")
                            st.markdown(f"**LLM:** {llm_model}")
                            st.markdown(f"**Embedding:** {metrics['embedding_model']}")
                            st.markdown(f"**Vector DB:** {metrics['vector_db']}")

                            st.markdown("---")

                            # ANSWER SECTION
                            st.subheader("Answer")
                            st.write(answer)

                            st.markdown("---")

                            # PERFORMANCE METRICS
                            st.subheader("Performance")

                            st.markdown(f"**Embedding Time:** {metrics['embedding_time']} sec")
                            st.markdown(f"**Retrieval Time:** {metrics['retrieval_time']} sec")
                            st.markdown(f"**Generation Time:** {metrics['generation_time']} sec")
                            st.markdown(f"**Total Time:** {metrics['total_time']} sec")
                            st.markdown(f"**Tokens/sec:** {metrics['tokens_per_second']}")


                            st.markdown("---")

                            # SOURCE TRANSPARENCY
                            st.subheader("Sources")

                            for i, chunk in enumerate(sources):
                                with st.expander(f"Source {i+1}"):
                                    st.write(chunk)

                    except Exception as e:
                        st.error(str(e))

