import os
import shutil
import time
import base64
import streamlit as st

from experiment_logger import log_experiment
from rag_engine import build_index, query_index, stream_query_index
from models.llm import stream_answer
from config import STORAGE_PATH, UPLOAD_PATH
import json

os.makedirs(UPLOAD_PATH, exist_ok=True)
os.makedirs(STORAGE_PATH, exist_ok=True)
os.makedirs("results", exist_ok=True)

st.set_page_config(page_title="DIRS", layout="wide", page_icon="📄")

# ── Global Styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .doc-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: border-color 0.2s;
    }
    .doc-card:hover { border-color: #89b4fa; }
    .doc-card h4 { margin: 0 0 8px 0; color: #cdd6f4; font-size: 15px; }
    .doc-card .meta { font-size: 12px; color: #a6adc8; line-height: 1.8; }
    .doc-card .badge {
        display: inline-block;
        background: #313244;
        color: #89b4fa;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 11px;
        margin-right: 4px;
    }
    .chat-user {
        background: #313244;
        border-radius: 10px 10px 2px 10px;
        padding: 10px 14px;
        margin: 8px 0 4px auto;
        max-width: 80%;
        color: #cdd6f4;
        font-size: 14px;
        width: fit-content;
        margin-left: auto;
    }
    .chat-assistant {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 10px 10px 10px 2px;
        padding: 10px 14px;
        margin: 4px auto 8px 0;
        max-width: 85%;
        color: #cdd6f4;
        font-size: 14px;
    }
    .metric-pill {
        display: inline-block;
        background: #181825;
        border: 1px solid #313244;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 12px;
        color: #a6adc8;
        margin-right: 6px;
        margin-top: 4px;
    }
    .step-done { color: #a6e3a1; }
    .step-active { color: #89b4fa; font-weight: bold; }
    .step-pending { color: #585b70; }
    .cursor {
        display: inline-block;
        width: 2px;
        background: #89b4fa;
        margin-left: 2px;
        border-radius: 1px;
        animation: blink 0.7s step-end infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 Document Intelligence & Retrieval System")

menu = st.sidebar.radio("Select Role", ["👤 Admin", "💬 User"])

# ── Session state init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_doc" not in st.session_state:
    st.session_state.last_doc = None
if "last_llm" not in st.session_state:
    st.session_state.last_llm = None


def safe_delete(path):
    """
    Delete a folder, first evicting any cached FAISS/BM25 objects that hold
    open file handles on Windows (which prevents shutil.rmtree from succeeding).
    """
    import rag_engine

    # Derive document name from path and evict all in-process caches
    doc_name = os.path.basename(path)
    rag_engine._faiss_cache.pop(doc_name, None)
    rag_engine._bm25_cache.pop(doc_name, None)
    rag_engine._chunks_cache.pop(doc_name, None)

    for _ in range(3):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            time.sleep(1)
    raise PermissionError(
        f"Could not delete '{path}' — the file is still locked.\n"
        f"Please restart the Streamlit app and try again."
    )


def get_doc_metadata(doc_name):
    """Read stored metadata.json for a document."""
    meta_path = os.path.join(STORAGE_PATH, doc_name, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    return {}


def get_doc_indexed_time(doc_name):
    """Get the creation time of the index folder."""
    folder = os.path.join(STORAGE_PATH, doc_name)
    if os.path.exists(folder):
        ts = os.path.getctime(folder)
        return time.strftime("%b %d, %Y  %H:%M", time.localtime(ts))
    return "Unknown"


def show_pdf_preview(file_bytes, filename):
    """Render PDF inline using base64 iframe."""
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{b64}"
        width="100%"
        height="500px"
        style="border: 1px solid #313244; border-radius: 8px;"
    ></iframe>
    """
    st.markdown(f"**Preview — {filename}**")
    st.markdown(pdf_display, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN PANEL
# ══════════════════════════════════════════════════════════════════════════════
if menu == "👤 Admin":

    st.header("Build Document Index")

    col_upload, col_preview = st.columns([1, 1], gap="large")

    with col_upload:
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

        embedding_model = st.selectbox(
            "Embedding Model",
            ["BGE-small", "MiniLM", "E5-small"],
            help="BGE-small is fastest; MiniLM offers the best balance."
        )

        vector_db = st.selectbox(
            "Vector Database",
            ["FAISS", "Chroma"],
            help="FAISS uses hybrid retrieval (vector + BM25). Chroma uses vector only."
        )

        force_rebuild = st.checkbox(
            "Force Rebuild",
            help="Overwrite an existing index for this document."
        )

        build_clicked = st.button("⚙️ Build Index", use_container_width=True, type="primary")

    # ── PDF Preview ───────────────────────────────────────────────────────────
    with col_preview:
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            show_pdf_preview(file_bytes, uploaded_file.name)
        else:
            st.markdown("""
            <div style="height:200px; border:1px dashed #313244; border-radius:8px;
                        display:flex; align-items:center; justify-content:center; color:#585b70;">
                PDF preview will appear here after upload
            </div>
            """, unsafe_allow_html=True)

    # ── Build Logic with Step Progress ────────────────────────────────────────
    if build_clicked:
        if uploaded_file is None:
            st.error("Please upload a PDF first.")
        else:
            # Save uploaded bytes to disk (already read above for preview)
            file_path = os.path.join(UPLOAD_PATH, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(file_bytes)

            document_name = os.path.splitext(uploaded_file.name)[0]
            document_folder = os.path.join(STORAGE_PATH, document_name)

            if os.path.exists(document_folder):
                if force_rebuild:
                    time.sleep(1)
                    try:
                        safe_delete(document_folder)
                    except PermissionError:
                        st.error("File is in use. Restart the app and try again.")
                        st.stop()
                else:
                    st.error("Index already exists. Enable 'Force Rebuild' to overwrite.")
                    st.stop()

            # ── Step-by-step progress UI ──────────────────────────────────────
            steps = [
                "📖 Extracting text from PDF",
                "✂️  Chunking text",
                "🔢 Generating embeddings",
                "💾 Saving index to disk",
            ]

            progress_bar = st.progress(0)
            status_box = st.empty()

            def render_steps(current):
                lines = []
                for i, s in enumerate(steps):
                    if i < current:
                        lines.append(f'<div class="step-done">✅ {s}</div>')
                    elif i == current:
                        lines.append(f'<div class="step-active">⏳ {s} ...</div>')
                    else:
                        lines.append(f'<div class="step-pending">○ {s}</div>')
                status_box.markdown(
                    "<div style='line-height:2.2; padding:10px 0'>" + "".join(lines) + "</div>",
                    unsafe_allow_html=True
                )

            try:
                render_steps(0)
                progress_bar.progress(10)

                # Patch build_index to report steps via callbacks
                build_index(
                    pdf_path=file_path,
                    embedding_model=embedding_model,
                    vector_db=vector_db,
                    on_step=lambda step: (
                        render_steps(step),
                        progress_bar.progress(25 * (step + 1))
                    )
                )

                progress_bar.progress(100)
                render_steps(len(steps))
                st.success(f"✅ Index built for **{document_name}**!")

            except Exception as e:
                st.error(str(e))

    # ── Existing Indexes ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Indexed Documents")

    if os.path.exists(STORAGE_PATH):
        docs = [d for d in os.listdir(STORAGE_PATH)
                if os.path.isdir(os.path.join(STORAGE_PATH, d))]
    else:
        docs = []

    if not docs:
        st.info("No documents indexed yet.")
    else:
        for doc in docs:
            meta = get_doc_metadata(doc)
            indexed_at = get_doc_indexed_time(doc)
            chunks = meta.get("chunk_count", "—")
            emb = meta.get("embedding_model", "—")
            vdb = meta.get("vector_db", "—")

            col_info, col_del = st.columns([5, 1])
            with col_info:
                st.markdown(f"""
                <div class="doc-card">
                    <h4>📁 {doc}</h4>
                    <div class="meta">
                        <span class="badge">{emb}</span>
                        <span class="badge">{vdb}</span>
                        <span class="badge">{chunks} chunks</span><br/>
                        🕒 Indexed on {indexed_at}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_del:
                st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
                if st.button("🗑️ Delete", key=f"del_{doc}"):
                    safe_delete(os.path.join(STORAGE_PATH, doc))
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# USER PANEL
# ══════════════════════════════════════════════════════════════════════════════
if menu == "💬 User":

    if not os.path.exists(STORAGE_PATH) or not os.listdir(STORAGE_PATH):
        st.warning("No documents indexed yet. Ask an Admin to upload and index a document.")
        st.stop()

    documents = [d for d in os.listdir(STORAGE_PATH)
                 if os.path.isdir(os.path.join(STORAGE_PATH, d))]

    if not documents:
        st.warning("No documents available.")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        selected_doc = st.selectbox(
            "Document",
            documents,
            index=documents.index(st.session_state.last_doc)
                  if st.session_state.last_doc in documents else 0
        )

        llm_model = st.selectbox(
            "LLM Model",
            ["llama3:latest", "qwen2.5:7b", "gemma:7b"],
            index=["llama3:latest", "qwen2.5:7b", "gemma:7b"].index(st.session_state.last_llm)
                  if st.session_state.last_llm in ["llama3:latest", "qwen2.5:7b", "gemma:7b"] else 0
        )

        top_k = st.slider(
            "Chunks to Retrieve (TOP_K)",
            min_value=1, max_value=10, value=3,
            help="More chunks = more context for the LLM, but slower response."
        )

        meta = get_doc_metadata(selected_doc)
        indexed_at = get_doc_indexed_time(selected_doc)
        st.markdown("---")
        st.markdown("**Selected Document**")
        st.markdown(f"""
        <div class="doc-card">
            <h4>📁 {selected_doc}</h4>
            <div class="meta">
                <span class="badge">{meta.get('embedding_model','—')}</span>
                <span class="badge">{meta.get('vector_db','—')}</span>
                <span class="badge">{meta.get('chunk_count','—')} chunks</span><br/>
                🕒 {indexed_at}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if selected_doc != st.session_state.last_doc:
            st.session_state.chat_history = []
            st.session_state.last_doc = selected_doc

        st.session_state.last_llm = llm_model

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ── CSS ───────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
        /* Hide Streamlit default title & main padding in chat mode */
        [data-testid="stAppViewContainer"] > .main > div:first-child { padding-top: 0 !important; }

        /* ── Message rows ── */
        .msg-row {
            display: flex;
            align-items: flex-start;
            gap: 14px;
            padding: 20px 0;
            border-bottom: 1px solid #2a2a2a;
            max-width: 760px;
            margin: 0 auto;
            width: 100%;
        }
        .msg-row.user-row      { flex-direction: row-reverse; }
        .msg-row.assistant-row { flex-direction: row; }

        .avatar {
            width: 34px; height: 34px;
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 16px; flex-shrink: 0; margin-top: 2px;
        }
        .avatar.user-av { background: #19c37d; }
        .avatar.bot-av  { background: #444654; }

        .msg-text {
            font-size: 15px; line-height: 1.75;
            color: #ececec; flex: 1;
            white-space: pre-wrap; word-break: break-word;
        }
        .msg-text.user-text { text-align: right; }
        .msg-text.bot-text  { text-align: left;  }

        /* Empty state */
        .empty-state {
            text-align: center;
            padding: 120px 0 60px 0;
            color: #555; font-size: 15px;
            max-width: 760px; margin: 0 auto;
        }
        .empty-state h2 {
            font-size: 28px; color: #888;
            margin-bottom: 8px; font-weight: 600;
        }

        /* Blinking cursor */
        .cursor {
            display: inline-block;
            width: 9px; height: 15px;
            background: #19c37d;
            margin-left: 2px; border-radius: 1px;
            vertical-align: middle;
            animation: blink 0.6s step-end infinite;
        }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

        /* ── Bottom input bar — fixed to viewport bottom ── */
        .input-outer {
            position: fixed;
            bottom: 0; left: 0; right: 0;
            z-index: 999;
            background: #212121;
            border-top: 1px solid #2a2a2a;
            padding: 14px 24px 18px 24px;
        }
        .input-inner {
            max-width: 760px;
            margin: 0 auto;
            display: flex;
            gap: 10px;
            align-items: center;
        }

        /* Streamlit form tweaks */
        div[data-testid="stForm"] { border: none !important; padding: 0 !important; }
        div[data-testid="stBottom"] { padding-bottom: 0 !important; }

        /* Input field */
        div[data-testid="stForm"] input[type="text"] {
            background: #2f2f2f !important;
            border: 1px solid #3a3a3a !important;
            border-radius: 12px !important;
            color: #ececec !important;
            font-size: 15px !important;
            padding: 14px 18px !important;
            height: 52px !important;
        }
        div[data-testid="stForm"] input[type="text"]:focus {
            border-color: #19c37d !important;
            box-shadow: 0 0 0 2px rgba(25,195,125,0.18) !important;
            outline: none !important;
        }

        /* Send button */
        div[data-testid="stForm"] button[kind="primaryFormSubmit"] {
            background: #19c37d !important;
            border: none !important;
            border-radius: 10px !important;
            color: #fff !important;
            font-size: 15px !important;
            height: 52px !important;
            font-weight: 600 !important;
        }
        div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover {
            background: #15a86a !important;
        }

        /* Push chat content above fixed input bar */
        .chat-scroll-area { padding-bottom: 100px; }
    </style>
    """, unsafe_allow_html=True)

    # ── STEP 1: Reserve the chat area container (renders visually above input) ─
    chat_area = st.container()

    # ── STEP 2: Input Area ─────────────────────────────────────────────
    with st.bottom:
        with st.form(key="chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([6, 1])

            with col_input:
                question = st.text_input(
                    "q",
                    placeholder=f"Ask anything about '{selected_doc}'...",
                    label_visibility="collapsed",
                    key="question_input",
                )

            with col_send:
                send = st.form_submit_button(
                    "Send ➤",
                    type="primary",
                    use_container_width=True
                )

    # ── STEP 3: Fill the chat area (above the input bar) ──────────────────────
    with chat_area:
        st.markdown('<div class="chat-scroll-area">', unsafe_allow_html=True)

        if not st.session_state.chat_history and not send:
            st.markdown(f"""
            <div class="empty-state">
                <h2>📄 {selected_doc}</h2>
                <p>Ask a question to get started.</p>
            </div>
            """, unsafe_allow_html=True)

        # Render history
        for entry in st.session_state.chat_history:
            st.markdown(f"""
            <div class="msg-row user-row">
                <div class="avatar user-av">🧑</div>
                <div class="msg-text user-text">{entry["question"]}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="msg-row assistant-row">
                <div class="avatar bot-av">🤖</div>
                <div class="msg-text bot-text">{entry["answer"]}</div>
            </div>
            """, unsafe_allow_html=True)

            if entry.get("sources"):
                with st.expander(f"📚 View {len(entry['sources'])} source chunks"):
                    for i, chunk in enumerate(entry["sources"]):
                        st.markdown(f"**Source {i+1}**")
                        st.caption(chunk)
                        if i < len(entry["sources"]) - 1:
                            st.divider()

        # Stream new answer inside chat_area
        if send:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                try:
                    # User bubble
                    st.markdown(f"""
                    <div class="msg-row user-row">
                        <div class="avatar user-av">🧑</div>
                        <div class="msg-text user-text">{question}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Retrieve
                    with st.spinner("Retrieving relevant context..."):
                        prep = stream_query_index(
                            document_name=selected_doc,
                            question=question,
                            llm_model=llm_model,
                            top_k=top_k,
                        )

                    if "error" in prep:
                        st.error(prep["error"])
                    else:
                        stream_placeholder = st.empty()
                        full_answer = ""
                        t_gen_start = time.time()

                        for token in stream_answer(prep["prompt"], model_name=llm_model):
                            full_answer += token
                            stream_placeholder.markdown(f"""
                            <div class="msg-row assistant-row">
                                <div class="avatar bot-av">🤖</div>
                                <div class="msg-text bot-text">{full_answer}<span class="cursor"></span></div>
                            </div>
                            """, unsafe_allow_html=True)

                        stream_placeholder.markdown(f"""
                        <div class="msg-row assistant-row">
                            <div class="avatar bot-av">🤖</div>
                            <div class="msg-text bot-text">{full_answer}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        generation_time = time.time() - t_gen_start
                        approx_tokens = len(full_answer.split())

                        full_metrics = {
                            **prep["metrics"],
                            "generation_time":     round(generation_time, 4),
                            "total_time":          round(
                                prep["metrics"]["embedding_time"]
                                + prep["metrics"]["retrieval_time"]
                                + generation_time, 4
                            ),
                            "tokens_per_second":   round(approx_tokens / generation_time, 2)
                                                   if generation_time > 0 else 0,
                            "answer_length_chars": len(full_answer),
                        }

                        log_experiment(
                            document=selected_doc,
                            llm=llm_model,
                            embedding=full_metrics["embedding_model"],
                            vector_db=full_metrics["vector_db"],
                            metrics=full_metrics,
                        )

                        st.session_state.chat_history.append({
                            "question": question,
                            "answer":   full_answer,
                            "metrics":  full_metrics,
                            "sources":  prep["retrieved_chunks"],
                        })

                        st.rerun()

                except Exception as e:
                    st.error(str(e))

        st.markdown('</div>', unsafe_allow_html=True)