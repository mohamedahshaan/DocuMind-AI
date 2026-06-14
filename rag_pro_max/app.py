import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_engine import (
    answer_question,
    clear_index,
    get_index_stats,
    index_pdfs,
    is_indexed,
    save_uploaded_files,
)

load_dotenv()

st.set_page_config(
    page_title="DocuMind AI | High Accuracy RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp {
    background:
      radial-gradient(circle at 15% 10%, rgba(90,80,255,.22), transparent 32%),
      radial-gradient(circle at 85% 15%, rgba(255,70,170,.16), transparent 30%),
      linear-gradient(135deg, #070b16 0%, #111827 48%, #130f25 100%);
    color: #f8fafc;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(15, 23, 42, .98), rgba(17, 24, 39, .95));
    border-right: 1px solid rgba(148, 163, 184, .15);
}
.block-container { padding-top: 2.2rem; max-width: 1180px; }
.hero {
    padding: 34px 36px;
    border: 1px solid rgba(148, 163, 184, .22);
    border-radius: 28px;
    background: linear-gradient(135deg, rgba(15,23,42,.90), rgba(30,41,59,.55));
    box-shadow: 0 24px 80px rgba(0,0,0,.35);
    margin-bottom: 22px;
}
.hero h1 { font-size: 44px; line-height: 1.1; margin: 0; font-weight: 850; letter-spacing: -1.5px; }
.hero p { color: #cbd5e1; font-size: 16px; margin-top: 14px; max-width: 850px; }
.badge-row { display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }
.badge { padding:8px 12px; border-radius:999px; background: rgba(99,102,241,.18); border:1px solid rgba(129,140,248,.25); color:#dbeafe; font-size:13px; }
.metric-card {
    padding: 18px 20px; border-radius: 20px; background: rgba(15, 23, 42, .72);
    border: 1px solid rgba(148, 163, 184, .14);
}
.metric-label { color:#94a3b8; font-size:13px; }
.metric-value { font-size:34px; font-weight:800; margin-top:4px; }
.chat-card {
    border-radius: 22px; padding: 18px 20px; margin: 14px 0;
    border: 1px solid rgba(148, 163, 184, .14);
}
.user-card { background: rgba(99,102,241,.14); }
.ai-card { background: rgba(15,23,42,.72); }
.source-box {
    padding: 14px 16px; border-radius: 14px; border:1px solid rgba(148,163,184,.18);
    background: rgba(2,6,23,.55); margin-top: 10px;
}
.confidence { font-size: 12px; color:#93c5fd; padding: 4px 8px; background: rgba(59,130,246,.12); border-radius: 999px; }
.stButton > button { border-radius: 12px; font-weight: 700; }
.stTextInput > div > div > input { border-radius: 16px; }
.stFileUploader section { border-radius: 16px; background: rgba(2,6,23,.25); }
hr { border-color: rgba(148,163,184,.15); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GROQ_API_KEY", "")

with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.caption("Upload PDFs → Index → Ask accurate questions")

    api_key = st.text_input(
        "Groq API Key",
        value=st.session_state.api_key,
        type="password",
        help="Paste your key starting with gsk_. This fixes invalid API key errors without editing .env.",
    )
    st.session_state.api_key = api_key

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can upload thesis, reports, papers, or coursework PDFs.",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        index_clicked = st.button("🚀 Index", use_container_width=True, type="primary")
    with col_b:
        reset_clicked = st.button("🧹 Reset", use_container_width=True)

    if reset_clicked:
        clear_index()
        st.session_state.messages = []
        st.success("Index cleared.")
        st.rerun()

    if index_clicked:
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            with st.spinner("Reading pages, creating smart chunks, embeddings, BM25 and vector index..."):
                paths = save_uploaded_files(uploaded_files)
                stats = index_pdfs(paths)
            st.session_state.messages = []
            st.success(f"Indexed {stats['files']} file(s), {stats['pages']} pages, {stats['chunks']} chunks.")

    st.divider()
    st.markdown("### 🎯 Retrieval Settings")
    k_value = st.slider("Final sources to use", min_value=3, max_value=10, value=6)
    st.caption("Best value: 5–7. Too high can confuse the LLM.")

    st.divider()
    st.markdown("### ✅ Accuracy Pipeline")
    st.markdown("""
- BGE embeddings  
- BM25 keyword search  
- Hybrid retrieval  
- Cross-encoder reranking  
- Duplicate removal  
- Strict grounded prompt  
""")

st.markdown(
    """
<div class="hero">
  <h1>🧠 DocuMind AI — Research PDF Q&A</h1>
  <p>Ask questions from your uploaded PDF and get grounded answers with page-level citations. Built for thesis, research papers, coursework, reports and viva preparation.</p>
  <div class="badge-row">
    <span class="badge">Hybrid Search</span>
    <span class="badge">Cross-Encoder Reranking</span>
    <span class="badge">Page Citations</span>
    <span class="badge">No Guessing Prompt</span>
    <span class="badge">Multi-PDF</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

stats = get_index_stats()
m1, m2, m3, m4 = st.columns(4)
for col, label, value in [
    (m1, "Files Indexed", stats["files"]),
    (m2, "Pages Read", stats["pages"]),
    (m3, "Smart Chunks", stats["chunks"]),
    (m4, "Status", "Ready" if is_indexed() else "Not Indexed"),
]:
    with col:
        st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)

st.write("")
st.markdown("## 💬 Ask your document")

sample_cols = st.columns(4)
samples = [
    "What is the aim of this project?",
    "What methodology was used?",
    "What dataset was used?",
    "Explain the system architecture.",
]
for c, s in zip(sample_cols, samples):
    if c.button(s, use_container_width=True):
        st.session_state.pending_question = s

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-card user-card'>👤 <b>You</b><br>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='chat-card ai-card'>🤖 <b>DocuMind AI</b> <span class='confidence'>{msg.get('confidence','')}</span><br><br>{msg['content']}</div>",
            unsafe_allow_html=True,
        )
        if msg.get("sources"):
            with st.expander("📚 View exact sources used", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"<div class='source-box'><b>Source {i}</b> — Page {src['page']} · {src['section']}<br><small>{src['file_name']}</small></div>",
                        unsafe_allow_html=True,
                    )
                    st.write(src["text"][:1600] + ("..." if len(src["text"]) > 1600 else ""))

pending = st.session_state.pop("pending_question", None) if "pending_question" in st.session_state else None
question = st.chat_input("Ask about aim, methodology, dataset, accuracy, architecture, limitations...")
question = pending or question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    if not is_indexed():
        st.session_state.messages.append({"role": "assistant", "content": "Please upload and index a PDF first.", "sources": [], "confidence": "Low"})
        st.rerun()

    try:
        with st.spinner("Retrieving best pages, reranking sources, and generating grounded answer..."):
            result = answer_question(question, st.session_state.messages, k=k_value, api_key=st.session_state.api_key)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"].replace("\n", "<br>"),
                "sources": result["sources"],
                "confidence": result["confidence"],
            }
        )
    except Exception as e:
        error = str(e)
        if "401" in error or "invalid" in error.lower() or "api key" in error.lower():
            msg = "Groq API key problem. Paste a new key in the sidebar. It must start with gsk_. Then ask again."
        else:
            msg = f"Error: {error}"
        st.session_state.messages.append({"role": "assistant", "content": msg, "sources": [], "confidence": "Low"})
    st.rerun()

st.divider()
st.caption("Built with Streamlit · LangChain · ChromaDB · BGE Embeddings · BM25 · Cross-Encoder Reranking · Groq Llama")
