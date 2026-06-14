"""
app.py — DocuMind AI | Upgraded UI
- API key loaded from .env only (no sidebar input)
- PDF upload at center top with drag & drop
- Clean chatbot reply style
"""

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

# Load API key from .env file — no UI input needed
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #070b16 0%, #111827 50%, #130f25 100%);
    color: #f1f5f9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1729 0%, #111827 100%);
    border-right: 1px solid rgba(148,163,184,0.12);
}

/* Remove default top padding */
.block-container { padding-top: 1.5rem; max-width: 1100px; }

/* ── Upload zone ── */
.upload-zone {
    background: rgba(15,23,42,0.7);
    border: 1.5px dashed rgba(99,102,241,0.45);
    border-radius: 20px;
    padding: 28px 32px;
    margin-bottom: 24px;
    text-align: center;
}
.upload-zone h2 {
    font-size: 26px;
    font-weight: 800;
    margin: 0 0 6px;
    letter-spacing: -0.8px;
}
.upload-zone p { color: #94a3b8; font-size: 14px; margin: 0 0 18px; }

/* ── Stat cards ── */
.stat-row { display: flex; gap: 12px; margin-bottom: 20px; }
.stat-card {
    flex: 1;
    background: rgba(15,23,42,0.65);
    border: 1px solid rgba(148,163,184,0.12);
    border-radius: 16px;
    padding: 14px 18px;
}
.stat-label { font-size: 12px; color: #64748b; margin-bottom: 4px; }
.stat-value { font-size: 26px; font-weight: 800; color: #f1f5f9; }

/* ── Chat messages ── */
.chat-wrap { display: flex; flex-direction: column; gap: 16px; margin-bottom: 20px; }

/* User bubble — right aligned */
.msg-user {
    display: flex;
    justify-content: flex-end;
}
.msg-user .bubble {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: #fff;
    border-radius: 20px 20px 4px 20px;
    padding: 12px 18px;
    max-width: 70%;
    font-size: 14px;
    line-height: 1.6;
    box-shadow: 0 4px 20px rgba(79,70,229,0.25);
}

/* AI bubble — left aligned */
.msg-ai {
    display: flex;
    justify-content: flex-start;
    gap: 10px;
    align-items: flex-start;
}
.msg-ai .avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1e40af, #4f46e5);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    margin-top: 2px;
}
.msg-ai .bubble {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(148,163,184,0.15);
    color: #e2e8f0;
    border-radius: 20px 20px 20px 4px;
    padding: 14px 18px;
    max-width: 75%;
    font-size: 14px;
    line-height: 1.7;
    box-shadow: 0 4px 24px rgba(0,0,0,0.2);
}
.msg-ai .bubble .ai-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}
.ai-name { font-weight: 700; font-size: 13px; color: #818cf8; }
.conf-badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 999px;
    background: rgba(59,130,246,0.15);
    color: #93c5fd;
    border: 1px solid rgba(59,130,246,0.2);
}

/* Source box */
.source-chip {
    display: inline-block;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 999px;
    background: rgba(99,102,241,0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99,102,241,0.2);
    margin: 4px 4px 0 0;
}

/* Sample question buttons */
.stButton > button {
    border-radius: 12px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    background: rgba(15,23,42,0.6) !important;
    border: 1px solid rgba(148,163,184,0.18) !important;
    color: #cbd5e1 !important;
    padding: 8px 12px !important;
}
.stButton > button:hover {
    background: rgba(99,102,241,0.18) !important;
    border-color: rgba(99,102,241,0.35) !important;
    color: #e0e7ff !important;
}

/* Chat input */
[data-testid="stChatInput"] > div {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(148,163,184,0.2) !important;
    border-radius: 16px !important;
}

/* Sidebar section headers */
.sidebar-section {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #64748b;
    text-transform: uppercase;
    margin: 16px 0 8px;
}

/* File uploader */
[data-testid="stFileUploader"] section {
    border-radius: 14px !important;
    background: rgba(2,6,23,0.3) !important;
}

hr { border-color: rgba(148,163,184,0.12) !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── SIDEBAR — no API key field ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.caption("Upload PDFs → Index → Ask accurate questions")

    # Retrieval settings
    st.markdown('<div class="sidebar-section">Retrieval Settings</div>', unsafe_allow_html=True)
    k_value = st.slider("Final sources to use", min_value=3, max_value=10, value=6)
    st.caption("Best value: 5–7. Too high can confuse the LLM.")

    st.divider()

    # Accuracy pipeline info
    st.markdown('<div class="sidebar-section">Accuracy Pipeline</div>', unsafe_allow_html=True)
    st.markdown("""
- ✅ BGE embeddings
- ✅ BM25 keyword search
- ✅ Hybrid retrieval
- ✅ Cross-encoder reranking
- ✅ Duplicate removal
- ✅ Strict grounded prompt
""")

    st.divider()

    # Index stats
    stats = get_index_stats()
    st.markdown('<div class="sidebar-section">Index Stats</div>', unsafe_allow_html=True)
    st.markdown(f"📄 **{stats['files']}** files · **{stats['pages']}** pages · **{stats['chunks']}** chunks")
    st.markdown(f"**Status:** {'🟢 Ready' if is_indexed() else '🔴 Not indexed'}")

# ── MAIN CONTENT ───────────────────────────────────────────────────────────────

# ── 1. PDF UPLOAD — center top ────────────────────────────────────────────────
st.markdown("""
<div class="upload-zone">
  <h2>🧠 DocuMind AI</h2>
  <p>Upload your PDF → Index → Ask anything with page-level citations</p>
</div>
""", unsafe_allow_html=True)

# PDF uploader inside a clean container
with st.container():
    uploaded_files = st.file_uploader(
        "Drag & drop your PDF here, or click to browse",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        index_btn = st.button("🚀 Index Document", type="primary", use_container_width=True)
    with col2:
        reset_btn = st.button("🧹 Reset Index", use_container_width=True)
    with col3:
        # Show indexed file count
        stats = get_index_stats()
        st.markdown(f"<div style='padding:8px;text-align:center;color:#64748b;font-size:13px;'>{stats['files']} file(s) indexed</div>", unsafe_allow_html=True)

    # Handle index button
    if index_btn:
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        elif not GROQ_API_KEY:
            st.error("GROQ_API_KEY not found in .env file. Please add it and restart.")
        else:
            with st.spinner("Reading pages, creating embeddings, building BM25 index..."):
                paths = save_uploaded_files(uploaded_files)
                stats = index_pdfs(paths)
            st.session_state.messages = []
            st.success(f"✅ Indexed {stats['files']} file(s) · {stats['pages']} pages · {stats['chunks']} chunks — Ready to chat!")
            st.rerun()

    # Handle reset button
    if reset_btn:
        clear_index()
        st.session_state.messages = []
        st.success("Index cleared. Upload a new PDF to start fresh.")
        st.rerun()

st.divider()

# ── 2. SAMPLE QUESTIONS ────────────────────────────────────────────────────────
st.markdown("#### 💡 Try these questions")
sample_cols = st.columns(4)
samples = [
    "What is the aim of this project?",
    "What methodology was used?",
    "What dataset was used?",
    "Explain the system architecture.",
]
for col, sample in zip(sample_cols, samples):
    if col.button(sample, use_container_width=True):
        st.session_state["pending_question"] = sample

# ── 3. CHAT HISTORY ────────────────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown("#### 💬 Conversation")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            # User bubble — right side
            st.markdown(f"""
<div class="msg-user">
  <div class="bubble">{msg['content']}</div>
</div>
""", unsafe_allow_html=True)
        else:
            # AI bubble — left side with avatar
            confidence = msg.get("confidence", "")
            answer = msg.get("content", "").replace("\n", "<br>")
            sources = msg.get("sources", [])

            st.markdown(f"""
<div class="msg-ai">
  <div class="avatar">🤖</div>
  <div class="bubble">
    <div class="ai-header">
      <span class="ai-name">DocuMind AI</span>
      <span class="conf-badge">{confidence}</span>
    </div>
    {answer}
  </div>
</div>
""", unsafe_allow_html=True)

            # Source chips below the bubble
            if sources:
                with st.expander("📚 View exact sources used", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f"<div class='source-box' style='padding:12px 14px;border-radius:12px;border:1px solid rgba(148,163,184,0.15);background:rgba(2,6,23,0.5);margin-bottom:8px;'>"
                            f"<b>Source {i}</b> — Page {src['page']} · {src['section']}<br>"
                            f"<small style='color:#64748b;'>{src['file_name']}</small></div>",
                            unsafe_allow_html=True,
                        )
                        st.write(src["text"][:1600] + ("..." if len(src["text"]) > 1600 else ""))

# ── 4. CHAT INPUT ──────────────────────────────────────────────────────────────
pending = st.session_state.pop("pending_question", None) if "pending_question" in st.session_state else None
question = st.chat_input("Ask about aim, methodology, dataset, accuracy, architecture, limitations...")
question = pending or question

if question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})

    if not is_indexed():
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Please upload and index a PDF first using the upload area above.",
            "sources": [],
            "confidence": "—"
        })
        st.rerun()

    try:
        with st.spinner("Searching document and generating grounded answer..."):
            result = answer_question(
                question,
                st.session_state.messages,
                k=k_value,
                api_key=GROQ_API_KEY   # loaded from .env, not from UI
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
        })

    except Exception as e:
        error = str(e)
        if "401" in error or "invalid" in error.lower() or "api key" in error.lower():
            msg = "❌ Groq API key error. Check your .env file — key must start with gsk_."
        else:
            msg = f"❌ Error: {error}"
        st.session_state.messages.append({
            "role": "assistant",
            "content": msg,
            "sources": [],
            "confidence": "Error"
        })

    st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · LangChain · ChromaDB · BGE Embeddings · BM25 · Cross-Encoder Reranking · Groq Llama")
