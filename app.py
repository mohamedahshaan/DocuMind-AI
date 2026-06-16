"""
app.py - DocuMind AI
- Clean sidebar with Control Panel info only (no API key input)
- Center top PDF upload with drag and drop
- After indexing, auto-generates 6 document-specific questions using Groq
- Dark and light mode toggle
- Chatbot style conversation (user right, AI left)
- Page-level source citations
"""

import os
import json
import requests

import streamlit as st
from dotenv import load_dotenv

# Page configuration must be the very first Streamlit call
st.set_page_config(
    page_title="DocuMind AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from rag_engine import (
        answer_question,
        clear_index,
        get_index_stats,
        index_pdfs,
        is_indexed,
        save_uploaded_files,
    )
except Exception as e:
    st.error(
        "Failed to load the RAG engine. This usually means a required "
        "package is missing or failed to install.\n\n"
        f"**Error:** {e}\n\n"
        "Try running: `pip install -r requirements.txt --break-system-packages` "
        "(or without that flag if you are using a virtual environment), "
        "then restart the app."
    )
    st.stop()

# Load GROQ_API_KEY from .env file - no UI input needed
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Theme state must exist before we render the CSS block below
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# Global CSS - handles both dark and light mode styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }

/* Dark mode - default Streamlit dark background */
.stApp {
    background: linear-gradient(135deg, #070b16 0%, #111827 50%, #130f25 100%);
    color: #f1f5f9;
}

/* Sidebar - only Control Panel content, no extra sections */
[data-testid="stSidebar"] {
    background: #0f1729;
    border-right: 0.5px solid rgba(148,163,184,0.1);
}

/* Limit main content width for readability */
.block-container { padding-top: 1.2rem; max-width: 1080px; }

/* Upload zone - dashed bordered box at center top */
.upload-zone {
    background: rgba(15,23,42,0.7);
    border: 1.5px dashed rgba(83,74,183,0.55);
    border-radius: 16px;
    padding: 22px 28px;
    margin-bottom: 16px;
    text-align: center;
}
.upload-zone h2 {
    font-size: 22px;
    font-weight: 600;
    margin: 0 0 5px;
    color: #f1f5f9;
    letter-spacing: -0.5px;
}
.upload-zone p {
    color: #64748b;
    font-size: 13px;
    margin: 0;
}

/* Suggested questions grid - shown after indexing */
.suggest-label {
    font-size: 12px;
    font-weight: 500;
    color: #7F77DD;
    margin-bottom: 8px;
    letter-spacing: 0.02em;
}

/* User chat bubble - right side, purple */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 6px 0;
}
.msg-user .bubble {
    background: #534AB7;
    color: #EEEDFE;
    border-radius: 14px 14px 3px 14px;
    padding: 10px 15px;
    max-width: 68%;
    font-size: 13px;
    line-height: 1.55;
}

/* AI chat bubble - left side with avatar */
.msg-ai {
    display: flex;
    justify-content: flex-start;
    gap: 8px;
    align-items: flex-start;
    margin: 6px 0;
}
.msg-ai .avatar {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background: #EEEDFE;
    border: 0.5px solid #AFA9EC;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 600;
    color: #534AB7;
    flex-shrink: 0;
    margin-top: 2px;
}
.msg-ai .bubble {
    background: rgba(15,23,42,0.75);
    border: 0.5px solid rgba(148,163,184,0.12);
    color: #cbd5e1;
    border-radius: 14px 14px 14px 3px;
    padding: 11px 15px;
    max-width: 76%;
    font-size: 13px;
    line-height: 1.65;
}
.ai-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 6px;
}
.ai-name {
    font-size: 11px;
    font-weight: 600;
    color: #7F77DD;
}
.conf-badge {
    font-size: 10px;
    padding: 1px 7px;
    border-radius: 999px;
    background: #EEEDFE;
    color: #3C3489;
    border: 0.5px solid #AFA9EC;
}
.src-link {
    font-size: 11px;
    color: #534AB7;
    margin-top: 6px;
    display: flex;
    align-items: center;
    gap: 4px;
}

/* Streamlit button override */
.stButton > button {
    border-radius: 10px !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    background: rgba(15,23,42,0.5) !important;
    border: 0.5px solid rgba(148,163,184,0.15) !important;
    color: #cbd5e1 !important;
}
.stButton > button:hover {
    background: rgba(83,74,183,0.15) !important;
    border-color: rgba(83,74,183,0.3) !important;
    color: #e0e7ff !important;
}

/* Chat input field */
[data-testid="stChatInput"] > div {
    background: rgba(15,23,42,0.75) !important;
    border: 0.5px solid rgba(148,163,184,0.18) !important;
    border-radius: 14px !important;
}

/* File uploader area */
[data-testid="stFileUploader"] section {
    border-radius: 12px !important;
    background: rgba(2,6,23,0.3) !important;
}

/* Divider line */
hr { border-color: rgba(148,163,184,0.1) !important; }

/* Sidebar label style */
.cp-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #475569;
    text-transform: uppercase;
    margin: 12px 0 6px;
}

/* Sidebar stat card */
.cp-stat {
    background: rgba(2,6,23,0.4);
    border: 0.5px solid rgba(148,163,184,0.1);
    border-radius: 10px;
    padding: 8px 11px;
}
.cp-stat-label { font-size: 10px; color: #475569; margin-bottom: 2px; }
.cp-stat-value { font-size: 18px; font-weight: 600; color: #e2e8f0; }

/* Sidebar file chip */
.cp-file {
    background: rgba(2,6,23,0.4);
    border: 0.5px solid rgba(148,163,184,0.1);
    border-radius: 10px;
    padding: 7px 10px;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 4px;
}
.cp-file-icon {
    width: 24px;
    height: 24px;
    border-radius: 5px;
    background: #EEEDFE;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #534AB7;
    font-size: 12px;
    flex-shrink: 0;
}
.cp-file-name { font-size: 11px; font-weight: 500; color: #cbd5e1; }
.cp-file-size { font-size: 10px; color: #475569; }

/* Pipeline checklist items */
.cp-pip-item {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 11px;
    color: #64748b;
    padding: 2px 0;
}
.cp-pip-dot {
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #1D9E75;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)

# Light mode override - applied on top of the base (dark) styles when the
# header toggle is switched off
if not st.session_state.dark_mode:
    st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #f5f3ff 100%);
    color: #0f172a;
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 0.5px solid rgba(15,23,42,0.08);
}
.upload-zone {
    background: rgba(255,255,255,0.7);
    border-color: rgba(83,74,183,0.35);
}
.upload-zone h2 { color: #0f172a; }
.upload-zone p { color: #64748b; }
.msg-ai .bubble {
    background: rgba(255,255,255,0.85);
    border-color: rgba(15,23,42,0.08);
    color: #1e293b;
}
.msg-ai .avatar {
    background: #EEEDFE;
    border-color: #AFA9EC;
    color: #534AB7;
}
.cp-stat, .cp-file, [data-testid="stFileUploader"] section {
    background: rgba(241,245,249,0.7) !important;
    border-color: rgba(15,23,42,0.08) !important;
}
.cp-stat-value { color: #0f172a; }
.cp-file-name { color: #1e293b; }
.stButton > button {
    background: rgba(241,245,249,0.8) !important;
    border-color: rgba(15,23,42,0.1) !important;
    color: #334155 !important;
}
.stButton > button:hover {
    background: rgba(83,74,183,0.08) !important;
    border-color: rgba(83,74,183,0.3) !important;
    color: #3C3489 !important;
}
[data-testid="stChatInput"] > div {
    background: rgba(255,255,255,0.85) !important;
    border-color: rgba(15,23,42,0.1) !important;
}
hr { border-color: rgba(15,23,42,0.08) !important; }
</style>
""", unsafe_allow_html=True)

# FRESH START
# Every time the Streamlit process is (re)started - e.g. `streamlit run app.py` -
# wipe any previously indexed PDFs, ChromaDB data, and chat history so the app
# always opens in a clean "new install" state.
#
# IMPORTANT: Streamlit re-executes this entire script top-to-bottom on every
# single rerun (button click, widget change, st.rerun(), etc.) within the SAME
# Python process. A plain module-level variable or a `globals()` check gets
# reset on every one of those reruns, so it can NEVER tell "first ever launch"
# apart from "the rerun right after I just indexed a document" - that bug was
# wiping the index immediately after indexing finished.
#
# st.cache_resource runs its function body exactly once per process (the
# result is cached across every rerun, for every user, until the process
# restarts), which is exactly the "did the process just start" signal we need.
@st.cache_resource
def _run_once_per_process_boot():
    clear_index()
    return True

_run_once_per_process_boot()

# Session state initialisation
# messages - full chat history for the conversation
# suggested_questions - AI-generated questions shown after indexing
if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []


def generate_suggested_questions(api_key: str) -> list:
    """
    After indexing, this function samples the indexed document content
    and asks Groq Llama 3 to generate 6 specific questions that are
    directly relevant to what is inside the uploaded PDF.

    Returns a list of 6 question strings.
    Falls back to 6 generic questions if the API call fails.
    """
    try:
        # Connect to the persisted ChromaDB to pull sample chunks
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from pathlib import Path

        # Use the same embedding model that was used during indexing
        embed_model = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
        embeddings = HuggingFaceEmbeddings(model_name=embed_model)

        # Load the ChromaDB vector store from disk (must match rag_engine.CHROMA_DIR)
        runtime_root = Path(os.getenv("RAG_PRO_MAX_DATA_DIR", str(Path.home() / ".rag_pro_max_runtime")))
        chroma_dir = runtime_root / "data" / "chroma_db"
        db = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=embeddings
        )

        # Retrieve 8 diverse chunks covering different topics in the document
        sample_docs = db.similarity_search(
            "main topic aim methodology results findings dataset architecture",
            k=8
        )

        # Combine sampled chunk text into a short context for the prompt
        sample_text = "\n\n".join([doc.page_content[:300] for doc in sample_docs])

        # Prompt that instructs the model to return only a JSON array
        prompt = f"""You are analyzing a document. Based on these excerpts, generate exactly 6 specific
questions a reader would want to ask about this document.

Document excerpts:
{sample_text}

Rules:
- Questions must be specific to THIS document content, not generic
- Each question must be answerable from the document
- Keep each question under 12 words
- Return ONLY a JSON array of 6 strings, no extra text
- Format: ["Question 1?", "Question 2?", ...]"""

        # Call Groq API to generate questions
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 400
            },
            timeout=15
        )

        data = response.json()
        raw = data["choices"][0]["message"]["content"].strip()

        # Extract JSON array from the response text
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            questions = json.loads(raw[start:end])
            # Return up to 6 valid string questions
            return [q for q in questions if isinstance(q, str)][:6]

    except Exception:
        # Never crash the app because of suggestion failure - silently fall back
        pass

    # Fallback generic questions if the API call or parsing fails
    return [
        "What is the main aim of this project?",
        "What methodology was used?",
        "What dataset was used for training?",
        "Explain the system architecture.",
        "What were the final accuracy results?",
        "What are the limitations of this project?",
    ]


# SIDEBAR - Control Panel only
# Shows file count, page count, chunk count, indexed files, pipeline info
with st.sidebar:

    # Sidebar header with logo icon and title
    st.markdown("""
<div style="display:flex;align-items:center;gap:9px;padding-bottom:12px;border-bottom:0.5px solid rgba(148,163,184,0.1);margin-bottom:4px">
  <div style="width:28px;height:28px;border-radius:7px;background:#534AB7;display:flex;align-items:center;justify-content:center;color:#EEEDFE;font-size:14px;flex-shrink:0">
    <i class="ti ti-brain" aria-hidden="true"></i>
  </div>
  <div>
    <div style="font-size:13px;font-weight:600;color:#e2e8f0">DocuMind AI</div>
    <div style="font-size:10px;color:#475569">Control Panel</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Live index statistics - updates on every rerun
    stats = get_index_stats()

    st.markdown('<div class="cp-label">Index stats</div>', unsafe_allow_html=True)

    # Show files, pages, chunks in a 2-column grid
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
<div class="cp-stat">
  <div class="cp-stat-label">Files</div>
  <div class="cp-stat-value">{stats['files']}</div>
</div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
<div class="cp-stat">
  <div class="cp-stat-label">Pages</div>
  <div class="cp-stat-value">{stats['pages']}</div>
</div>""", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"""
<div class="cp-stat">
  <div class="cp-stat-label">Chunks</div>
  <div class="cp-stat-value">{stats['chunks']}</div>
</div>""", unsafe_allow_html=True)

    with c4:
        status_color = "#1D9E75" if is_indexed() else "#E24B4A"
        status_text = "Ready" if is_indexed() else "Empty"
        st.markdown(f"""
<div class="cp-stat">
  <div class="cp-stat-label">Status</div>
  <div class="cp-stat-value" style="font-size:13px;color:{status_color}">{status_text}</div>
</div>""", unsafe_allow_html=True)

    # Show indexed file chips if any files are indexed
    if stats["files"] > 0:
        st.markdown('<div class="cp-label">Indexed files</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="cp-file">
  <div class="cp-file-icon"><i class="ti ti-file-type-pdf" aria-hidden="true"></i></div>
  <div>
    <div class="cp-file-name">Document indexed</div>
    <div class="cp-file-size">Ready to answer questions</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # Accuracy pipeline - shows which techniques are active
    st.markdown('<div class="cp-label">Accuracy pipeline</div>', unsafe_allow_html=True)
    pipeline_items = [
        "BGE embeddings",
        "BM25 keyword search",
        "Hybrid retrieval",
        "Cross-encoder reranking",
        "Duplicate removal",
        "Strict grounded prompt",
    ]
    for item in pipeline_items:
        st.markdown(f"""
<div class="cp-pip-item">
  <div class="cp-pip-dot"></div>
  {item}
</div>""", unsafe_allow_html=True)

    st.divider()

    # Retrieval slider - controls how many chunks are passed to the LLM
    st.markdown('<div class="cp-label">Retrieval settings</div>', unsafe_allow_html=True)
    k_value = st.slider(
        "Sources to use",
        min_value=3,
        max_value=10,
        value=6,
        help="Number of document chunks retrieved per question. Best: 5 to 7."
    )
    st.caption("Best value: 5 to 7")


# MAIN CONTENT

# Top header bar - app name on the left, dark/light mode toggle on the right
header_left, header_right = st.columns([5, 1])
with header_left:
    st.markdown(
        '<h2 style="margin:6px 0 0;font-size:22px;font-weight:600;">'
        'Welcome to DocuMind AI</h2>',
        unsafe_allow_html=True,
    )
with header_right:
    st.session_state.dark_mode = st.toggle(
        "Dark mode",
        value=st.session_state.dark_mode,
        label_visibility="collapsed",
    )

st.divider()

# App title and subtitle - center top upload zone
st.markdown("""
<div class="upload-zone">
  <h2>Upload your document</h2>
  <p>Upload multiple PDFs and ask anything — get grounded answers with exact page citations</p>
</div>
""", unsafe_allow_html=True)

# PDF file uploader - drag and drop or click to browse
with st.container():
    uploaded_files = st.file_uploader(
        "Drag and drop your PDF here, or click to browse",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    # Action buttons row
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Primary button - triggers the full indexing pipeline
        index_btn = st.button(
            "Index Document",
            type="primary",
            use_container_width=True
        )

    with col2:
        # Reset button - clears ChromaDB and chat history
        reset_btn = st.button(
            "Reset Index",
            use_container_width=True
        )

    with col3:
        # Show how many files are indexed as a simple label
        stats = get_index_stats()
        st.markdown(
            f"<div style='padding:8px 0;text-align:center;color:#475569;font-size:12px;'>"
            f"{stats['files']} file(s) indexed</div>",
            unsafe_allow_html=True
        )

    # Index button handler
    if index_btn:
        if not uploaded_files:
            # No file selected - prompt the user
            st.warning("Please upload at least one PDF before indexing.")
        elif not GROQ_API_KEY:
            # API key missing from .env file
            st.error("GROQ_API_KEY not found in .env file. Please add it and restart.")
        else:
            # Step 1: Save PDFs to disk and run the indexing pipeline
            with st.spinner("Reading pages, creating embeddings, building index..."):
                paths = save_uploaded_files(uploaded_files)
                stats = index_pdfs(paths)

            # Step 2: Generate document-specific suggested questions via Groq
            with st.spinner("Generating smart questions for your document..."):
                questions = generate_suggested_questions(GROQ_API_KEY)
                st.session_state.suggested_questions = questions

            # Clear previous chat so the user starts fresh with this document
            st.session_state.messages = []

            st.success(
                f"Indexed {stats['files']} file(s), {stats['pages']} pages, "
                f"{stats['chunks']} chunks. Ready to answer questions."
            )
            st.rerun()

    # Reset button handler
    if reset_btn:
        # Wipe ChromaDB, clear chat, clear suggested questions
        clear_index()
        st.session_state.messages = []
        st.session_state.suggested_questions = []
        st.success("Index cleared. Upload a new PDF to start fresh.")
        st.rerun()

# SMART SUGGESTED QUESTIONS
# Shown after indexing - questions are specific to the uploaded document
if st.session_state.suggested_questions:
    st.divider()
    st.markdown(
        '<div class="suggest-label">Smart questions for your document</div>',
        unsafe_allow_html=True
    )

    # Display 6 questions in 2 rows of 3 columns
    questions = st.session_state.suggested_questions

    # First row - questions 1, 2, 3
    row1 = st.columns(3)
    for col, q in zip(row1, questions[:3]):
        if col.button(q, use_container_width=True, key=f"sq_{q[:20]}"):
            st.session_state["pending_question"] = q

    # Second row - questions 4, 5, 6
    row2 = st.columns(3)
    for col, q in zip(row2, questions[3:6]):
        if col.button(q, use_container_width=True, key=f"sq_{q[:20]}_b"):
            st.session_state["pending_question"] = q


# CHAT HISTORY
# Renders all previous messages in chatbot bubble style
if st.session_state.messages:
    st.markdown("#### Conversation")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            # User message - right aligned purple bubble
            st.markdown(f"""
<div class="msg-user">
  <div class="bubble">{msg['content']}</div>
</div>""", unsafe_allow_html=True)

        else:
            # AI message - left aligned with avatar and confidence badge
            confidence = msg.get("confidence", "")
            answer = msg.get("content", "").replace("\n", "<br>")
            sources = msg.get("sources", [])

            st.markdown(f"""
<div class="msg-ai">
  <div class="avatar">AI</div>
  <div class="bubble">
    <div class="ai-header">
      <span class="ai-name">DocuMind AI</span>
      <span class="conf-badge">{confidence}</span>
    </div>
    {answer}
  </div>
</div>""", unsafe_allow_html=True)

            # Expandable section showing exact source chunks used
            if sources:
                with st.expander("View exact sources used", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(
                            f"<div style='padding:10px 13px;border-radius:10px;"
                            f"border:0.5px solid rgba(148,163,184,0.12);"
                            f"background:rgba(2,6,23,0.45);margin-bottom:7px;'>"
                            f"<b>Source {i}</b> — Page {src['page']} · {src['section']}<br>"
                            f"<small style='color:#475569;'>{src['file_name']}</small></div>",
                            unsafe_allow_html=True,
                        )
                        # Show up to 1600 characters of the source chunk text
                        st.write(
                            src["text"][:1600] + ("..." if len(src["text"]) > 1600 else "")
                        )


# CHAT INPUT
# Handles both typed questions and clicked suggestion buttons
pending = st.session_state.pop("pending_question", None) \
    if "pending_question" in st.session_state else None

question = st.chat_input(
    "Ask about aim, methodology, dataset, accuracy, architecture, limitations..."
)

# Use clicked suggestion or typed question, whichever is present
question = pending or question

if question:
    # Add user turn to chat history
    st.session_state.messages.append({"role": "user", "content": question})

    # Guard: require a document to be indexed before answering
    if not is_indexed():
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Please upload and index a PDF first using the upload area above.",
            "sources": [],
            "confidence": "—"
        })
        st.rerun()
        st.stop()

    try:
        # Run the full RAG pipeline: retrieve, rerank, generate answer
        with st.spinner("Searching document and generating grounded answer..."):
            result = answer_question(
                question,
                st.session_state.messages,
                k=k_value,
                api_key=GROQ_API_KEY   # loaded from .env, not from the UI
            )

        # Add AI response to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
        })

    except Exception as e:
        # Provide a helpful error message for API key issues vs other errors
        error = str(e)
        if "401" in error or "invalid" in error.lower() or "api key" in error.lower():
            msg = "Groq API key error. Check your .env file — key must start with gsk_."
        else:
            msg = f"Error: {error}"

        st.session_state.messages.append({
            "role": "assistant",
            "content": msg,
            "sources": [],
            "confidence": "Error"
        })

    # Rerun to display the new messages immediately
    st.rerun()


# Footer
st.divider()
st.caption(
    "Built with Streamlit, LangChain, ChromaDB, BGE Embeddings, "
    "BM25, Cross-Encoder Reranking, Groq Llama"
)