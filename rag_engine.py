"""High-accuracy RAG engine for PDF thesis / coursework Q&A.

Upgrades included:
- Multi-PDF support
- Page-aware chunking with section detection
- Hybrid retrieval: dense vector search + BM25 keyword search
- Cross-encoder reranking
- Duplicate source removal
- Strict grounded answer prompt
- API-key error handling
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()

ROOT = Path(__file__).parent
# Store runtime databases OUTSIDE the project folder.
# This avoids macOS/iCloud/ZIP permission issues such as:
# chromadb.errors.InternalError: attempt to write a readonly database
RUNTIME_ROOT = Path(os.getenv("RAG_PRO_MAX_DATA_DIR", str(Path.home() / ".rag_pro_max_runtime")))
DATA_DIR = RUNTIME_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
CHROMA_DIR = DATA_DIR / "chroma_db"
INDEX_DIR = DATA_DIR / "index"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"

DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
FALLBACK_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Thesis PDFs usually need bigger chunks than normal notes.
CHUNK_SIZE = 1300
CHUNK_OVERLAP = 220


@dataclass
class SourceChunk:
    chunk_id: str
    text: str
    file_name: str
    page: int
    section: str = "Unknown section"
    start_char: int = 0
    end_char: int = 0


def _make_writable(path: Path) -> None:
    """Best-effort chmod for old Chroma SQLite files before deletion."""
    try:
        if path.exists():
            for item in path.rglob("*"):
                try:
                    item.chmod(0o700 if item.is_dir() else 0o600)
                except Exception:
                    pass
            path.chmod(0o700)
    except Exception:
        pass


def _safe_rmtree(path: Path) -> None:
    if not path.exists():
        return
    _make_writable(path)
    shutil.rmtree(path, ignore_errors=True)


def _ensure_dirs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _make_writable(DATA_DIR)


def clear_index() -> None:
    _safe_rmtree(DATA_DIR)
    _ensure_dirs()


def is_indexed() -> bool:
    return CHUNKS_FILE.exists() and CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())


def get_index_stats() -> Dict[str, int]:
    if not CHUNKS_FILE.exists():
        return {"files": 0, "pages": 0, "chunks": 0}
    chunks = _load_chunks()
    files = len({c.file_name for c in chunks})
    pages = len({(c.file_name, c.page) for c in chunks})
    return {"files": files, "pages": pages, "chunks": len(chunks)}


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\u000c", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _detect_section(text: str, current: str) -> str:
    patterns = [
        r"^(Chapter\s+\d+\s*[:\-].{0,90})$",
        r"^(\d+(?:\.\d+)*\s+[A-Z][A-Za-z0-9 ,\-/&()]{3,90})$",
        r"^(Research Aim|Research Objectives|Project Scope|In-Scope|Out of Scope|Testing|Evaluation|References)$",
    ]
    for raw in text.splitlines()[:8]:
        line = raw.strip()
        if len(line) > 120:
            continue
        for pat in patterns:
            if re.match(pat, line):
                return line
    return current


def _window_chunks(page_text: str, file_name: str, page: int, section: str) -> List[SourceChunk]:
    chunks: List[SourceChunk] = []
    if not page_text:
        return chunks

    start = 0
    n = len(page_text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        if end < n:
            # Prefer splitting at paragraph/sentence boundaries.
            split_at = max(page_text.rfind("\n\n", start, end), page_text.rfind(". ", start, end))
            if split_at > start + 500:
                end = split_at + 1
        text = page_text[start:end].strip()
        if len(text) > 120:
            digest = hashlib.md5(f"{file_name}|{page}|{start}|{text[:80]}".encode()).hexdigest()[:12]
            chunks.append(
                SourceChunk(
                    chunk_id=digest,
                    text=text,
                    file_name=file_name,
                    page=page,
                    section=section,
                    start_char=start,
                    end_char=end,
                )
            )
        if end >= n:
            break
        start = max(0, end - CHUNK_OVERLAP)
    return chunks


def _load_pdf_pages(pdf_path: str) -> List[Document]:
    loader = PyPDFLoader(pdf_path)
    return loader.load()


def _get_embeddings() -> HuggingFaceEmbeddings:
    # BGE gives better retrieval. Fallback keeps the app usable on low-resource machines.
    model_name = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    encode_kwargs = {"normalize_embeddings": True}
    try:
        return HuggingFaceEmbeddings(model_name=model_name, encode_kwargs=encode_kwargs)
    except Exception:
        return HuggingFaceEmbeddings(model_name=FALLBACK_EMBED_MODEL, encode_kwargs=encode_kwargs)


def index_pdfs(pdf_paths: List[str]) -> Dict[str, int]:
    """Index one or many PDF files and return stats."""
    _ensure_dirs()
    _safe_rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    if CHUNKS_FILE.exists():
        CHUNKS_FILE.unlink()

    all_chunks: List[SourceChunk] = []
    total_pages = 0

    for pdf_path in pdf_paths:
        pages = _load_pdf_pages(pdf_path)
        total_pages += len(pages)
        current_section = "Document start"
        file_name = Path(pdf_path).name
        for page_doc in pages:
            page_number = int(page_doc.metadata.get("page", 0)) + 1
            text = _clean_text(page_doc.page_content)
            current_section = _detect_section(text, current_section)
            all_chunks.extend(_window_chunks(text, file_name, page_number, current_section))

    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    docs = [
        Document(
            page_content=c.text,
            metadata={
                "chunk_id": c.chunk_id,
                "file_name": c.file_name,
                "page": c.page,
                "section": c.section,
            },
        )
        for c in all_chunks
    ]
    embeddings = _get_embeddings()
    Chroma.from_documents(docs, embeddings, persist_directory=str(CHROMA_DIR))

    return {"files": len(pdf_paths), "pages": total_pages, "chunks": len(all_chunks)}


def _load_chunks() -> List[SourceChunk]:
    if not CHUNKS_FILE.exists():
        return []
    chunks: List[SourceChunk] = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(SourceChunk(**json.loads(line)))
    return chunks


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _query_expansion(question: str) -> str:
    q = question.lower()
    expansions = []
    mapping = {
        "aim": "research aim objective purpose goal",
        "methodology": "Chapter 3 methodology research methodology development methodology waterfall OOAD positivist deductive quantitative",
        "dataset": "data selection Kaggle Brain Tumor MRI Dataset classes glioma meningioma pituitary no tumor",
        "accuracy": "accuracy precision recall f1-score confusion matrix benchmarking evaluation results testing",
        "architecture": "system architecture design component diagram process flow backend frontend model chatbot",
        "chatbot": "LLM conversational AI chatbot assistant explanation natural language responses",
        "gradcam": "Grad-CAM heatmap explainable AI visualization",
        "grad-cam": "Grad-CAM heatmap explainable AI visualization",
        "tumor types": "glioma meningioma pituitary adenoma no tumor four classes",
        "types": "glioma meningioma pituitary adenoma no tumor four classes",
        "scope": "in scope out of scope project scope limitations",
        "functional": "functional requirements requirement list priority",
        "non-functional": "non functional requirements accuracy usability performance security",
        "future": "future enhancements limitations development",
    }
    for key, value in mapping.items():
        if key in q:
            expansions.append(value)
    return question + " " + " ".join(expansions)


def _dense_results(question: str, fetch_k: int = 30) -> List[Tuple[SourceChunk, float]]:
    embeddings = _get_embeddings()
    db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings)
    docs_scores = db.similarity_search_with_relevance_scores(_query_expansion(question), k=fetch_k)
    id_to_chunk = {c.chunk_id: c for c in _load_chunks()}
    results: List[Tuple[SourceChunk, float]] = []
    for doc, score in docs_scores:
        cid = doc.metadata.get("chunk_id")
        if cid in id_to_chunk:
            results.append((id_to_chunk[cid], float(score)))
    return results


def _bm25_results(question: str, fetch_k: int = 30) -> List[Tuple[SourceChunk, float]]:
    chunks = _load_chunks()
    if not chunks:
        return []
    corpus = [_tokenize(c.text + " " + c.section) for c in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(_query_expansion(question)))
    if np.max(scores) > 0:
        scores = scores / np.max(scores)
    top_idx = np.argsort(scores)[::-1][:fetch_k]
    return [(chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]


def _heuristic_boost(question: str, chunk: SourceChunk) -> float:
    q = question.lower()
    hay = (chunk.text + " " + chunk.section).lower()
    boost = 0.0
    important_terms = ["methodology", "dataset", "accuracy", "research aim", "architecture", "chatbot", "grad-cam", "functional requirements", "non-functional"]
    for term in important_terms:
        if term in q and term in hay:
            boost += 0.18
    # If user asks a page/chapter specific query.
    m = re.search(r"chapter\s*(\d+)", q)
    if m and f"chapter {m.group(1)}" in hay:
        boost += 0.25
    return boost


def retrieve(question: str, final_k: int = 6) -> List[SourceChunk]:
    if not is_indexed():
        return []

    candidates: Dict[str, Tuple[SourceChunk, float]] = {}
    for chunk, score in _dense_results(question, fetch_k=35):
        candidates[chunk.chunk_id] = (chunk, candidates.get(chunk.chunk_id, (chunk, 0))[1] + score * 0.65)
    for chunk, score in _bm25_results(question, fetch_k=35):
        candidates[chunk.chunk_id] = (chunk, candidates.get(chunk.chunk_id, (chunk, 0))[1] + score * 0.45)

    scored = []
    for chunk, score in candidates.values():
        scored.append((chunk, score + _heuristic_boost(question, chunk)))
    scored.sort(key=lambda x: x[1], reverse=True)
    pre = [c for c, _ in scored[:25]]

    # Cross-encoder rerank for true semantic relevance.
    try:
        reranker = CrossEncoder(os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL))
        pairs = [(question, f"{c.section}\nPage {c.page}\n{c.text}") for c in pre]
        scores = reranker.predict(pairs)
        ranked = [c for c, _ in sorted(zip(pre, scores), key=lambda x: float(x[1]), reverse=True)]
    except Exception:
        ranked = pre

    # Deduplicate near-identical text and avoid too many same-page chunks unless needed.
    selected: List[SourceChunk] = []
    seen_hashes = set()
    page_count: Dict[Tuple[str, int], int] = {}
    for c in ranked:
        norm = re.sub(r"\W+", "", c.text.lower())[:700]
        h = hashlib.md5(norm.encode()).hexdigest()
        if h in seen_hashes:
            continue
        page_key = (c.file_name, c.page)
        if page_count.get(page_key, 0) >= 2:
            continue
        selected.append(c)
        seen_hashes.add(h)
        page_count[page_key] = page_count.get(page_key, 0) + 1
        if len(selected) >= final_k:
            break
    return selected


def _format_context(chunks: List[SourceChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[SOURCE {i}] File: {c.file_name} | Page: {c.page} | Section: {c.section}\n{c.text}"
        )
    return "\n\n---\n\n".join(parts)


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a strict research document assistant. Answer ONLY using the provided PDF context.\n\nRules:\n1. Do not use outside knowledge.\n2. If the context does not contain the answer, say exactly: \"The document does not provide this information.\"\n3. Never guess or invent numbers, models, results, datasets, or names.\n4. Give page citations like (p. 7) after important claims.\n5. For direct questions, answer first in 2-5 clear bullet points.\n6. If relevant context is conflicting, mention the conflict clearly.\n7. Keep the answer concise but complete.\n""",
        ),
        (
            "human",
            "Question: {question}\n\nPDF Context:\n{context}\n\nAnswer with page citations:",
        ),
    ]
)


def _get_llm(api_key: Optional[str] = None) -> ChatGroq:
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key or not key.strip() or not key.strip().startswith("gsk_"):
        raise ValueError("Missing or invalid Groq API key. Add a valid key in the sidebar or .env file.")
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL),
        api_key=key.strip(),
        temperature=0.0,
        max_tokens=900,
    )


def answer_question(question: str, chat_history: Optional[List[Dict[str, str]]] = None, k: int = 6, api_key: Optional[str] = None) -> Dict:
    chunks = retrieve(question, final_k=k)
    if not chunks:
        return {
            "answer": "Please upload and index a PDF first.",
            "sources": [],
            "confidence": "Low",
        }

    context = _format_context(chunks)
    chain = PROMPT | _get_llm(api_key)
    response = chain.invoke({"question": question, "context": context})
    answer = response.content.strip()

    # Confidence is a simple retrieval-quality signal for UI display.
    q_terms = set(_tokenize(question))
    covered = sum(1 for c in chunks for t in q_terms if t in _tokenize(c.text + " " + c.section))
    confidence = "High" if covered >= max(2, len(q_terms) // 2) else "Medium"

    return {"answer": answer, "sources": [asdict(c) for c in chunks], "confidence": confidence}


# Convenience for Streamlit upload handling.
def save_uploaded_files(uploaded_files: Iterable) -> List[str]:
    _ensure_dirs()
    paths = []
    for uploaded_file in uploaded_files:
        path = DOCS_DIR / uploaded_file.name
        with path.open("wb") as f:
            f.write(uploaded_file.getbuffer())
        paths.append(str(path))
    return paths