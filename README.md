<div align="center">

<img src="https://img.shields.io/badge/DocuMind%20AI-RAG%20Document%20Q%26A-6366f1?style=for-the-badge&logo=openai&logoColor=white" alt="DocuMind AI"/>

# 🧠 DocuMind AI

### High-Accuracy RAG Document Q&A System

*Ask anything from your PDF — get grounded answers with exact page citations*

<br/>

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-FF6B35?style=flat-square)](https://trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-Llama%203-F55036?style=flat-square)](https://groq.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

<br/>

![DocuMind AI Demo](https://img.shields.io/badge/Status-Production%20Ready-22c55e?style=for-the-badge)

</div>

---

## ✨ What is DocuMind AI?

**DocuMind AI** is a production-ready Retrieval-Augmented Generation (RAG) system that lets you upload any PDF — thesis, research paper, coursework, or report — and ask natural language questions. It returns grounded answers with **exact page citations**, so you always know where the information came from.

Built for students, researchers, and professionals who need accurate, fast document intelligence.

---

## 🚀 Key Features

| Feature | Description |
|---|---|
| 📄 **Multi-PDF Upload** | Upload and index multiple PDFs at once |
| 🔍 **Hybrid Retrieval** | Dense vector search + BM25 keyword search combined |
| 🎯 **BGE Embeddings** | High-quality `BAAI/bge-base-en-v1.5` embeddings |
| 🏆 **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` reranker for precision |
| 📍 **Page Citations** | Every answer cites exact page numbers |
| 🚫 **No Hallucination** | Strict grounded prompt — never guesses |
| 🗑️ **Duplicate Removal** | Deduplicates retrieved sources automatically |
| 💬 **Chat Interface** | Clean conversational UI with confidence scoring |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     DocuMind AI                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  PDF Upload ──► Page Chunking ──► BGE Embeddings        │
│                                        │                 │
│                                   ChromaDB               │
│                                   + BM25 Index           │
│                                        │                 │
│  User Question ──► Hybrid Retrieval ──► Top-K Chunks    │
│                                        │                 │
│                              Cross-Encoder Reranking     │
│                                        │                 │
│                              Groq Llama 3 (LLM)         │
│                                        │                 │
│                         Grounded Answer + Page Citations │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

```
Frontend        →  Streamlit (dark theme, chatbot UI)
LLM             →  Groq API — Llama 3.3 70B Versatile (free & fast)
Embeddings      →  HuggingFace — BAAI/bge-base-en-v1.5 (local)
Vector Store    →  ChromaDB (persistent local storage)
Keyword Search  →  BM25 (rank-bm25)
Reranker        →  cross-encoder/ms-marco-MiniLM-L-6-v2
RAG Framework   →  LangChain 0.3+
PDF Parsing     →  PyPDF
```

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/mohamedahshaan/DocuMind-AI.git
cd DocuMind-AI
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and add your Groq API key:

```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
EMBED_MODEL=BAAI/bge-base-en-v1.5
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

> 🔑 Get your **free** Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 💡 How to Use

```
1. Upload your PDF  ──►  drag & drop or click to browse
2. Click "Index Document"  ──►  wait ~30 seconds
3. Ask any question  ──►  get cited answers instantly
```

**Example questions to try:**
- `What is the aim of this project?`
- `What methodology was used?`
- `What dataset was used for training?`
- `Explain the system architecture.`
- `What are the limitations of this project?`
- `Summarize the entire project.`

---

## 🔧 Troubleshooting

**Reset the database (if indexing fails):**
```bash
rm -rf ~/.rag_pro_max_runtime
streamlit run app.py
```

**Suppress torchvision warnings:**
```bash
streamlit run app.py --server.fileWatcherType none
```

**API key error:**
Make sure `.env` file exists and `GROQ_API_KEY` starts with `gsk_`.

---

## 📁 Project Structure

```
DocuMind-AI/
├── app.py              ← Streamlit UI (chatbot interface)
├── rag_engine.py       ← RAG pipeline (indexing + retrieval)
├── requirements.txt    ← Python dependencies
├── .env.example        ← Environment variable template
└── README.md           ← This file
```

---

## 🎯 CV / Resume Description

> **DocuMind AI — RAG Document Q&A System**
> *Solo Project | LangChain · ChromaDB · BGE Embeddings · BM25 · Cross-Encoder Reranking · Groq Llama 3 · Streamlit*
>
> Built a production-ready Retrieval-Augmented Generation pipeline with hybrid dense+sparse retrieval and Cross-Encoder reranking, achieving high-accuracy document Q&A with page-level source citations. Deployed as an interactive Streamlit web application supporting multi-PDF indexing and conversational queries.

---

## 👤 Author

**Mohamed Ahshaan**
AI & Data Science Undergraduate | NLP · LLM · Computer Vision

[![GitHub](https://img.shields.io/badge/GitHub-mohamedahshaan-181717?style=flat-square&logo=github)](https://github.com/mohamedahshaan)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-mohamedahshaan-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/mohamedahshaan)

---

<div align="center">

*Built with ❤️ using LangChain · ChromaDB · BGE Embeddings · BM25 · Cross-Encoder Reranking · Groq Llama · Streamlit*

</div>
