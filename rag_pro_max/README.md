# DocuMind AI — RAG Pro Max

High-accuracy PDF Q&A system for thesis, research papers, coursework and reports.

## Features
- Multi-PDF upload
- BGE embeddings
- Hybrid retrieval: vector search + BM25
- Cross-encoder reranking
- Duplicate source removal
- Strict no-guessing prompt
- Page citations
- Modern Streamlit UI
- Runtime database stored in `~/.rag_pro_max_runtime` to avoid ChromaDB readonly errors

## Run on macOS

```bash
cd ~/Desktop/rag_pro_max
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Paste your Groq API key in the sidebar. It must start with `gsk_`.

## Fix / Reset database

```bash
rm -rf ~/.rag_pro_max_runtime
streamlit run app.py
```

## Good test questions
- What is the aim of this project?
- What methodology was used?
- What dataset was used?
- Explain the system architecture.
- What are the limitations?
