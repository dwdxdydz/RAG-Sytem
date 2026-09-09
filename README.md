# 📄 RAG Document Assistant

A local **Retrieval-Augmented Generation (RAG)** application for asking questions over PDF documents. The project combines document processing, semantic embeddings, vector search, evidence inspection, and optional local answer generation in a Streamlit interface.

## Architecture

```text
PDF documents
     ↓
Text extraction + cleaning
     ↓
Overlapping chunks + metadata
     ↓
SentenceTransformer embeddings
     ↓
FAISS vector index
     ↓
User question → query embedding
     ↓
Top-k relevant evidence
     ↓
Grounded answer / evidence inspection
```

## Features

- PDF ingestion with source and page metadata
- Text cleaning and overlapping chunking
- SentenceTransformer embeddings
- Normalized cosine-similarity retrieval with FAISS
- Multi-document retrieval
- Similarity scores and source/page evidence
- Optional local FLAN-T5 answer generation
- Streamlit user interface
- Tests and CI

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The embedding model is downloaded on first use. Answer generation can optionally download the additional FLAN-T5 model.

## Design choices

Retrieval is kept independent from generation. This allows the retrieval layer to be tested and inspected separately, makes evidence visible to the user, and keeps the application usable without a large generative model.

## Portfolio value

This project demonstrates practical skills in **Python, NLP, embeddings, vector databases/search, information retrieval, document processing, AI application design, Streamlit, testing, and system architecture**.

## Future improvements

- Persistent vector storage
- Document upload/delete management
- Reranking for improved retrieval quality
- Retrieval evaluation dataset and metrics
- Conversation memory
- Authentication
- REST API deployment
- More robust citation and hallucination safeguards
