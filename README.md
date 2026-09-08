# RAG Document Assistant

A complete local Retrieval-Augmented Generation (RAG) application for asking questions over PDF documents.

## Pipeline

```text
PDFs → page-aware extraction → overlapping chunks → embeddings → FAISS
                                      ↓
Question → query embedding → top-k evidence → grounded answer
```

## Features

- PDF ingestion with source and page metadata
- Overlapping text chunking
- SentenceTransformer embeddings with cosine-similarity search
- FAISS vector index
- Multi-document retrieval
- Similarity scores and evidence inspection
- Optional local FLAN-T5 generation
- Streamlit UI

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the embedding model. Enable answer generation in the UI only when you want to download the additional FLAN-T5 model.

## Design choices

The application keeps retrieval independent from generation. This makes it possible to evaluate retrieval quality, inspect the exact evidence returned, and run the project without a large generative model.

## Future extensions

Persistent vector storage, document deletion, reranking, retrieval evaluation, conversation memory, authentication, and an API layer can be added without changing the core retrieval contract.
