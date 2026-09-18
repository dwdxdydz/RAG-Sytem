# 📚 Local RAG Document Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FAISS](https://img.shields.io/badge/vector_search-FAISS-green.svg)](https://github.com/facebookresearch/faiss)
[![SentenceTransformers](https://img.shields.io/badge/embeddings-MiniLM--L6--v2-orange.svg)](https://www.sbert.net/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![CI](https://github.com/dwdxdydz/RAG-Sytem/actions/workflows/ci.yml/badge.svg)](https://github.com/dwdxdydz/RAG-Sytem/actions)

An end-to-end, privacy-preserving **Retrieval-Augmented Generation (RAG)** pipeline that indexes PDF and plain text documents locally, performs dense vector similarity search with FAISS, and synthesizes source-grounded answers with verifiable citations.

---

## 🏗️ Architecture & Pipeline Flow

```
┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│  Source Files   │ ────> │ Character Segmentation │ ────> │ Dense Embeddings     │
│ (PDF, TXT, MD)  │       │ (800 char, 120 overlap)│       │ (384-dim MiniLM-L6)  │
└─────────────────┘       └────────────────────────┘       └──────────────────────┘
                                                                       │
                                                                       ▼
┌─────────────────┐       ┌────────────────────────┐       ┌──────────────────────┐
│ Verified Answer │ <──── │ Top-K Evidence Ranking │ <──── │ FAISS Vector Index   │
│ (FLAN-T5 Small) │       │ (Cosine Similarity)    │       │ (IndexFlatIP)        │
└─────────────────┘       └────────────────────────┘       └──────────────────────┘
```

---

## ✨ Key Features

- **Multi-Format Ingestion**: Ingests multi-page PDFs, markdown, and text files with page-aware metadata tracking.
- **Normalized Vector Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense semantic vectors.
- **Blazing Fast Vector Indexing**: In-memory `faiss.IndexFlatIP` performing exact inner-product (cosine similarity) search in sub-millisecond latencies.
- **Evidence-First Inspection**: Returns ranked source excerpts with document name, page number, and similarity score to prevent ungrounded hallucinations.
- **Local Generation**: Optional on-device generation with Google's `flan-t5-small` without sending data to external APIs.
- **Interactive Web UI**: Streamlit web dashboard with real-time indexing status, similarity threshold sliders, and expandable evidence cards.
- **CLI & Automated Testing**: Standalone CLI test mode and a comprehensive `pytest` test suite with GitHub Actions CI.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/dwdxdydz/RAG-Sytem.git
cd RAG-Sytem
pip install -r requirements.txt
```

### 2. Run the Interactive Web UI

```bash
streamlit run app.py
```

### 3. Run via CLI

```bash
# Run the built-in demo knowledge base query
python rag_system.py --demo --query "What embedding model is used?" -k 3
```

---

## 🧪 Testing

```bash
pytest -v
```

---

## 📊 Technical Specifications

| Component | Technology | Specification |
| :--- | :--- | :--- |
| **Embedding Model** | `all-MiniLM-L6-v2` | 384 dimensions, normalized unit vectors |
| **Vector Index** | `faiss.IndexFlatIP` | Exact Inner Product / Cosine Similarity |
| **Chunking Strategy** | Windowing | 800 characters, 120 character overlap |
| **Generation Model** | `google/flan-t5-small` | 80M parameters (Local Text2Text) |
| **Web Interface** | Streamlit | Responsive layout with real-time reactive filters |

---

## 📄 License
MIT License. Free for educational and commercial use.
