# Retrieval-Augmented Generation (RAG) System

Prototype RAG pipeline for turning PDF documents into searchable vector embeddings and using retrieved context to support language-model responses.

## Intended Pipeline

```text
PDF → text extraction → cleaning/chunking → embeddings → FAISS index
                                              ↓
Query → query embedding → similarity search → relevant context → LLM
```

## Tech Stack

**Python · Sentence Transformers · FAISS · PyPDF2 · Hugging Face Transformers · NumPy**

## Current Status

The repository began as an experimental prototype. The original implementation mixed model loading, indexing, extraction, and response generation in one script and contained incomplete preprocessing/retrieval logic. The project documentation now makes the prototype status explicit rather than presenting the unfinished path as production-ready.

## Resume Description

**RAG Document Q&A Prototype | Python, FAISS, Sentence Transformers, Transformers**

Explored a retrieval-augmented generation pipeline that extracts text from PDFs, chunks documents, generates dense embeddings, performs vector similarity search with FAISS, and supplies retrieved context to a language model for grounded question answering.
