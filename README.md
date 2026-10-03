# Local RAG Document Assistant

## What is this?

This is a local **RAG (Retrieval-Augmented Generation)** application.

You give it documents such as PDFs or text files. You can then ask questions about those documents.

Instead of asking an AI model to answer from memory, the system first searches the documents for relevant information.

## Simple flow

```
PDF / TXT / MD files
        ↓
Split into smaller text chunks
        ↓
Create embeddings
        ↓
Store vectors in FAISS
        ↓
User asks a question
        ↓
Find relevant chunks
        ↓
Generate an answer
        ↓
Show supporting sources
```

## Why use RAG?

Suppose you have a 100-page document.

Instead of sending the entire document every time:

1. The document is processed once.
2. Its sections are converted into vectors.
3. A question is converted into a vector.
4. The closest sections are found.
5. Those sections are given to the language model.
6. The model generates an answer using that information.

## Main technologies

- Python
- Sentence Transformers
- FAISS
- FLAN-T5
- Streamlit
- Pytest

## Main files

- `app.py` — Streamlit user interface
- `rag_system.py` — main RAG pipeline
- `RAG System.py` — older/simple implementation
- `tests/` — automated tests
- `requirements.txt` — dependencies

## Run it

Install:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

Run tests:

```bash
pytest -q
```

## Important terms

**RAG:** retrieve relevant information first, then use an AI model to generate an answer.

**Embedding:** a numerical representation of text that helps compare meaning.

**Vector:** a list of numbers representing information mathematically.

**FAISS:** a library used to search large collections of vectors efficiently.

**Similarity search:** finding text that is mathematically close to the user's question.

**Chunk:** a smaller piece of a larger document.

**LLM:** a language model that generates text.

## What this project demonstrates

- AI application development
- RAG architecture
- Semantic search
- Embeddings
- Vector databases/search
- Document processing
- Python
- Streamlit
- Testing
