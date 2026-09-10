# 📄 RAG Document Assistant

## What is this project?

This is a small AI application that lets you **ask questions about PDF documents**.

Instead of reading a long document from beginning to end, you can upload it, ask a question, and the application finds the most relevant parts of the document before generating an answer.

For example:

```text
PDF: Company policy document

Question: How many days of leave can an employee take?

        ↓

Find the relevant part of the PDF
        ↓

Use that information to answer
```

## How does it work?

```text
PDF document
     ↓
Extract the text
     ↓
Break text into smaller chunks
     ↓
Convert chunks into numbers (embeddings)
     ↓
Store them in a searchable index
     ↓
User asks a question
     ↓
Find the most similar chunks
     ↓
Show evidence / generate an answer
```

This approach is called **RAG — Retrieval-Augmented Generation**.

The important idea is simple: **find useful information first, then use that information to answer the question.**

## Main features

- Upload and read PDF documents
- Keep track of the page where information came from
- Split large documents into smaller overlapping chunks
- Search for relevant information based on meaning, not only exact words
- Search across multiple documents
- Show similarity scores and source/page information
- Optionally generate answers using a local FLAN-T5 model
- Streamlit web interface
- Automated tests and CI

## Example

Imagine a 100-page PDF contains the sentence you need on page 73.

Instead of sending the entire document to an AI model, the application first searches the document and finds the most relevant sections.

That makes the system more focused and lets the user inspect the evidence used for the answer.

## Why separate search from answer generation?

The project keeps **retrieval** and **generation** as separate steps.

This is useful because you can test:

```text
Did we find the correct information?
        ↓
Yes → Generate an answer from it
No  → Improve the search
```

This also makes it easier to inspect what information the application actually found.

## Run the application

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The embedding model is downloaded the first time it is needed. The optional answer-generation model requires additional model files.

## Project structure

```text
app.py / application files → User interface and application flow
PDF processing             → Extracts and cleans document text
Chunking                    → Splits text into searchable pieces
Embeddings                  → Converts text into numerical representations
FAISS                       → Searches for similar pieces of text
Tests/                      → Checks that important parts work correctly
```

## Technical terms explained

**RAG (Retrieval-Augmented Generation)** — A method where an application first retrieves relevant information and then uses it to help generate an answer.

**Retrieval** — Finding the pieces of stored information that are most relevant to a user's question.

**Embedding** — A list of numbers that represents the meaning of text. Similar pieces of text tend to have similar numerical representations.

**SentenceTransformer** — A machine-learning model used here to convert text into embeddings.

**FAISS** — A library designed to search large collections of vectors quickly. Here it is used to find document chunks whose embeddings are closest to the question embedding.

**Vector** — In this project, simply a list of numbers representing text. The numbers allow mathematical comparison of meaning.

**Cosine similarity** — A mathematical measure used to estimate how similar two vectors are. A higher score means the texts are more similar in meaning.

**Chunk** — A smaller piece of a document. Large documents are divided into chunks so the system can search them more effectively.

**Metadata** — Extra information stored alongside a chunk, such as the document name and page number.

**FLAN-T5** — A text-generation model that can turn retrieved information into a natural-language answer.

**Streamlit** — A Python framework for creating a simple web interface for data and AI applications.

**CI (Continuous Integration)** — Automatic checks, such as tests, that run when code changes are pushed to GitHub.

## What does this project demonstrate?

The project demonstrates a complete AI information-search workflow:

**PDF processing → embeddings → vector search → evidence retrieval → answer generation**

It demonstrates practical **Python, NLP, machine learning, information retrieval, embeddings, vector search, AI application development, Streamlit and testing** skills.

## Current limitations

The included generation model is intentionally optional and local. The project is primarily focused on understanding and demonstrating the RAG workflow rather than providing production-scale AI infrastructure.

## Future improvements

- Persistent vector storage
- Better document management
- Reranking for improved search results
- Retrieval evaluation metrics
- Conversation memory
- Authentication
- REST API deployment
- Stronger citations and hallucination protection
