# 📄 RAG Document Assistant

## What is this project?

This is a small AI application that lets you **ask questions about PDF documents**.

Instead of reading a long document from beginning to end, you can give the application a PDF, ask a question, and let the application find the most relevant parts of the document.

It can then show that evidence and optionally use a local AI model to generate an answer from the retrieved information.

## A simple example

Imagine you have a 100-page company policy document and want to know:

> How many days of leave can an employee take?

Instead of giving the entire document to an AI model, the application searches the document first.

```text
100-page PDF
     ↓
Extract the text
     ↓
Break text into smaller pieces
     ↓
Create numerical representations of the text
     ↓
Search for pieces related to the question
     ↓
Show the relevant evidence
     ↓
Optionally generate an answer
```

This approach is called **RAG — Retrieval-Augmented Generation**.

The main idea is simple:

**Find useful information first → then use that information to answer the question.**

## How does it work?

```text
PDF documents
      ↓
Text extraction + cleaning
      ↓
Text chunks + page information
      ↓
Embeddings
      ↓
FAISS vector index
      ↓
User question
      ↓
Question embedding
      ↓
Find similar document chunks
      ↓
Relevant evidence
      ↓
Answer / evidence inspection
```

The application keeps **retrieval** separate from **answer generation**. This is useful because we can inspect whether the correct information was found before asking a model to produce an answer.

## Main features

- Upload and process PDF documents.
- Keep the source document and page number with the extracted text.
- Split large documents into smaller overlapping chunks.
- Search based on meaning rather than only exact words.
- Search across multiple documents.
- Display similarity scores and source/page information.
- Use SentenceTransformer to create text embeddings.
- Use FAISS for fast similarity search.
- Optionally generate answers with a local FLAN-T5 model.
- Provide a Streamlit web interface.
- Include automated tests and CI.

## Why use retrieval before generation?

AI models can produce an answer even when they do not have the correct information. This can lead to incorrect or invented answers.

This project tries to reduce that problem by first finding relevant information from the user's documents.

```text
Question
   ↓
Search document
   ↓
Did we find useful evidence?
   ↙              ↘
 Yes               No
  ↓                 ↓
Use evidence     Improve search / show no evidence
  ↓
Generate answer
```

The retrieval results are also visible, which makes it easier for a user to inspect where the information came from.

## Run the application

Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```

The embedding model is downloaded the first time it is needed. The optional answer-generation model requires additional model files.

## Project structure

```text
app.py / application files → User interface and application flow
PDF processing             → Extracts and cleans document text
Chunking                   → Splits documents into smaller pieces
Embeddings                 → Converts text into numerical representations
FAISS                      → Searches for similar text representations
Tests/                     → Checks important application behaviour
```

## Main technologies

- **Python** — application and data-processing logic
- **PyPDF2** — extracts text from PDF files
- **SentenceTransformers** — converts text into embeddings
- **FAISS** — searches embeddings for similar content
- **FLAN-T5** — optional local text-generation model
- **Streamlit** — provides the web interface
- **Pytest** — automated testing
- **GitHub Actions** — automated CI checks

## Technical terms explained

**AI (Artificial Intelligence)** — Software designed to perform tasks that normally require some form of human-like reasoning or pattern recognition.

**RAG (Retrieval-Augmented Generation)** — A method where an application first retrieves relevant information and then gives that information to a generation model to help produce an answer.

**Retrieval** — Finding the stored pieces of information that are most relevant to a question.

**Generation** — Producing new text, such as an answer, from information and instructions given to a model.

**NLP (Natural Language Processing)** — The area of computing focused on working with human language.

**Embedding** — A list of numbers used to represent the meaning or characteristics of text. Texts with similar meaning can have similar embeddings.

**Vector** — A list of numbers. In this project, vectors are used to represent pieces of text so they can be compared mathematically.

**Vector search** — Searching through vectors to find the ones most similar to another vector, such as a user's question.

**FAISS (Facebook AI Similarity Search)** — A library designed to search large collections of vectors efficiently. Here it finds document chunks that are similar to the question.

**Cosine similarity** — A mathematical measure of similarity between two vectors. A higher similarity generally means the text representations are more alike.

**Chunk** — A smaller piece of a larger document. Documents are split into chunks because searching many small pieces is easier than treating a very large document as one piece.

**Overlapping chunks** — Chunks that share some text with the previous or next chunk. This helps preserve context when an important sentence crosses a chunk boundary.

**Metadata** — Extra information stored alongside the main data. Here, metadata can include the document name and page number.

**SentenceTransformer** — A machine-learning model used to convert sentences or passages into useful numerical embeddings.

**FLAN-T5** — A text-generation model that can be used to produce natural-language responses from provided information.

**Streamlit** — A Python framework for building interactive web applications without requiring a separate frontend framework.

**Hallucination** — When an AI model produces information that sounds believable but is unsupported or incorrect.

**CI (Continuous Integration)** — Automatically running checks such as tests when code changes are pushed. This helps catch problems early.

**Pytest** — A Python testing framework used to automatically check whether code behaves as expected.

## What does this project demonstrate?

The project shows a complete document-question-answering workflow:

**PDF → text → chunks → embeddings → vector search → evidence → answer**

It demonstrates practical **Python, NLP, machine learning, information retrieval, embeddings, vector search, AI application development, Streamlit and testing** skills.

## Current limitations

This is primarily an educational and portfolio project. The local generation model is optional, and the included retrieval system is not designed as a production-scale document platform.

## Future improvements

- Persistent vector storage.
- Better document upload and deletion management.
- Reranking for improved retrieval quality.
- Retrieval evaluation datasets and metrics.
- Conversation memory.
- Authentication.
- REST API deployment.
- Stronger citations and hallucination protection.
