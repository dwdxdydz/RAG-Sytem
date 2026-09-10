# 📄 RAG Document Assistant

## What is this project?

This is an AI application that lets you **ask questions about PDF documents**.

Normally, if you have a 100-page document and want to find one specific answer, you have to search through the document yourself.

This application does the searching for you. You upload a PDF, ask a question, and the application finds the most relevant parts of the document before producing an answer.

Think of it as a **chat assistant for your documents**.

## Example

You upload:

```text
Company Annual Report.pdf
```

Then ask:

```text
What was the company's revenue last year?
```

The application searches the document for the most relevant sections and uses those sections to help answer the question.

## How does it work?

```text
PDF document
     ↓
Read the text
     ↓
Break text into small sections
     ↓
Convert sections into numbers that represent their meaning
     ↓
Store those representations for searching
     ↓
User asks a question
     ↓
Find the most relevant sections
     ↓
Show the relevant evidence
     ↓
Generate an answer (optional)
```

## What does RAG mean?

**RAG** stands for **Retrieval-Augmented Generation**.

The idea is simple:

- **Retrieval** → find the useful information in your documents.
- **Augmented** → give that information to the AI as additional context.
- **Generation** → use the context to generate an answer.

Instead of asking an AI to answer from memory, the application first looks inside your documents.

## Main features

- Upload and read PDF documents.
- Break large documents into smaller searchable sections.
- Search by meaning, not only exact words.
- Search across multiple documents.
- See which document and page contained the relevant information.
- See similarity scores for search results.
- Optionally generate an answer using a local AI model.
- Use the application through a simple Streamlit interface.
- Run automated tests.

## Why are embeddings and FAISS used?

The application converts each piece of text into a list of numbers called an **embedding**. These numbers represent the meaning of the text.

This makes it possible to find text that has a similar meaning even when the exact words are different.

**FAISS** is used to quickly search through these numerical representations.

You can think of it like this:

```text
Question:
"How much money did the company make?"

        ↓

Search by meaning

        ↓

Find text such as:
"The company generated revenue of..."
```

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

The required embedding model is downloaded when it is first used. The optional answer-generation model requires an additional download.

## Project structure

```text
app.py                  → User interface
rag.py / core modules   → Document processing and search
requirements.txt        → Python packages
tests/                  → Automated tests
.github/workflows/      → Automatic testing
```

## Main technologies

- **Python** — application logic
- **PyPDF2** — reads PDF files
- **SentenceTransformers** — creates meaning-based text representations
- **FAISS** — quickly searches those representations
- **Streamlit** — creates the web interface
- **FLAN-T5** — optional local AI answer generation

## Why this project is useful

This project shows how an AI application can work with real documents instead of only receiving a question and producing a generic answer.

The important idea is:

**Document → search relevant information → provide evidence → generate answer**

It demonstrates practical Python, NLP, AI, semantic search, document processing and application-development skills.

## Current limitation

The project is mainly an educational and portfolio application. It is not designed to compete with large production document-AI systems.

## Future improvements

- Save document indexes permanently.
- Add document management.
- Improve search ranking.
- Measure search accuracy with a test dataset.
- Remember previous questions in a conversation.
- Add stronger citation and hallucination protection.
- Provide an API for other applications to use.
