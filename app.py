"""Streamlit interface for the local RAG pipeline."""

import tempfile
from pathlib import Path
from importlib.machinery import SourceFileLoader

import streamlit as st
from transformers import pipeline

rag = SourceFileLoader("rag_core", "RAG System.py").load_module()

st.set_page_config(page_title="RAG Document Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG Document Assistant")
st.caption("Upload PDFs, retrieve evidence, and optionally generate an answer from retrieved context.")

@st.cache_resource
def get_store():
    return rag.DocumentStore()

@st.cache_resource
def get_generator():
    return pipeline("text2text-generation", model="google/flan-t5-small")

store = get_store()
files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

if files:
    indexed_sources = {c.source for c in store.chunks}
    for uploaded in files:
        if uploaded.name in indexed_sources:
            continue
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = Path(tmp.name)
        count = store.add_pdf(str(tmp_path), source_name=uploaded.name)
        indexed_sources.add(uploaded.name)
        st.success(f"Indexed {uploaded.name}: {count} chunks")

query = st.text_input("Ask a question about your documents")
k = st.slider("Retrieved chunks", 1, 8, 4)
generate = st.checkbox("Generate an answer with FLAN-T5 (downloads a local model)")

if query:
    results = store.search(query, k=k)
    if not results:
        st.warning("No relevant content found. Upload a PDF and try again.")
    else:
        if generate:
            context = rag.build_context(results)
            prompt = (
                "Answer using only the supplied context. If the context is insufficient, say so.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}"
            )
            with st.spinner("Generating answer..."):
                answer = get_generator()(prompt, max_new_tokens=180, do_sample=False)[0]["generated_text"]
            st.subheader("Answer")
            st.write(answer)

        st.subheader("Retrieved evidence")
        for chunk, score in results:
            with st.expander(f"{chunk.source} — page {chunk.page} — similarity {score:.3f}"):
                st.write(chunk.text)
