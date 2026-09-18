"""
Streamlit Web Interface for Local RAG Document Assistant.
Provides document ingestion, semantic search, evidence inspection, and optional FLAN-T5 generation.
"""

from pathlib import Path
import tempfile
import streamlit as st

import rag_system as rag

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📚 Local RAG Document Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Grounded Question-Answering with Vector Similarity Search & Evidence Citations</div>', unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading embedding model (all-MiniLM-L6-v2)...")
def get_store():
    return rag.DocumentStore()

@st.cache_resource(show_spinner="Loading FLAN-T5 generation model...")
def get_generator():
    from transformers import pipeline
    return pipeline("text2text-generation", model="google/flan-t5-small")

store = get_store()

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    k = st.slider("Top-K Passages", min_value=1, max_value=8, value=4, help="Number of relevant chunks to retrieve")
    min_score = st.slider("Minimum Similarity Threshold", min_value=0.0, max_value=1.0, value=0.15, step=0.05)
    generate = st.checkbox("Generate Synthesized Answer (FLAN-T5)", value=False, help="Uses local text2text model to synthesize final answer from context")

    st.divider()
    st.markdown("### 📊 Store Statistics")
    st.metric(label="Indexed Chunks", value=store.total_chunks)

    if st.button("🗑️ Clear Indexed Documents", use_container_width=True):
        store.clear()
        st.success("Document store cleared.")
        st.rerun()

# Document Ingestion Section
st.subheader("1. Ingest Documents")
files = st.file_uploader(
    "Upload PDF or Text files",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
    help="Upload source documents to index into the FAISS vector store"
)

if files:
    indexed_sources = {c.source for c in store.chunks}
    for uploaded in files:
        if uploaded.name in indexed_sources:
            continue

        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = Path(tmp.name)

        try:
            count = store.add_file(tmp_path, source_name=uploaded.name)
        except Exception as error:
            st.error(f"Could not index {uploaded.name}: {error}")
        else:
            indexed_sources.add(uploaded.name)
            if count > 0:
                st.success(f"✓ Indexed **{uploaded.name}** ({count} chunks)")
            else:
                st.warning(f"⚠️ No extractable text found in **{uploaded.name}**")
        finally:
            tmp_path.unlink(missing_ok=True)

# Query & Retrieval Section
st.subheader("2. Ask a Question")
query = st.text_input(
    "Search or query your indexed documents:",
    placeholder="e.g. What were the quarterly financial findings or operational metrics?",
)

if query:
    results = store.search(query, k=k, min_score=min_score)
    if not results:
        st.warning("No relevant passages found matching your query above the similarity threshold.")
    else:
        if generate:
            context = rag.build_context(results)
            prompt = (
                "Answer the question concisely using only the supplied context. "
                "If the context is insufficient, state that the documents do not contain the answer.\n\n"
                f"Context:\n{context}\n\nQuestion: {query}"
            )
            with st.spinner("Generating synthesized answer..."):
                gen_pipeline = get_generator()
                output = gen_pipeline(prompt, max_new_tokens=180, do_sample=False)
                answer = output[0]["generated_text"]

            st.markdown("### 💡 Synthesized Answer")
            st.info(answer)

        st.markdown(f"### 📑 Retrieved Evidence ({len(results)} matches)")
        for rank, (chunk, score) in enumerate(results, 1):
            with st.expander(f"Match #{rank} — **{chunk.source}** (Page {chunk.page}) — Similarity: **{score:.3f}**", expanded=(rank == 1)):
                st.markdown(chunk.text)
