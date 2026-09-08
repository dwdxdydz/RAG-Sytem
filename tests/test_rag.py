from importlib.machinery import SourceFileLoader

import pytest

rag = SourceFileLoader("rag_core", "RAG System.py").load_module()


def test_chunk_overlap_validation():
    with pytest.raises(ValueError):
        rag.DocumentStore._split("hello", chunk_size=10, overlap=10)


def test_chunking_cleans_whitespace():
    chunks = rag.DocumentStore._split("hello   world\nagain", chunk_size=20, overlap=2)
    assert chunks == ["hello world again"]


def test_context_contains_source_and_page():
    chunk = rag.Chunk("answer text", "report.pdf", 3)
    context = rag.build_context([(chunk, 0.91)])
    assert "report.pdf" in context
    assert "page 3" in context
