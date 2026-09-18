import pytest
import rag_system as rag


def test_chunk_overlap_validation():
    with pytest.raises(ValueError):
        rag.DocumentStore._split("hello", chunk_size=10, overlap=10)


def test_chunking_cleans_whitespace():
    chunks = rag.DocumentStore._split("hello   world\nagain", chunk_size=20, overlap=2)
    assert chunks == ["hello world again"]


def test_chunking_rejects_non_text_input():
    with pytest.raises(TypeError, match="text must be a string"):
        rag.DocumentStore._split(None)


def test_chunking_empty_string():
    chunks = rag.DocumentStore._split("   ")
    assert chunks == []


def test_search_validates_k():
    store = object.__new__(rag.DocumentStore)
    store.index = None
    store.chunks = []
    with pytest.raises(ValueError, match="positive integer"):
        store.search("question", k=0)


def test_context_contains_source_and_page():
    chunk = rag.Chunk("Machine learning pipeline details", "annual_report.pdf", 3)
    context = rag.build_context([(chunk, 0.912)])
    assert "annual_report.pdf" in context
    assert "Page 3" in context
    assert "0.912" in context
    assert "Machine learning pipeline details" in context


def test_build_context_empty():
    assert rag.build_context([]) == ""
