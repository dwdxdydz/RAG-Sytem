"""
Production-Ready Local Retrieval-Augmented Generation (RAG) Pipeline.
Provides document chunking, semantic vector indexing with FAISS, and source-attributed retrieval.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class Chunk:
    """Represents a text chunk with document metadata and page attribution."""
    text: str
    source: str
    page: int


class DocumentStore:
    """In-memory document store powered by FAISS vector similarity search.

    Supports PDF and plain text ingestion, configurable chunking with overlap,
    and normalized Inner Product (Cosine Similarity) retrieval.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model_name = embedding_model
        self.encoder = SentenceTransformer(embedding_model)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Chunk] = []

    @staticmethod
    def _split(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
        """Split text into normalized, overlapping character windows.

        Args:
            text: Input string to segment.
            chunk_size: Maximum character length per window.
            overlap: Character overlap between consecutive windows.

        Returns:
            List of non-empty text chunks.
        """
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        cleaned = " ".join(text.split())
        if not cleaned:
            return []

        step = chunk_size - overlap
        return [
            cleaned[i : i + chunk_size]
            for i in range(0, len(cleaned), step)
            if cleaned[i : i + chunk_size].strip()
        ]

    def add_text(self, text: str, source_name: str = "raw_text", page: int = 1) -> int:
        """Ingest plain text or markdown directly into the vector store."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        new_chunks = [Chunk(c, source_name, page) for c in self._split(text)]
        if not new_chunks:
            return 0

        self._index_chunks(new_chunks)
        return len(new_chunks)

    def add_pdf(self, pdf_path: str | Path, source_name: Optional[str] = None) -> int:
        """Extract text from a PDF file page-by-page and index all chunks."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        source = source_name or path.name
        reader = PdfReader(str(path))
        new_chunks: List[Chunk] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for chunk_text in self._split(text):
                new_chunks.append(Chunk(chunk_text, source, page_number))

        if not new_chunks:
            return 0

        self._index_chunks(new_chunks)
        return len(new_chunks)

    def add_file(self, file_path: str | Path, source_name: Optional[str] = None) -> int:
        """Automatically dispatch to PDF or text parser based on file suffix."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() == ".pdf":
            return self.add_pdf(path, source_name=source_name)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return self.add_text(text, source_name=source_name or path.name)

    def _index_chunks(self, new_chunks: List[Chunk]) -> None:
        """Compute normalized vector embeddings and insert into FAISS index."""
        embeddings = self.encoder.encode(
            [c.text for c in new_chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        if embeddings.ndim != 2 or embeddings.shape[0] != len(new_chunks):
            raise ValueError("The embedding model returned an unexpected embedding shape")

        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        elif dim != self.index.d:
            raise ValueError(f"Embedding dimension {dim} does not match existing index dimension {self.index.d}")

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)

    def search(self, query: str, k: int = 5, min_score: float = 0.0) -> List[Tuple[Chunk, float]]:
        """Perform semantic similarity search against indexed document chunks.

        Args:
            query: Natural language query.
            k: Maximum number of chunks to return.
            min_score: Minimum cosine similarity threshold (0.0 to 1.0).

        Returns:
            List of (Chunk, similarity_score) tuples ordered by descending relevance.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not query.strip() or self.index is None or not self.chunks:
            return []

        vector = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        if vector.ndim != 2 or vector.shape != (1, self.index.d):
            raise ValueError("The embedding model returned an unexpected query shape")

        scores, indices = self.index.search(vector, min(k, len(self.chunks)))

        results: List[Tuple[Chunk, float]] = []
        for score, i in zip(scores[0], indices[0]):
            if i >= 0 and float(score) >= min_score:
                results.append((self.chunks[i], float(score)))

        return results

    def clear(self) -> None:
        """Clear all indexed chunks and reset FAISS vector index."""
        self.index = None
        self.chunks.clear()

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)


def build_context(results: List[Tuple[Chunk, float]]) -> str:
    """Format retrieved chunks into structured context for LLM prompt ingestion."""
    if not results:
        return ""
    return "\n\n".join(
        f"[{chunk.source} | Page {chunk.page} | Similarity: {score:.3f}]\n{chunk.text}"
        for chunk, score in results
    )


def cli() -> None:
    """Command-line interface for testing RAG indexing and retrieval."""
    parser = argparse.ArgumentParser(description="Local RAG Document Assistant CLI")
    parser.add_argument("--demo", action="store_true", help="Run with a sample built-in knowledge base")
    parser.add_argument("--query", type=str, default="What are the key capabilities of this RAG system?", help="Query to search")
    parser.add_argument("-k", type=int, default=3, help="Top-k chunks to retrieve")
    args = parser.parse_args()

    if args.demo:
        print("🚀 Initializing DocumentStore and indexing sample knowledge base...")
        store = DocumentStore()
        sample_doc = (
            "Retrieval-Augmented Generation (RAG) is an AI framework that enhances Large Language Models "
            "by retrieving relevant facts from an external knowledge base before generating a response. "
            "This local RAG architecture uses sentence-transformers/all-MiniLM-L6-v2 for generating dense "
            "384-dimensional vector embeddings and Facebook AI Similarity Search (FAISS) for lightning-fast "
            "inner-product vector similarity search. By grounding responses in verified source documents, "
            "RAG drastically reduces hallucinations, provides verifiable citations, and keeps data completely private."
        )
        count = store.add_text(sample_doc, source_name="RAG_Architecture_Whitepaper.txt", page=1)
        print(f"✓ Indexed {count} chunks successfully.")

        print(f"\n🔍 Searching for: \"{args.query}\" (Top-{args.k})...")
        results = store.search(args.query, k=args.k)

        print("\n📄 Retrieved Context:")
        print(build_context(results))
    else:
        print("RAG DocumentStore ready. Run with --demo to test or launch 'streamlit run app.py' for the UI.")


if __name__ == "__main__":
    cli()
