"""Lightweight RAG building blocks.

This module keeps document extraction, chunking, retrieval, and generation
separate so each stage can be tested independently.
"""

from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


class DocumentStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks: List[str] = []

    @staticmethod
    def extract_text(pdf_path: str) -> str:
        reader = PdfReader(str(Path(pdf_path)))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 800) -> List[str]:
        cleaned = " ".join(text.split())
        return [cleaned[i:i + chunk_size] for i in range(0, len(cleaned), chunk_size) if cleaned[i:i + chunk_size].strip()]

    def add_pdf(self, pdf_path: str) -> None:
        chunks = self.chunk_text(self.extract_text(pdf_path))
        if not chunks:
            return
        embeddings = self.encoder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if self.index is None or not self.chunks:
            return []
        vector = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(vector, min(k, len(self.chunks)))
        return [(self.chunks[index], float(score)) for score, index in zip(scores[0], indices[0]) if index >= 0]


if __name__ == "__main__":
    print("DocumentStore ready. Add a PDF with DocumentStore.add_pdf() and query it with search().")
