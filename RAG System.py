"""End-to-end local RAG pipeline.

The pipeline extracts PDF text, chunks it, creates normalized embeddings,
retrieves the most relevant chunks, and optionally generates an answer with a
small local seq2seq model. The retrieval layer remains usable without a
text-generation model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int


class DocumentStore:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        self.index = None
        self.chunks: List[Chunk] = []

    @staticmethod
    def _split(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
        cleaned = " ".join(text.split())
        step = chunk_size - overlap
        return [cleaned[i:i + chunk_size] for i in range(0, len(cleaned), step) if cleaned[i:i + chunk_size].strip()]

    def add_pdf(self, pdf_path: str) -> int:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(path)
        reader = PdfReader(str(path))
        new_chunks = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for chunk in self._split(text):
                new_chunks.append(Chunk(chunk, path.name, page_number))
        if not new_chunks:
            return 0

        embeddings = self.encoder.encode(
            [c.text for c in new_chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def search(self, query: str, k: int = 5) -> List[tuple[Chunk, float]]:
        if not query.strip() or self.index is None or not self.chunks:
            return []
        vector = self.encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        scores, indices = self.index.search(vector, min(k, len(self.chunks)))
        return [
            (self.chunks[i], float(score))
            for score, i in zip(scores[0], indices[0])
            if i >= 0
        ]


def build_context(results: List[tuple[Chunk, float]]) -> str:
    return "\n\n".join(
        f"[{chunk.source}, page {chunk.page}, score={score:.3f}]\n{chunk.text}"
        for chunk, score in results
    )


if __name__ == "__main__":
    print("RAG DocumentStore ready. Use app.py for the interactive application.")
