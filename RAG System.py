"""End-to-end local RAG pipeline with page-aware evidence."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import faiss
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
        """Return normalized, overlapping text chunks.

        ``text`` can be empty (as is common for scanned PDF pages), but it
        must be a string so extraction errors do not get silently obscured.
        """
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        cleaned = " ".join(text.split())
        step = chunk_size - overlap
        return [cleaned[i:i + chunk_size] for i in range(0, len(cleaned), step) if cleaned[i:i + chunk_size].strip()]

    def add_pdf(self, pdf_path: str, source_name: Optional[str] = None) -> int:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(path)
        source = source_name or path.name
        reader = PdfReader(str(path))
        new_chunks = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for chunk in self._split(text):
                new_chunks.append(Chunk(chunk, source, page_number))
        if not new_chunks:
            return 0
        embeddings = self.encoder.encode(
            [c.text for c in new_chunks], convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype("float32")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(new_chunks):
            raise ValueError("The embedding model returned an unexpected embedding shape")
        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
        elif embeddings.shape[1] != self.index.d:
            raise ValueError("Embedding dimension does not match the existing index")
        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def search(self, query: str, k: int = 5) -> List[tuple[Chunk, float]]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not query.strip() or self.index is None or not self.chunks:
            return []
        vector = self.encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        if vector.ndim != 2 or vector.shape != (1, self.index.d):
            raise ValueError("The embedding model returned an unexpected query shape")
        scores, indices = self.index.search(vector, min(k, len(self.chunks)))
        return [(self.chunks[i], float(score)) for score, i in zip(scores[0], indices[0]) if i >= 0]


def build_context(results: List[tuple[Chunk, float]]) -> str:
    return "\n\n".join(
        f"[{chunk.source}, page {chunk.page}, score={score:.3f}]\n{chunk.text}"
        for chunk, score in results
    )


if __name__ == "__main__":
    print("RAG DocumentStore ready. Use app.py for the interactive application.")
