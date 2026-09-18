"""Backward-compatibility wrapper for rag_system module."""

from rag_system import Chunk, DocumentStore, build_context, cli

__all__ = ["Chunk", "DocumentStore", "build_context", "cli"]

if __name__ == "__main__":
    cli()
