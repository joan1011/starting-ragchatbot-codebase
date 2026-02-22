"""
Pytest configuration and shared fixtures.
All tests are run from the backend/ directory so imports resolve correctly.
"""
import sys
import os
import pytest

# Ensure backend/ is on the path so `import vector_store`, `import search_tools`, etc. work
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "..")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_search_results(documents=None, metadata=None, distances=None, error=None):
    """Build a SearchResults object with sensible defaults."""
    from vector_store import SearchResults
    return SearchResults(
        documents=documents or [],
        metadata=metadata or [],
        distances=distances or [],
        error=error,
    )


def make_fake_chroma_result(documents, metadatas, distances=None):
    """Return a dict in the shape ChromaDB returns from .query()."""
    n = len(documents)
    return {
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances or [0.1] * n],
        "ids": [[f"id_{i}" for i in range(n)]],
    }
