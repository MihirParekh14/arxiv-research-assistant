"""Build and query the vector index.

An index answers one question fast: given a query vector, which stored vectors
are closest? Without it, every search would compare the query against all
49,000 chunks one at a time.

FAISS is imported inside the functions that need it, so importing this module
costs nothing and the pure-numpy parts stay testable on their own.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import config


def build_index(vectors: np.ndarray):
    """Build an exact inner-product index over ``vectors``.

    Two choices worth defending.

    **Inner product, not Euclidean distance.** The vectors arrive normalised to
    length 1 (see :func:`src.embed.normalize`), and for unit vectors the inner
    product *is* the cosine of the angle between them. So a plain dot product
    gives cosine similarity with no extra work, and higher means more similar.

    **Flat, not approximate.** ``IndexFlatIP`` compares the query against every
    stored vector -- no approximation, no training step, and the top result is
    guaranteed correct. Approximate indexes (IVF, HNSW) trade some of that
    accuracy for speed, and they earn their keep at millions of vectors. At
    49,000 a flat scan is roughly 19 million multiply-adds, which takes a few
    milliseconds. Paying for approximation here would cost recall and buy
    nothing.
    """
    import faiss  # noqa: PLC0415

    vectors = np.asarray(vectors, dtype="float32")
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D array of vectors, got shape {vectors.shape}")
    if vectors.shape[0] == 0:
        raise ValueError("cannot build an index from zero vectors")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search_index(index, query_vectors: np.ndarray, top_k: int = config.TOP_K):
    """Return ``(scores, positions)`` for each query vector.

    ``positions`` are row numbers into the vector array, which are also line
    numbers in ``chunks.jsonl``. That shared numbering is the only link between
    a vector and the text it came from.

    ``top_k`` is clamped to the size of the index. Asking FAISS for more
    neighbours than it holds returns ``-1`` padding positions, which would turn
    into a crash or a wrong citation further down.
    """
    query_vectors = np.asarray(query_vectors, dtype="float32")
    if query_vectors.ndim == 1:
        query_vectors = query_vectors.reshape(1, -1)

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    return index.search(query_vectors, min(top_k, index.ntotal))


def save_index(index, path: str | Path) -> None:
    """Write the index to disk so searching does not rebuild it every time."""
    import faiss  # noqa: PLC0415

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: str | Path):
    """Read an index back from disk."""
    import faiss  # noqa: PLC0415

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.build_index` first.")
    return faiss.read_index(str(path))


def save_index_meta(
    path: str | Path,
    model_name: str,
    dim: int,
    vector_count: int,
    chunks_file: str,
) -> None:
    """Record which model and corpus produced this index.

    An index is only meaningful alongside the model that built it. Query a
    MiniLM index with vectors from a different model and nothing errors -- the
    shapes may even match -- you just get nonsense results that look like real
    ones. Writing the model name down lets the search step refuse that case
    instead of answering confidently from noise.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embedding_model": model_name,
        "embedding_dim": dim,
        "vector_count": vector_count,
        "chunks_file": chunks_file,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_index_meta(path: str | Path) -> dict:
    """Read the index metadata written by :func:`save_index_meta`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.build_index` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def check_index_matches(meta: dict, chunk_count: int, model_name: str) -> None:
    """Fail loudly when the index does not match the corpus or the model.

    Both of these go wrong quietly otherwise. Re-running the ingest with a
    different ``--limit`` leaves an index built over the old corpus, so result
    number 4,000 points at whatever text now sits on line 4,000 -- a real
    passage from the wrong paper. Switching the embedding model is worse,
    because nothing about the shapes changes.
    """
    if meta.get("vector_count") != chunk_count:
        raise ValueError(
            f"index holds {meta.get('vector_count')} vectors but chunks.jsonl has "
            f"{chunk_count} chunks. They must line up. "
            f"Re-run `python -m src.build_index`."
        )
    if meta.get("embedding_model") != model_name:
        raise ValueError(
            f"index was built with {meta.get('embedding_model')} but the current "
            f"model is {model_name}. Re-run `python -m src.build_index`."
        )
