"""Turn text into vectors.

A chunk of text cannot be searched by meaning. A vector can. The embedding
model reads a passage and returns 384 numbers that place it in a space where
passages about the same thing sit close together, even when they share no
words at all -- "image inpainting" and "filling masked regions" land near each
other because the model learned they mean the same thing.

Nothing in this module downloads anything at import time. The model is loaded
inside :func:`load_encoder`, so the tests here run with no model and no
network.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from src import config


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale every vector to length 1.

    This is what turns a dot product into cosine similarity, and cosine is the
    right measure for text. Raw dot products reward long vectors, and vector
    length tracks things like passage length rather than meaning -- so without
    this step a wordy paragraph would outrank a precise one on every query.

    A zero vector cannot be scaled to length 1. Dividing by its length would
    give ``nan``, which then silently poisons every comparison it takes part
    in, so its length is treated as 1 and the zero vector is left as it is.
    """
    vectors = np.asarray(vectors, dtype="float32")
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D array of vectors, got shape {vectors.shape}")

    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return vectors / lengths


def load_encoder(model_name: str = config.EMBEDDING_MODEL):
    """Load the sentence-transformer model.

    Imported inside the function so the rest of this module stays importable
    without torch installed, and so no test accidentally pulls a 90 MB model
    off the network.
    """
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    return SentenceTransformer(model_name)


def embed_texts(
    texts: Sequence[str],
    encoder,
    batch_size: int = config.EMBED_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode texts into a ``(len(texts), EMBEDDING_DIM)`` array of unit vectors.

    ``encoder`` is passed in rather than built here. That keeps this function
    testable with a stand-in encoder, and it means a caller encoding both a
    corpus and a query loads the model once instead of twice.
    """
    if len(texts) == 0:
        return np.zeros((0, config.EMBEDDING_DIM), dtype="float32")

    vectors = encoder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return normalize(vectors)


def save_vectors(vectors: np.ndarray, path: str | Path) -> None:
    """Write vectors to a ``.npy`` file.

    ``.npy`` rather than JSON or CSV: it stores the raw float32 block, so
    49,000 vectors load in a few milliseconds instead of being parsed number by
    number. Row order is the file's whole contract -- row *i* belongs to chunk
    *i* of ``chunks.jsonl``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(vectors, dtype="float32"))


def load_vectors(path: str | Path) -> np.ndarray:
    """Read vectors back from a ``.npy`` file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.build_index` first.")
    return np.load(path)
