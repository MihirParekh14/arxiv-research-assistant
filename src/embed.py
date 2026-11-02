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

import json
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


def block_path(blocks_dir: str | Path, start: int) -> Path:
    """Where the block beginning at row ``start`` is written."""
    return Path(blocks_dir) / f"vectors_{start:09d}.npy"


def _manifest_path(blocks_dir: str | Path) -> Path:
    return Path(blocks_dir) / "manifest.json"


def _completed_rows(blocks_dir: Path, start: int, expected: int) -> bool:
    """True when the block at ``start`` is on disk and the right length.

    The length check matters. A run killed partway through writing a block
    would otherwise leave a short file that resume treats as finished, and the
    missing rows would shift every vector after it out of line with its text.
    """
    path = block_path(blocks_dir, start)
    if not path.exists():
        return False
    try:
        return np.load(path, mmap_mode="r").shape[0] == expected
    except (ValueError, OSError):
        return False


def clear_blocks(blocks_dir: str | Path) -> None:
    """Delete every block and the manifest."""
    blocks_dir = Path(blocks_dir)
    if not blocks_dir.exists():
        return
    for path in blocks_dir.glob("vectors_*.npy"):
        path.unlink()
    _manifest_path(blocks_dir).unlink(missing_ok=True)


def embed_in_blocks(
    texts: Sequence[str],
    encoder,
    blocks_dir: str | Path,
    block_size: int = config.EMBED_BLOCK_SIZE,
    batch_size: int = config.EMBED_BATCH_SIZE,
    model_name: str = config.EMBEDDING_MODEL,
    resume: bool = True,
    on_block=None,
) -> int:
    """Encode ``texts`` a block at a time, writing each block to disk.

    Returns the number of vectors written.

    Resume is only safe when the inputs line up with what the earlier run was
    encoding, so a manifest records the corpus size, the block size and the
    model. If any of those changed, the old blocks describe a different corpus
    and are deleted rather than silently reused -- otherwise a resumed run
    would stitch together vectors from two different datasets and every
    citation after the seam would point at the wrong paper.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    blocks_dir = Path(blocks_dir)
    blocks_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "total": len(texts),
        "block_size": block_size,
        "embedding_model": model_name,
    }
    existing = None
    if _manifest_path(blocks_dir).exists():
        try:
            existing = json.loads(_manifest_path(blocks_dir).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None

    if not resume or existing != manifest:
        clear_blocks(blocks_dir)
    _manifest_path(blocks_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for start in range(0, len(texts), block_size):
        rows = min(block_size, len(texts) - start)

        if _completed_rows(blocks_dir, start, rows):
            if on_block:
                on_block(start, rows, True)
            continue

        vectors = embed_texts(
            texts[start : start + rows],
            encoder=encoder,
            batch_size=batch_size,
            show_progress=False,
        )

        # Write to a temporary name and rename, so a run killed mid-write
        # leaves no half-finished block for the next run to trust.
        # np.save appends ".npy" when handed a path, so write through an open
        # file to keep the temporary name exactly as given.
        temp = block_path(blocks_dir, start).with_suffix(".npy.tmp")
        with temp.open("wb") as handle:
            np.save(handle, vectors)
        temp.replace(block_path(blocks_dir, start))

        if on_block:
            on_block(start, rows, False)

    return len(texts)


def load_blocks(blocks_dir: str | Path, total: int, block_size: int) -> np.ndarray:
    """Read every block back and join them into one array, in row order."""
    blocks_dir = Path(blocks_dir)
    parts = []
    for start in range(0, total, block_size):
        path = block_path(blocks_dir, start)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing. Re-run `python -m src.build_index` to finish encoding."
            )
        parts.append(np.load(path))

    if not parts:
        return np.zeros((0, config.EMBEDDING_DIM), dtype="float32")
    return np.concatenate(parts, axis=0)


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
