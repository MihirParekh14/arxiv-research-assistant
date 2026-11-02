"""Tests for turning text into vectors.

No test here loads the real model. A stand-in encoder returns fixed vectors,
which keeps the suite fast, offline, and able to assert exact numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.embed import (
    block_path,
    clear_blocks,
    embed_in_blocks,
    embed_texts,
    load_blocks,
    load_vectors,
    normalize,
    save_vectors,
)


class FakeEncoder:
    """Returns a predictable vector per text, and records how it was called."""

    def __init__(self):
        self.batch_sizes: list[int] = []
        self.texts_seen: list[str] = []

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        self.batch_sizes.append(batch_size)
        self.texts_seen.extend(texts)
        return np.array([[float(len(t)), 1.0, 0.0, 0.0] for t in texts], dtype="float32")


def many(count: int) -> list[str]:
    """Distinct texts whose first vector component is their length."""
    return [f"t{'x' * i}" for i in range(count)]


# --- normalize -------------------------------------------------------------


def test_normalized_vectors_have_length_one():
    result = normalize(np.array([[3.0, 4.0], [5.0, 12.0]]))
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)


def test_normalize_keeps_direction():
    """Only the length changes, so the ratio between components is preserved."""
    result = normalize(np.array([[3.0, 4.0]]))
    assert np.allclose(result, [[0.6, 0.8]])


def test_normalize_leaves_a_zero_vector_alone_instead_of_producing_nan():
    """Dividing by zero would give nan, which poisons every later comparison."""
    result = normalize(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert not np.isnan(result).any()
    assert np.allclose(result[0], [0.0, 0.0])


def test_normalize_output_is_float32():
    """FAISS wants float32. Handing it float64 raises at index time."""
    assert normalize(np.array([[1.0, 2.0]], dtype="float64")).dtype == np.float32


def test_normalize_rejects_a_single_flat_vector():
    with pytest.raises(ValueError, match="2-D array"):
        normalize(np.array([1.0, 2.0, 3.0]))


# --- embed_texts -----------------------------------------------------------


def test_embed_texts_returns_one_unit_vector_per_text():
    vectors = embed_texts(["a", "bb", "ccc"], encoder=FakeEncoder(), show_progress=False)

    assert vectors.shape == (3, 4)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_embed_texts_on_an_empty_list_returns_an_empty_array_not_a_crash():
    vectors = embed_texts([], encoder=FakeEncoder(), show_progress=False)
    assert vectors.shape[0] == 0


def test_embed_texts_passes_the_batch_size_through():
    encoder = FakeEncoder()
    embed_texts(["a", "b"], encoder=encoder, batch_size=16, show_progress=False)
    assert encoder.batch_sizes == [16]


def test_embed_texts_keeps_input_order():
    """Row i must belong to text i, or every citation later is wrong."""
    vectors = embed_texts(["a", "bbbb"], encoder=FakeEncoder(), show_progress=False)

    # The fake encoder's first component is the text length, so the longer
    # text must dominate its own row more than the shorter one does.
    assert vectors[1][0] > vectors[0][0]


# --- saving and loading ----------------------------------------------------


def test_save_then_load_returns_the_same_vectors(tmp_path):
    vectors = normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
    path = tmp_path / "vectors.npy"

    save_vectors(vectors, path)
    assert np.allclose(load_vectors(path), vectors)


def test_save_vectors_creates_missing_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "vectors.npy"
    save_vectors(np.array([[1.0, 0.0]]), path)
    assert path.exists()


def test_saved_vectors_are_float32(tmp_path):
    path = tmp_path / "vectors.npy"
    save_vectors(np.array([[1.0, 2.0]], dtype="float64"), path)
    assert load_vectors(path).dtype == np.float32


def test_load_vectors_raises_a_helpful_error_when_file_is_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        load_vectors(tmp_path / "nothing.npy")


# --- block encoding and resume ---------------------------------------------
#
# Encoding a quarter of a million passages in one call exhausts memory and the
# machine starts paging, which slows it by a factor of sixty rather than a few
# percent. Blocks keep memory flat and let an interrupted run continue. These
# tests cover the parts of that which fail quietly.


def test_blocks_cover_every_text_including_a_short_final_block(tmp_path):
    written = embed_in_blocks(many(25), FakeEncoder(), tmp_path, block_size=10)

    assert written == 25
    assert np.load(block_path(tmp_path, 0)).shape == (10, 4)
    assert np.load(block_path(tmp_path, 10)).shape == (10, 4)
    assert np.load(block_path(tmp_path, 20)).shape == (5, 4)


def test_joined_blocks_match_a_single_pass_exactly(tmp_path):
    texts = many(25)
    embed_in_blocks(texts, FakeEncoder(), tmp_path, block_size=10)

    joined = load_blocks(tmp_path, total=25, block_size=10)
    one_pass = embed_texts(texts, encoder=FakeEncoder(), show_progress=False)
    assert np.allclose(joined, one_pass)


def test_a_resumed_run_only_encodes_what_is_missing(tmp_path):
    texts = many(25)
    embed_in_blocks(texts, FakeEncoder(), tmp_path, block_size=10)
    block_path(tmp_path, 20).unlink()

    encoder = FakeEncoder()
    embed_in_blocks(texts, encoder, tmp_path, block_size=10)

    assert len(encoder.texts_seen) == 5
    assert encoder.texts_seen == texts[20:]


def test_fresh_run_re_encodes_everything(tmp_path):
    texts = many(25)
    embed_in_blocks(texts, FakeEncoder(), tmp_path, block_size=10)

    encoder = FakeEncoder()
    embed_in_blocks(texts, encoder, tmp_path, block_size=10, resume=False)
    assert len(encoder.texts_seen) == 25


def test_a_truncated_block_is_re_encoded_rather_than_trusted(tmp_path):
    """A run killed mid-write leaves a short file. Reusing it would shift every
    later vector out of line with its text."""
    texts = many(25)
    embed_in_blocks(texts, FakeEncoder(), tmp_path, block_size=10)
    np.save(block_path(tmp_path, 10), np.zeros((3, 4), dtype="float32"))

    encoder = FakeEncoder()
    embed_in_blocks(texts, encoder, tmp_path, block_size=10)
    assert encoder.texts_seen == texts[10:20]


def test_a_corpus_of_a_different_size_discards_the_old_blocks(tmp_path):
    """Resuming across two different corpora would stitch together vectors from
    both, and every citation after the seam would be wrong."""
    embed_in_blocks(many(25), FakeEncoder(), tmp_path, block_size=10)

    encoder = FakeEncoder()
    embed_in_blocks(many(15), encoder, tmp_path, block_size=10)

    assert len(encoder.texts_seen) == 15
    assert not block_path(tmp_path, 20).exists()


def test_changing_the_embedding_model_discards_the_old_blocks(tmp_path):
    """Nothing about the shapes changes, so this fails silently otherwise."""
    texts = many(20)
    embed_in_blocks(texts, FakeEncoder(), tmp_path, block_size=10, model_name="old")

    encoder = FakeEncoder()
    embed_in_blocks(texts, encoder, tmp_path, block_size=10, model_name="new")
    assert len(encoder.texts_seen) == 20


def test_changing_the_block_size_discards_the_old_blocks(tmp_path):
    """Old blocks start at different row offsets, so they cannot be reused."""
    texts = many(20)
    embed_in_blocks(texts, FakeEncoder(), tmp_path, block_size=10)

    encoder = FakeEncoder()
    embed_in_blocks(texts, encoder, tmp_path, block_size=5)
    assert len(encoder.texts_seen) == 20


def test_a_block_size_of_zero_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="block_size must be positive"):
        embed_in_blocks(many(5), FakeEncoder(), tmp_path, block_size=0)


def test_clear_blocks_removes_blocks_and_manifest(tmp_path):
    embed_in_blocks(many(25), FakeEncoder(), tmp_path, block_size=10)
    clear_blocks(tmp_path)

    assert list(tmp_path.glob("vectors_*.npy")) == []
    assert not (tmp_path / "manifest.json").exists()


def test_load_blocks_names_the_missing_block_instead_of_returning_short(tmp_path):
    embed_in_blocks(many(25), FakeEncoder(), tmp_path, block_size=10)
    block_path(tmp_path, 10).unlink()

    with pytest.raises(FileNotFoundError, match="build_index"):
        load_blocks(tmp_path, total=25, block_size=10)


def test_no_temporary_files_survive_a_finished_run(tmp_path):
    embed_in_blocks(many(25), FakeEncoder(), tmp_path, block_size=10)
    assert list(tmp_path.glob("*.tmp")) == []
