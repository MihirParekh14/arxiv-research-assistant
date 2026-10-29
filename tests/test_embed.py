"""Tests for turning text into vectors.

No test here loads the real model. A stand-in encoder returns fixed vectors,
which keeps the suite fast, offline, and able to assert exact numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.embed import embed_texts, load_vectors, normalize, save_vectors


class FakeEncoder:
    """Returns a predictable vector per text, and records how it was called."""

    def __init__(self):
        self.batch_sizes: list[int] = []

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        self.batch_sizes.append(batch_size)
        return np.array([[float(len(t)), 1.0, 0.0, 0.0] for t in texts], dtype="float32")


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
