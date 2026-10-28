"""Tests for splitting papers into overlapping chunks.

Chunking is where a retrieval system quietly goes wrong: an off-by-one in the
step size either loses text between chunks or repeats it forever. These tests
pin down the boundaries.
"""

from __future__ import annotations

import pytest

from src.chunking import Chunk, chunk_paper, chunk_papers, chunk_stats, split_words
from src.data import Paper


def words(count: int) -> str:
    """A text of exactly ``count`` distinct, countable words."""
    return " ".join(f"w{i}" for i in range(count))


def make_paper(word_count: int, paper_id: str = "2401.00001") -> Paper:
    return Paper(
        paper_id=paper_id,
        title="A Study of Things",
        abstract="We study things.",
        text=words(word_count),
        authors="A. Researcher",
        published="2024-01-01",
        url="https://arxiv.org/abs/2401.00001",
    )


# --- argument validation ---------------------------------------------------


def test_overlap_equal_to_chunk_size_is_rejected():
    """Equal values give a step of zero, which would loop forever."""
    with pytest.raises(ValueError, match="never advances"):
        split_words(words(500), chunk_words=100, overlap_words=100)


def test_overlap_larger_than_chunk_size_is_rejected():
    with pytest.raises(ValueError, match="never advances"):
        split_words(words(500), chunk_words=100, overlap_words=150)


def test_negative_overlap_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        split_words(words(500), chunk_words=100, overlap_words=-5)


def test_zero_chunk_size_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        split_words(words(500), chunk_words=0)


# --- sizes and boundaries --------------------------------------------------

def test_text_shorter_than_minimum_produces_no_chunks():
    assert split_words(words(5), chunk_words=100, overlap_words=20, min_words=25) == []


def test_text_exactly_one_chunk_long_produces_one_chunk():
    chunks = split_words(words(100), chunk_words=100, overlap_words=20, min_words=25)
    assert len(chunks) == 1
    assert chunks[0].split()[0] == "w0"
    assert chunks[0].split()[-1] == "w99"


def test_text_between_minimum_and_chunk_size_produces_one_short_chunk():
    chunks = split_words(words(40), chunk_words=100, overlap_words=20, min_words=25)
    assert len(chunks) == 1
    assert len(chunks[0].split()) == 40


def test_no_chunk_ever_exceeds_the_configured_size():
    chunks = split_words(words(1000), chunk_words=100, overlap_words=20)
    assert all(len(chunk.split()) <= 100 for chunk in chunks)


def test_trailing_fragment_below_the_minimum_is_dropped():
    """80 words at step 50 leaves a 30-word tail; min_words=40 drops it."""
    chunks = split_words(words(80), chunk_words=60, overlap_words=10, min_words=40)
    assert all(len(chunk.split()) >= 40 for chunk in chunks)


# --- overlap behaviour -----------------------------------------------------


def test_consecutive_chunks_share_exactly_the_overlap():
    chunks = split_words(words(300), chunk_words=100, overlap_words=20, min_words=10)

    first_tail = chunks[0].split()[-20:]
    second_head = chunks[1].split()[:20]
    assert first_tail == second_head


def test_chunk_starts_advance_by_the_step_size():
    chunks = split_words(words(300), chunk_words=100, overlap_words=20, min_words=10)

    starts = [int(chunk.split()[0].lstrip("w")) for chunk in chunks]
    assert starts[:3] == [0, 80, 160]


def test_zero_overlap_produces_no_shared_words():
    chunks = split_words(words(300), chunk_words=100, overlap_words=0, min_words=10)

    assert chunks[0].split()[-1] == "w99"
    assert chunks[1].split()[0] == "w100"


def test_every_word_appears_in_at_least_one_chunk():
    """Nothing may fall through the gap between two chunks."""
    chunks = split_words(words(500), chunk_words=100, overlap_words=20, min_words=10)

    covered = {word for chunk in chunks for word in chunk.split()}
    assert covered == {f"w{i}" for i in range(500)}


# --- chunk records ---------------------------------------------------------


def test_chunk_carries_its_source_metadata():
    """A retrieved chunk must be able to name the paper it came from."""
    chunks = chunk_paper(make_paper(300), chunk_words=100, overlap_words=20)

    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.paper_id == "2401.00001" for chunk in chunks)
    assert all(chunk.title == "A Study of Things" for chunk in chunks)


def test_chunk_ids_are_unique_and_indices_run_in_order():
    chunks = chunk_paper(make_paper(500), chunk_words=100, overlap_words=20)

    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_chunk_word_count_matches_its_text():
    chunks = chunk_paper(make_paper(300), chunk_words=100, overlap_words=20)
    assert all(chunk.word_count == len(chunk.text.split()) for chunk in chunks)


def test_chunk_papers_keeps_papers_separate():
    """Chunks must never merge text across two different papers."""
    papers = [make_paper(200, paper_id="a"), make_paper(200, paper_id="b")]
    chunks = list(chunk_papers(papers, chunk_words=100, overlap_words=20))

    assert {chunk.paper_id for chunk in chunks} == {"a", "b"}
    for chunk in chunks:
        assert chunk.chunk_id.startswith(f"{chunk.paper_id}::")


def test_a_paper_too_short_to_chunk_yields_nothing():
    chunks = chunk_paper(make_paper(5), chunk_words=100, overlap_words=20)
    assert chunks == []


# --- stats -----------------------------------------------------------------


def test_chunk_stats_reports_counts_and_sizes():
    chunks = chunk_paper(make_paper(500), chunk_words=100, overlap_words=20)
    stats = chunk_stats(chunks)

    assert stats["chunks"] == len(chunks)
    assert stats["max_words"] <= 100
    assert stats["min_words"] <= stats["mean_words"] <= stats["max_words"]


def test_chunk_stats_on_an_empty_list_does_not_divide_by_zero():
    assert chunk_stats([]) == {
        "chunks": 0,
        "min_words": 0,
        "max_words": 0,
        "mean_words": 0.0,
    }
