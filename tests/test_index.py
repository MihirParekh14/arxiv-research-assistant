"""Tests for the vector index and end-to-end search.

The last group is the important one. It runs a real query through a real FAISS
index and checks that the passage handed back is the passage the query asked
for -- not just that *a* passage came back. Everything between a query and an
answer is numbers, so a result that points at the wrong row looks exactly like
a result that points at the right one.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.chunking import Chunk
from src.embed import normalize
from src.index import (
    build_index,
    check_index_matches,
    load_index,
    load_index_meta,
    save_index,
    save_index_meta,
    search_index,
)
from src.search import search

VOCAB = ("inpainting", "transformer", "protein")


class KeywordEncoder:
    """A tiny stand-in for the real model.

    Each text becomes a vector counting how often it mentions each vocabulary
    word. Texts about the same subject point the same way, which is the only
    property these tests need -- and it needs no download.
    """

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        rows = [[float(t.lower().count(word)) for word in VOCAB] for t in texts]
        return np.array(rows, dtype="float32")


def make_chunk(index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"paper{index}::0",
        paper_id=f"paper{index}",
        title=f"Paper About {text.split()[0].title()}",
        text=text,
        chunk_index=0,
        word_count=len(text.split()),
    )


CORPUS = [
    make_chunk(0, "inpainting fills masked regions of an image inpainting inpainting"),
    make_chunk(1, "transformer attention layers transformer transformer"),
    make_chunk(2, "protein folding structure prediction protein protein"),
]


# --- building --------------------------------------------------------------


def test_index_holds_one_entry_per_vector():
    index = build_index(normalize(np.eye(3)))
    assert index.ntotal == 3


def test_building_from_zero_vectors_is_rejected():
    with pytest.raises(ValueError, match="zero vectors"):
        build_index(np.zeros((0, 4), dtype="float32"))


def test_building_from_a_flat_array_is_rejected():
    with pytest.raises(ValueError, match="2-D array"):
        build_index(np.array([1.0, 0.0, 0.0]))


# --- searching -------------------------------------------------------------


def test_a_vector_finds_itself_first_with_a_score_of_one():
    """Cosine similarity of a unit vector with itself is exactly 1."""
    vectors = normalize(np.eye(4))
    index = build_index(vectors)

    scores, positions = search_index(index, vectors[2], top_k=1)

    assert positions[0][0] == 2
    assert scores[0][0] == pytest.approx(1.0, abs=1e-5)


def test_a_one_dimensional_query_is_accepted():
    """Callers should not have to reshape a single query by hand."""
    index = build_index(normalize(np.eye(3)))
    scores, positions = search_index(index, np.array([1.0, 0.0, 0.0]), top_k=1)
    assert positions.shape == (1, 1)


def test_top_k_larger_than_the_index_is_clamped_instead_of_padded():
    """FAISS pads with -1 positions, which would become a wrong citation."""
    index = build_index(normalize(np.eye(3)))
    scores, positions = search_index(index, np.eye(3)[0], top_k=50)

    assert positions.shape == (1, 3)
    assert (positions >= 0).all()


def test_top_k_of_zero_is_rejected():
    index = build_index(normalize(np.eye(3)))
    with pytest.raises(ValueError, match="top_k must be positive"):
        search_index(index, np.eye(3)[0], top_k=0)


def test_results_come_back_in_descending_score_order():
    vectors = normalize(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]))
    index = build_index(vectors)

    scores, _ = search_index(index, vectors[0], top_k=3)
    assert list(scores[0]) == sorted(scores[0], reverse=True)


# --- saving and loading ----------------------------------------------------


def test_a_saved_index_returns_the_same_results_after_loading(tmp_path):
    vectors = normalize(np.eye(5))
    path = tmp_path / "index.faiss"

    save_index(build_index(vectors), path)
    reloaded = load_index(path)

    _, positions = search_index(reloaded, vectors[3], top_k=1)
    assert positions[0][0] == 3


def test_load_index_raises_a_helpful_error_when_file_is_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        load_index(tmp_path / "nothing.faiss")


def test_index_metadata_survives_a_round_trip(tmp_path):
    path = tmp_path / "index_meta.json"
    save_index_meta(path, model_name="a-model", dim=384, vector_count=10, chunks_file="c.jsonl")

    meta = load_index_meta(path)
    assert meta["embedding_model"] == "a-model"
    assert meta["vector_count"] == 10


# --- guards against a stale index ------------------------------------------


def test_a_matching_index_passes_the_check():
    meta = {"embedding_model": "a-model", "vector_count": 3}
    check_index_matches(meta, chunk_count=3, model_name="a-model")


def test_an_index_built_over_a_different_corpus_is_refused():
    """Re-ingesting with a new --limit leaves an index pointing at old rows."""
    meta = {"embedding_model": "a-model", "vector_count": 49000}
    with pytest.raises(ValueError, match="must line up"):
        check_index_matches(meta, chunk_count=3, model_name="a-model")


def test_an_index_built_with_a_different_model_is_refused():
    """Nothing errors on its own here -- the results would just be nonsense."""
    meta = {"embedding_model": "old-model", "vector_count": 3}
    with pytest.raises(ValueError, match="was built with"):
        check_index_matches(meta, chunk_count=3, model_name="new-model")


# --- end to end ------------------------------------------------------------


def test_a_query_retrieves_the_passage_that_is_actually_about_it():
    """The whole point: the text handed back must match what was asked for."""
    encoder = KeywordEncoder()
    vectors = normalize(encoder.encode([chunk.text for chunk in CORPUS]))
    index = build_index(vectors)

    results = search("protein", list(CORPUS), index, encoder, top_k=1)

    assert results[0].chunk.paper_id == "paper2"
    assert "protein" in results[0].chunk.text


def test_search_returns_the_chunk_sitting_at_the_position_faiss_reported():
    """Guards the link between a vector's row number and its text."""
    encoder = KeywordEncoder()
    vectors = normalize(encoder.encode([chunk.text for chunk in CORPUS]))
    index = build_index(vectors)

    results = search("transformer", list(CORPUS), index, encoder, top_k=3)
    _, positions = search_index(index, normalize(encoder.encode(["transformer"])), top_k=3)

    returned_ids = [result.chunk.chunk_id for result in results]
    expected_ids = [CORPUS[position].chunk_id for position in positions[0]]
    assert returned_ids == expected_ids


def test_search_asks_for_no_more_results_than_the_corpus_holds():
    encoder = KeywordEncoder()
    index = build_index(normalize(encoder.encode([chunk.text for chunk in CORPUS])))

    results = search("inpainting", list(CORPUS), index, encoder, top_k=100)
    assert len(results) == len(CORPUS)


# --- the per-paper cap -----------------------------------------------------
#
# Measured on the real index: "how do diffusion models handle image inpainting"
# returned four of its five results from one paper. For search that is untidy;
# for a generated answer it means every citation points at a single source.


def big_corpus() -> list[Chunk]:
    """One paper that matches "inpainting" strongly across five chunks, and two
    weaker papers. Each hog chunk mixes in a little more of another vocabulary
    word, so their scores descend but all stay above the other two papers."""
    chunks = [
        Chunk(chunk_id=f"hog::{i}", paper_id="hog", title="The Hogging Paper",
              text="inpainting " * 5 + "transformer " * i, chunk_index=i,
              word_count=5 + i)
        for i in range(5)
    ]
    chunks += [
        Chunk(chunk_id="other1::0", paper_id="other1", title="Second Paper",
              text="inpainting inpainting transformer transformer",
              chunk_index=0, word_count=4),
        Chunk(chunk_id="other2::0", paper_id="other2", title="Third Paper",
              text="transformer transformer", chunk_index=0, word_count=2),
    ]
    return chunks


def built(corpus, encoder):
    return build_index(normalize(encoder.encode([c.text for c in corpus])))


def test_without_a_cap_one_paper_can_take_the_whole_result_set():
    """The behaviour the cap exists to stop."""
    encoder, corpus = KeywordEncoder(), big_corpus()
    results = search("inpainting", corpus, built(corpus, encoder), encoder,
                     top_k=4, max_per_paper=99)

    assert [r.chunk.paper_id for r in results].count("hog") == 4


def test_the_cap_limits_how_many_passages_one_paper_contributes():
    encoder, corpus = KeywordEncoder(), big_corpus()
    results = search("inpainting", corpus, built(corpus, encoder), encoder,
                     top_k=4, max_per_paper=2)

    assert [r.chunk.paper_id for r in results].count("hog") <= 2


def test_capping_pulls_in_other_papers_rather_than_returning_fewer_results():
    """Oversampling is what makes this work: the index is asked for more than
    is wanted, so discarded results are replaced rather than missing."""
    encoder, corpus = KeywordEncoder(), big_corpus()
    results = search("inpainting", corpus, built(corpus, encoder), encoder,
                     top_k=3, max_per_paper=2)

    assert len(results) == 3
    assert len({r.chunk.paper_id for r in results}) >= 2


def test_capped_results_stay_in_descending_score_order():
    encoder, corpus = KeywordEncoder(), big_corpus()
    results = search("inpainting", corpus, built(corpus, encoder), encoder,
                     top_k=4, max_per_paper=2)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_a_cap_of_one_gives_one_passage_per_paper():
    encoder, corpus = KeywordEncoder(), big_corpus()
    results = search("inpainting", corpus, built(corpus, encoder), encoder,
                     top_k=3, max_per_paper=1)

    paper_ids = [r.chunk.paper_id for r in results]
    assert len(paper_ids) == len(set(paper_ids))


def test_a_cap_of_zero_is_rejected():
    encoder, corpus = KeywordEncoder(), big_corpus()
    with pytest.raises(ValueError, match="max_per_paper must be positive"):
        search("inpainting", corpus, built(corpus, encoder), encoder, max_per_paper=0)


def test_scores_are_between_minus_one_and_one():
    """Cosine similarity cannot leave that range. Anything else means the
    vectors were never normalised."""
    encoder = KeywordEncoder()
    index = build_index(normalize(encoder.encode([chunk.text for chunk in CORPUS])))

    results = search("inpainting", list(CORPUS), index, encoder, top_k=3)
    assert all(-1.0 <= result.score <= 1.0 for result in results)
