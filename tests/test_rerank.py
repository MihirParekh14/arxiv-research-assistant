"""Tests for re-scoring a shortlist with a cross-encoder.

No model is loaded. A stand-in reranker returns scores chosen by the test, so
what is covered is the part that goes wrong silently: whether each score ends
up attached to the passage it was computed for, and whether the pipeline steps
happen in an order that lets reranking actually change the outcome.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.answer import confidence_for
from src.chunking import Chunk
from src.embed import normalize
from src.index import build_index
from src.rerank import attach_scores, rerank
from src.search import SearchResult, cap_per_paper, search

VOCAB = ("inpainting", "transformer", "protein")


class KeywordEncoder:
    """Stands in for the embedding model: counts vocabulary words."""

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
        rows = [[float(t.lower().count(word)) for word in VOCAB] for t in texts]
        return np.array(rows, dtype="float32")


def chunk(paper_id: str, index: int = 0, text: str = "a passage") -> Chunk:
    return Chunk(chunk_id=f"{paper_id}::{index}", paper_id=paper_id,
                 title=f"Paper {paper_id}", text=text, chunk_index=index,
                 word_count=len(text.split()))


def results_for(*specs) -> list[SearchResult]:
    """specs are (paper_id, retrieval_score, text) triples."""
    return [SearchResult(chunk=chunk(pid, 0, text), score=score)
            for pid, score, text in specs]


class FakeReranker:
    """Scores a pair by how often the query word appears in the passage."""

    def __init__(self):
        self.pairs_seen: list[tuple[str, str]] = []

    def predict(self, pairs, batch_size=32):
        self.pairs_seen.extend(pairs)
        return [float(passage.lower().count(query.lower())) for query, passage in pairs]


# --- attaching scores ------------------------------------------------------


def test_each_score_lands_on_the_passage_it_was_computed_for():
    """The quiet failure: scores shifted by one against their passages."""
    results = results_for(("a", 0.9, "x"), ("b", 0.8, "y"), ("c", 0.7, "z"))
    reordered = attach_scores(results, [1.0, 9.0, 5.0])

    by_paper = {r.chunk.paper_id: r.rerank_score for r in reordered}
    assert by_paper == {"a": 1.0, "b": 9.0, "c": 5.0}


def test_results_come_back_sorted_by_the_new_score():
    results = results_for(("a", 0.9, "x"), ("b", 0.8, "y"), ("c", 0.7, "z"))
    reordered = attach_scores(results, [1.0, 9.0, 5.0])

    assert [r.chunk.paper_id for r in reordered] == ["b", "c", "a"]


def test_the_retrieval_score_is_kept_not_overwritten():
    """Both numbers are needed to tell whether reranking is earning its cost."""
    reordered = attach_scores(results_for(("a", 0.91, "x")), [4.2])

    assert reordered[0].score == pytest.approx(0.91)
    assert reordered[0].rerank_score == pytest.approx(4.2)


def test_a_mismatched_number_of_scores_is_rejected():
    with pytest.raises(ValueError, match="must line up"):
        attach_scores(results_for(("a", 0.9, "x"), ("b", 0.8, "y")), [1.0])


def test_reranking_an_empty_shortlist_returns_nothing():
    assert rerank("q", [], FakeReranker()) == []


def test_the_reranker_is_shown_the_query_with_every_passage():
    """A cross-encoder only helps because it sees both at once."""
    reranker = FakeReranker()
    rerank("lora", results_for(("a", 0.9, "lora lora"), ("b", 0.8, "radio")), reranker)

    assert reranker.pairs_seen == [("lora", "lora lora"), ("lora", "radio")]


def test_reranking_can_overturn_the_retrieval_order():
    """If it could not, there would be no reason to run it."""
    results = results_for(("radio", 0.95, "chirp bandwidth"), ("lora", 0.60, "lora lora lora"))
    reordered = rerank("lora", results, FakeReranker())

    assert [r.chunk.paper_id for r in reordered] == ["lora", "radio"]


# --- capping ---------------------------------------------------------------


def test_capping_preserves_whatever_order_it_is_given():
    results = results_for(("a", 0.9, "x"), ("b", 0.8, "y"), ("a", 0.7, "z"))
    kept = cap_per_paper(results, top_k=3, max_per_paper=2)

    assert [r.score for r in kept] == [0.9, 0.8, 0.7]


def test_capping_stops_at_top_k():
    results = results_for(*[(f"p{i}", 0.9 - i / 100, "x") for i in range(10)])
    assert len(cap_per_paper(results, top_k=3)) == 3


def test_capping_skips_a_paper_that_has_had_its_allowance():
    results = results_for(("a", 0.9, "x"), ("a", 0.8, "y"), ("a", 0.7, "z"), ("b", 0.1, "w"))
    kept = cap_per_paper(results, top_k=3, max_per_paper=2)

    assert [r.chunk.paper_id for r in kept] == ["a", "a", "b"]


# --- the order of the pipeline ---------------------------------------------


def test_capping_runs_after_reranking_not_before():
    """The whole point of the ordering.

    Two passages from one paper are retrieved above a third passage that the
    reranker will promote to the top. With a cap of 1 applied *before*
    reranking, the second of those two would be dropped while it still looked
    worse than it is -- and reranking would never get to see it.

    Here the promoted passage must appear, and it must appear first.
    """
    corpus = [
        chunk("hog", 0, "inpainting inpainting inpainting"),
        chunk("hog", 1, "inpainting inpainting"),
        chunk("gold", 0, "inpainting transformer"),
    ]
    encoder = KeywordEncoder()
    index = build_index(normalize(encoder.encode([c.text for c in corpus])))

    class PromoteGold:
        def predict(self, pairs, batch_size=32):
            # The third passage is the one the embedding search ranked lowest.
            return [1.0 if "transformer" in passage else -1.0 for _, passage in pairs]

    kept = search("inpainting", corpus, index, encoder, top_k=2,
                  max_per_paper=1, reranker=PromoteGold())

    assert kept[0].chunk.paper_id == "gold"


def test_search_without_a_reranker_leaves_scores_unreranked():
    corpus = [chunk("a", 0, "inpainting"), chunk("b", 0, "transformer")]
    encoder = KeywordEncoder()
    index = build_index(normalize(encoder.encode([c.text for c in corpus])))

    results = search("inpainting", corpus, index, encoder, top_k=2)
    assert all(not r.was_reranked for r in results)
    assert all(r.relevance == r.score for r in results)


def test_search_with_a_reranker_marks_every_result_as_reranked():
    corpus = [chunk("a", 0, "inpainting"), chunk("b", 0, "transformer")]
    encoder = KeywordEncoder()
    index = build_index(normalize(encoder.encode([c.text for c in corpus])))

    results = search("inpainting", corpus, index, encoder, top_k=2,
                     reranker=FakeReranker())
    assert all(r.was_reranked for r in results)
    assert all(r.relevance == r.rerank_score for r in results)


def test_reranking_pulls_a_wider_shortlist_than_it_returns():
    """Reranking only helps if it is given more than the final answer needs."""
    corpus = [chunk(f"p{i}", 0, "inpainting " * (10 - i)) for i in range(10)]
    encoder = KeywordEncoder()
    index = build_index(normalize(encoder.encode([c.text for c in corpus])))

    reranker = FakeReranker()
    search("inpainting", corpus, index, encoder, top_k=2, candidates=8,
           reranker=reranker)

    assert len(reranker.pairs_seen) == 8


# --- confidence on two different scales ------------------------------------


def test_a_reranked_result_is_graded_on_the_cross_encoder_scale():
    """A logit of 4.0 is a strong match. Judged as a cosine it is nonsense."""
    reranked = [SearchResult(chunk=chunk("a"), score=0.31, rerank_score=4.0)]
    assert confidence_for(reranked) == "ok"


def test_a_low_cross_encoder_score_is_refused_even_when_cosine_looked_fine():
    """The case reranking exists for: retrieval liked it, reading it did not."""
    reranked = [SearchResult(chunk=chunk("a"), score=0.95, rerank_score=-8.0)]
    assert confidence_for(reranked) == "none"


def test_an_unreranked_result_still_uses_the_cosine_thresholds():
    assert confidence_for([SearchResult(chunk=chunk("a"), score=0.73)]) == "ok"
    assert confidence_for([SearchResult(chunk=chunk("a"), score=0.10)]) == "none"


def test_the_two_threshold_pairs_are_each_ordered():
    assert config.MIN_ANSWER_SCORE < config.WEAK_ANSWER_SCORE
    assert config.MIN_RERANK_SCORE < config.WEAK_RERANK_SCORE


def test_the_scales_are_genuinely_different():
    """If these ever converged, the two-scale handling would be pointless and
    someone would be tempted to delete it."""
    assert config.MIN_RERANK_SCORE < 0 < config.MIN_ANSWER_SCORE


def test_a_passage_below_the_floor_never_reaches_the_prompt():
    """Ranking it last is not enough -- everything in the context is material
    the model may answer from."""
    from src.answer import answer_question, usable

    results = [
        SearchResult(chunk=chunk("good", 0, "relevant text"), score=0.9, rerank_score=5.0),
        SearchResult(chunk=chunk("wrong", 0, "radio chirps"), score=0.9, rerank_score=-8.0),
    ]
    assert [r.chunk.paper_id for r in usable(results)] == ["good"]

    calls = []

    def spy(prompt, passed, backend):
        calls.append(prompt)
        return "answer"

    answer = answer_question("q", results, generator=spy)
    assert "radio chirps" not in calls[0]
    assert [r.chunk.paper_id for r in answer.results] == ["good"]


def test_the_sources_shown_are_the_ones_the_answer_used():
    """A citation list that includes a passage the model never saw is a lie."""
    from src.answer import answer_question

    results = [
        SearchResult(chunk=chunk("good"), score=0.9, rerank_score=5.0),
        SearchResult(chunk=chunk("wrong"), score=0.9, rerank_score=-8.0),
    ]
    answer = answer_question("q", results, generator=lambda p, r, b: "answer")
    assert len(answer.results) == 1


def test_relevance_prefers_the_rerank_score_when_present():
    assert SearchResult(chunk=chunk("a"), score=0.9).relevance == pytest.approx(0.9)
    assert SearchResult(chunk=chunk("a"), score=0.9,
                        rerank_score=-2.5).relevance == pytest.approx(-2.5)


def test_a_rerank_score_of_zero_is_not_mistaken_for_absent():
    """`if rerank_score:` would read 0.0 as missing and fall back to cosine."""
    result = SearchResult(chunk=chunk("a"), score=0.9, rerank_score=0.0)

    assert result.was_reranked
    assert result.relevance == pytest.approx(0.0)
    assert not np.isnan(result.relevance)
