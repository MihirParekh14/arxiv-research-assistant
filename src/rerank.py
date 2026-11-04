"""Re-score a shortlist by reading the question and each passage together.

The embedding search is a bi-encoder: it turns the question into a vector and
every passage into a vector, separately, and compares the two. Nothing ever
looks at a question and a passage side by side. That is what makes it fast
enough to scan a quarter of a million passages, and it is also why it confuses
LoRA the fine-tuning method with LoRa the radio protocol -- at the vector level
those are two similar strings in similar technical company.

A cross-encoder takes the pair as one input and returns a single relevance
score. It cannot be precomputed and it cannot be indexed, so it costs one
forward pass per candidate. Run over 253,000 passages that would be hopeless.
Run over the 30 the bi-encoder already shortlisted, it costs a second or two
and fixes the cases the bi-encoder cannot see.

So the two are not alternatives. Retrieval finds candidates cheaply; reranking
orders them properly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from src import config
from src.search import SearchResult


def load_reranker(model_name: str = config.RERANK_MODEL):
    """Load the cross-encoder.

    Imported inside the function so this module stays importable, and the test
    suite runnable, without the model or the network.
    """
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    return CrossEncoder(model_name)


def attach_scores(results: Sequence[SearchResult], scores: Sequence[float]) -> list[SearchResult]:
    """Attach rerank scores to results and sort by them, best first.

    Kept separate from the model call so the ordering logic -- the part that
    can silently pair a score with the wrong passage -- is testable without
    loading anything.
    """
    if len(results) != len(scores):
        raise ValueError(
            f"got {len(scores)} scores for {len(results)} results; "
            f"they must line up one to one"
        )

    scored = [
        replace(result, rerank_score=float(score))
        for result, score in zip(results, scores, strict=True)
    ]
    return sorted(scored, key=lambda result: result.rerank_score, reverse=True)


def rerank(
    query: str,
    results: Sequence[SearchResult],
    reranker,
    batch_size: int = config.RERANK_BATCH_SIZE,
) -> list[SearchResult]:
    """Re-score and reorder ``results`` for ``query``.

    The original retrieval score is kept on each result rather than overwritten.
    Both numbers are worth having: one says how the passage was found, the
    other says how well it actually answers the question, and comparing them is
    the only way to see whether reranking is earning its cost.
    """
    if not results:
        return []

    pairs = [(query, result.chunk.text) for result in results]
    scores = reranker.predict(pairs, batch_size=batch_size)
    return attach_scores(results, list(scores))
