"""Search the indexed papers from the command line.

Usage:
    python -m src.search "how do diffusion models handle image inpainting"
    python -m src.search "contrastive loss" --top-k 10
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass

from src import config
from src.chunking import Chunk, load_chunks
from src.embed import embed_texts, load_encoder
from src.index import check_index_matches, load_index, load_index_meta, search_index


@dataclass(frozen=True)
class SearchResult:
    """One retrieved passage, how it was found, and how well it actually fits.

    ``score`` is cosine similarity from the embedding index. ``rerank_score``
    is a cross-encoder logit, present only when reranking ran. They are on
    different scales and neither replaces the other: the first says why this
    passage surfaced at all, the second says whether it answers the question.
    """

    chunk: Chunk
    score: float
    rerank_score: float | None = None

    @property
    def relevance(self) -> float:
        """The score to rank and grade by -- the reranker's when it exists."""
        return self.score if self.rerank_score is None else self.rerank_score

    @property
    def was_reranked(self) -> bool:
        return self.rerank_score is not None


def cap_per_paper(
    results: list[SearchResult],
    top_k: int,
    max_per_paper: int = config.MAX_CHUNKS_PER_PAPER,
) -> list[SearchResult]:
    """Take the first ``top_k`` results, allowing at most ``max_per_paper`` each.

    Order is preserved: whatever ranking comes in is the ranking that comes
    out, minus the passages that exceed a paper's allowance.
    """
    if max_per_paper <= 0:
        raise ValueError("max_per_paper must be positive")

    kept: list[SearchResult] = []
    taken: dict[str, int] = {}

    for result in results:
        paper_id = result.chunk.paper_id
        if taken.get(paper_id, 0) >= max_per_paper:
            continue
        taken[paper_id] = taken.get(paper_id, 0) + 1
        kept.append(result)
        if len(kept) >= top_k:
            break

    return kept


def retrieve(
    query: str,
    chunks: list[Chunk],
    index,
    encoder,
    candidates: int,
) -> list[SearchResult]:
    """Pull a shortlist of candidates from the index, uncapped and unranked.

    The query goes through exactly the same encoder as the corpus did. That is
    not a convenience -- a query encoded by a different model would land
    somewhere unrelated in the vector space, and the results would be random
    while still looking plausible.
    """
    query_vector = embed_texts([query], encoder=encoder, show_progress=False)
    scores, positions = search_index(index, query_vector, top_k=candidates)

    results = []
    for score, position in zip(scores[0], positions[0], strict=True):
        if position < 0:  # FAISS pads with -1 when it holds fewer than asked for
            continue
        results.append(SearchResult(chunk=chunks[position], score=float(score)))
    return results


def search(
    query: str,
    chunks: list[Chunk],
    index,
    encoder,
    top_k: int = config.TOP_K,
    max_per_paper: int = config.MAX_CHUNKS_PER_PAPER,
    oversample: int = config.RETRIEVAL_OVERSAMPLE,
    reranker=None,
    candidates: int = config.RERANK_CANDIDATES,
):
    """Retrieve, optionally rerank, then cap, then take ``top_k``.

    **The cap** exists because a strongly matching paper otherwise takes the
    whole list. On the 5,000-paper index, one inpainting paper supplied four of
    five results. Capping costs a little raw similarity and buys the thing an
    answer actually needs: more than one source.

    **The order of the three steps matters.** Capping runs last, after
    reranking, because reranking changes which passages are best -- capping
    first would discard a passage on the strength of a score the reranker was
    about to overrule. Retrieval therefore pulls a wider shortlist than is
    wanted (``candidates`` when reranking, ``top_k * oversample`` otherwise),
    so that both later steps have spares to work with.
    """
    if max_per_paper <= 0:
        raise ValueError("max_per_paper must be positive")

    pool_size = candidates if reranker is not None else top_k * max(oversample, 1)
    pool = retrieve(query, chunks, index, encoder, candidates=pool_size)

    if reranker is not None:
        from src.rerank import rerank  # noqa: PLC0415  (avoids a circular import)

        pool = rerank(query, pool, reranker)

    return cap_per_paper(pool, top_k=top_k, max_per_paper=max_per_paper)


def _search_uncapped(query: str, chunks: list[Chunk], index, encoder, top_k: int):
    """Retrieval without the per-paper cap. Kept for comparison in tests."""
    query_vector = embed_texts([query], encoder=encoder, show_progress=False)
    scores, positions = search_index(index, query_vector, top_k=top_k)

    results = []
    for score, position in zip(scores[0], positions[0], strict=True):
        if position < 0:  # FAISS pads with -1 when it has fewer vectors than asked for
            continue
        results.append(SearchResult(chunk=chunks[position], score=float(score)))
    return results


def _score_label(result: SearchResult) -> str:
    """Both numbers when reranking ran, since they measure different things."""
    if result.was_reranked:
        return f"rerank {result.rerank_score:+.2f}  retrieval {result.score:.3f}"
    return f"score {result.score:.3f}"


def format_result(result: SearchResult, rank: int, width: int = 88) -> str:
    """Render one result for the terminal."""
    body = textwrap.fill(result.chunk.text[:400], width=width, initial_indent="    ",
                         subsequent_indent="    ")
    return (
        f"\n[{rank}]  {_score_label(result)}   {result.chunk.title}\n"
        f"     {result.chunk.paper_id}  chunk {result.chunk.chunk_index}\n"
        f"{body} ..."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the indexed arXiv papers.")
    parser.add_argument("query", help="What you want to find, in plain English.")
    parser.add_argument("--top-k", type=int, default=config.TOP_K, help="How many results.")
    parser.add_argument("--max-per-paper", type=int, default=config.MAX_CHUNKS_PER_PAPER,
                        help="At most this many passages from any one paper.")
    parser.add_argument("--rerank", dest="rerank", action="store_true", default=None,
                        help="Re-score the shortlist with a cross-encoder (default on).")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false",
                        help="Skip reranking, to compare against the embedding search alone.")
    parser.add_argument("--candidates", type=int, default=config.RERANK_CANDIDATES,
                        help="How many passages to shortlist before reranking.")
    parser.add_argument("--chunks-file", default=str(config.CHUNKS_FILE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks = load_chunks(args.chunks_file)
    meta = load_index_meta(config.INDEX_META_FILE)
    check_index_matches(meta, chunk_count=len(chunks), model_name=config.EMBEDDING_MODEL)

    index = load_index(config.INDEX_FILE)
    encoder = load_encoder()

    use_rerank = config.RERANK_ENABLED if args.rerank is None else args.rerank
    reranker = None
    if use_rerank:
        from src.rerank import load_reranker  # noqa: PLC0415

        print(f"Reranking with {config.RERANK_MODEL} ...")
        reranker = load_reranker()

    results = search(args.query, chunks, index, encoder, top_k=args.top_k,
                     max_per_paper=args.max_per_paper, reranker=reranker,
                     candidates=args.candidates)

    print(f'\nQuery: "{args.query}"')
    print(f"Searched {index.ntotal} passages from "
          f"{len({chunk.paper_id for chunk in chunks})} papers.")

    if not results:
        print("\nNo results.")
        return

    for rank, result in enumerate(results, start=1):
        print(format_result(result, rank))
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
