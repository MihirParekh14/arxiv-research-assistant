"""Search the indexed papers from the command line.

Usage:
    python -m src.search "how do diffusion models handle image inpainting"
    python -m src.search "contrastive loss" --top-k 10
"""

from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass

from src import config
from src.chunking import Chunk, load_chunks
from src.embed import embed_texts, load_encoder
from src.index import check_index_matches, load_index, load_index_meta, search_index


@dataclass(frozen=True)
class SearchResult:
    """One retrieved passage and how close it was to the query."""

    chunk: Chunk
    score: float


def search(query: str, chunks: list[Chunk], index, encoder, top_k: int = config.TOP_K):
    """Return the ``top_k`` chunks closest in meaning to ``query``.

    The query goes through exactly the same encoder as the corpus did. That is
    not a convenience -- a query encoded by a different model would land
    somewhere unrelated in the vector space, and the results would be random
    while still looking plausible.
    """
    query_vector = embed_texts([query], encoder=encoder, show_progress=False)
    scores, positions = search_index(index, query_vector, top_k=top_k)

    results = []
    for score, position in zip(scores[0], positions[0], strict=True):
        if position < 0:  # FAISS pads with -1 when it has fewer vectors than asked for
            continue
        results.append(SearchResult(chunk=chunks[position], score=float(score)))
    return results


def format_result(result: SearchResult, rank: int, width: int = 88) -> str:
    """Render one result for the terminal."""
    body = textwrap.fill(result.chunk.text[:400], width=width, initial_indent="    ",
                         subsequent_indent="    ")
    return (
        f"\n[{rank}]  score {result.score:.3f}   {result.chunk.title}\n"
        f"     {result.chunk.paper_id}  chunk {result.chunk.chunk_index}\n"
        f"{body} ..."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the indexed arXiv papers.")
    parser.add_argument("query", help="What you want to find, in plain English.")
    parser.add_argument("--top-k", type=int, default=config.TOP_K, help="How many results.")
    parser.add_argument("--chunks-file", default=str(config.CHUNKS_FILE))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    chunks = load_chunks(args.chunks_file)
    meta = load_index_meta(config.INDEX_META_FILE)
    check_index_matches(meta, chunk_count=len(chunks), model_name=config.EMBEDDING_MODEL)

    index = load_index(config.INDEX_FILE)
    encoder = load_encoder()

    results = search(args.query, chunks, index, encoder, top_k=args.top_k)

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
    main()
