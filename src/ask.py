"""Ask a question and get an answer with citations.

Usage:
    python -m src.ask "how do diffusion models handle image inpainting"
    python -m src.ask "what is speculative decoding" --backend ollama
    python -m src.ask "how does LoRA work" --backend anthropic --top-k 8
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from src import config
from src.answer import BACKENDS, answer_question, format_sources
from src.chunking import load_chunks
from src.embed import load_encoder
from src.index import check_index_matches, load_index, load_index_meta
from src.search import search

CONFIDENCE_NOTE = {
    "ok": "the retrieved passages match the question well",
    "weak": "the retrieved passages only loosely match the question",
    "none": "nothing in the corpus matched closely enough to answer from",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the indexed arXiv papers a question.")
    parser.add_argument("question", help="Your question, in plain English.")
    parser.add_argument("--backend", default=config.ANSWER_BACKEND, choices=BACKENDS,
                        help="Who writes the answer. 'extractive' needs no model.")
    parser.add_argument("--top-k", type=int, default=config.TOP_K,
                        help="How many passages to answer from.")
    parser.add_argument("--max-per-paper", type=int, default=config.MAX_CHUNKS_PER_PAPER,
                        help="At most this many passages from any one paper.")
    parser.add_argument("--rerank", dest="rerank", action="store_true", default=None,
                        help="Re-score the shortlist with a cross-encoder (default on).")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false",
                        help="Skip reranking, to compare against the embedding search alone.")
    parser.add_argument("--candidates", type=int, default=config.RERANK_CANDIDATES,
                        help="How many passages to shortlist before reranking.")
    parser.add_argument("--chunks-file", default=str(config.CHUNKS_FILE))
    parser.add_argument("--show-prompt", action="store_true",
                        help="Print the prompt sent to the model, then the answer.")
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

    results = search(args.question, chunks, index, encoder,
                     top_k=args.top_k, max_per_paper=args.max_per_paper,
                     reranker=reranker, candidates=args.candidates)

    if args.show_prompt:
        from src.answer import build_prompt  # noqa: PLC0415

        print("\n--- prompt ---")
        print(build_prompt(args.question, results))
        print("--- end prompt ---")

    answer = answer_question(args.question, results, backend=args.backend)

    print(f'\nQuestion: "{args.question}"')
    print(f"Searched {index.ntotal} passages from "
          f"{len({chunk.paper_id for chunk in chunks})} papers.")
    print(f"Confidence: {answer.confidence} "
          f"(top {answer.scale} score {answer.top_score:+.3f}) "
          f"-- {CONFIDENCE_NOTE[answer.confidence]}")
    print(f"Answer by: {answer.backend}\n")

    print(textwrap.fill(answer.text, width=88) if answer.confidence == "none"
          else answer.text)

    if answer.results:
        papers = len({r.chunk.paper_id for r in answer.results})
        # On a refusal these passages support nothing, so they are not sources.
        # Showing them anyway is useful -- it lets the reader see how close the
        # corpus got -- but calling them sources would be a lie.
        heading = (f"Closest matches, not used ({papers} papers)"
                   if answer.confidence == "none" else f"Sources ({papers} papers)")
        print(f"\n{heading}")
        print(format_sources(answer.results))
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
