"""Download arXiv papers, chunk them, and write both to disk.

Usage:
    python -m src.ingest --limit 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src import config
from src.chunking import chunk_papers, chunk_stats, save_chunks
from src.data import fetch_papers, load_papers, save_papers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest arXiv papers for retrieval.")
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="How many papers to download. Start small (1000) to check it works.",
    )
    parser.add_argument(
        "--papers-file",
        default=str(config.RAW_DIR / "papers.jsonl"),
        help="Where to write the downloaded papers.",
    )
    parser.add_argument(
        "--chunks-file",
        default=str(config.PROCESSED_DIR / "chunks.jsonl"),
        help="Where to write the chunks.",
    )
    parser.add_argument(
        "--reuse-papers",
        action="store_true",
        help="Skip the download and re-chunk papers already on disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dirs()

    papers_path = Path(args.papers_file)
    chunks_path = Path(args.chunks_file)

    # --- Step 1: get the papers -------------------------------------------

    if args.reuse_papers and papers_path.exists():
        print(f"Reusing papers already at {papers_path}")
    else:
        print(f"Downloading up to {args.limit} papers from {config.HF_DATASET} ...")
        started = time.time()
        written = save_papers(fetch_papers(limit=args.limit), papers_path)
        elapsed = time.time() - started
        print(f"Saved {written} papers to {papers_path} in {elapsed:.1f}s")

    # --- Step 2: chunk them -----------------------------------------------

    print(
        f"\nChunking at {config.CHUNK_WORDS} words "
        f"with {config.CHUNK_OVERLAP_WORDS} words of overlap ..."
    )

    chunks = list(chunk_papers(load_papers(papers_path)))
    save_chunks(chunks, chunks_path)

    # --- Step 3: report ----------------------------------------------------

    stats = chunk_stats(chunks)
    papers_seen = len({chunk.paper_id for chunk in chunks})

    print(f"Saved {stats['chunks']} chunks to {chunks_path}\n")
    print("Summary")
    print(f"  papers chunked     {papers_seen}")
    print(f"  chunks produced    {stats['chunks']}")
    print(f"  chunks per paper   {stats['chunks'] / max(papers_seen, 1):.1f}")
    print(f"  words per chunk    min {stats['min_words']}, "
          f"mean {stats['mean_words']}, max {stats['max_words']}")

    if stats["max_words"] > config.CHUNK_WORDS:
        print("\nWARNING: a chunk is longer than the configured size.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped before writing. Re-run to start again.")
        sys.exit(130)
