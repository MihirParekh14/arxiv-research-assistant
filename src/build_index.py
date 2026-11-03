"""Encode every chunk and build the search index.

Encoding runs in blocks that are written to disk as they finish, so memory
stays flat on a corpus of any size and an interrupted run picks up where it
stopped.

Usage:
    python -m src.build_index
    python -m src.build_index --limit 5000     # a quick trial run
    python -m src.build_index --fresh          # ignore finished blocks, start over
"""

from __future__ import annotations

import argparse
import sys
import time

from src import config
from src.chunking import load_chunks
from src.embed import embed_in_blocks, load_blocks, load_encoder, save_vectors
from src.index import build_index, save_index, save_index_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the vector index.")
    parser.add_argument("--chunks-file", default=str(config.CHUNKS_FILE),
                        help="Chunks to encode.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only encode the first N chunks. Useful for a trial run.")
    parser.add_argument("--batch-size", type=int, default=config.EMBED_BATCH_SIZE,
                        help="How many chunks the encoder processes at once.")
    parser.add_argument("--block-size", type=int, default=config.EMBED_BLOCK_SIZE,
                        help="How many chunks are encoded before writing to disk.")
    parser.add_argument("--fresh", action="store_true",
                        help="Re-encode everything instead of resuming from finished blocks.")
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def main() -> None:
    args = parse_args()
    config.ensure_dirs()

    chunks = load_chunks(args.chunks_file)
    if args.limit is not None:
        chunks = chunks[: args.limit]

    if not chunks:
        raise SystemExit("No chunks to encode. Run `python -m src.ingest --limit 1000` first.")

    total = len(chunks)
    blocks = (total + args.block_size - 1) // args.block_size

    print(f"Loaded {total} chunks from {args.chunks_file}")
    print(f"Encoding with {config.EMBEDDING_MODEL} in {blocks} blocks "
          f"of up to {args.block_size}.")
    print("Each finished block is written to disk, so stopping this is safe "
          "-- it resumes from the last one.\n")

    encoder = load_encoder()

    started = time.time()
    state = {"done": 0, "encoded": 0}

    def report(start: int, rows: int, skipped: bool) -> None:
        state["done"] += 1
        if skipped:
            print(f"  block {state['done']:>4}/{blocks}  rows {start:>8}  already done")
            return

        state["encoded"] += rows
        elapsed = time.time() - started
        rate = state["encoded"] / max(elapsed, 0.001)
        left = total - (start + rows)
        print(f"  block {state['done']:>4}/{blocks}  rows {start:>8}  "
              f"{rate:.0f}/s  about {format_duration(left / max(rate, 0.001))} left")

    embed_in_blocks(
        [chunk.text for chunk in chunks],
        encoder=encoder,
        blocks_dir=config.BLOCKS_DIR,
        block_size=args.block_size,
        batch_size=args.batch_size,
        resume=not args.fresh,
        on_block=report,
    )
    elapsed = time.time() - started

    print("\nJoining blocks and building the index ...")
    vectors = load_blocks(config.BLOCKS_DIR, total=total, block_size=args.block_size)
    save_vectors(vectors, config.VECTORS_FILE)

    index = build_index(vectors)
    save_index(index, config.INDEX_FILE)
    save_index_meta(
        config.INDEX_META_FILE,
        model_name=config.EMBEDDING_MODEL,
        dim=int(vectors.shape[1]),
        vector_count=total,
        chunks_file=str(args.chunks_file),
    )

    print("\nSummary")
    print(f"  chunks encoded     {total}")
    print(f"  vector shape       {vectors.shape}")
    print(f"  time this run      {format_duration(elapsed)}")
    print(f"  index size         {index.ntotal} vectors")
    print(f"  written to         {config.INDEX_DIR}")
    print('\nTry it:  python -m src.search "your question here"')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Finished blocks are already on disk, so stopping here loses at most
        # the block in progress. Say so instead of printing a traceback.
        print("\n\nStopped. Finished blocks are saved.")
        print("Run `python -m src.build_index` again to carry on from the last one.")
        sys.exit(130)
