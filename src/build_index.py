"""Encode every chunk and build the search index.

Usage:
    python -m src.build_index
    python -m src.build_index --limit 2000     # a quick trial run
"""

from __future__ import annotations

import argparse
import time

from src import config
from src.chunking import load_chunks
from src.embed import embed_texts, load_encoder, save_vectors
from src.index import build_index, save_index, save_index_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the vector index.")
    parser.add_argument(
        "--chunks-file",
        default=str(config.CHUNKS_FILE),
        help="Chunks to encode.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only encode the first N chunks. Useful for a quick trial run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.EMBED_BATCH_SIZE,
        help="How many chunks to encode at once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.ensure_dirs()

    chunks = load_chunks(args.chunks_file)
    if args.limit is not None:
        chunks = chunks[: args.limit]

    if not chunks:
        raise SystemExit("No chunks to encode. Run `python -m src.ingest --limit 1000` first.")

    print(f"Loaded {len(chunks)} chunks from {args.chunks_file}")
    print(f"Encoding with {config.EMBEDDING_MODEL} ...")
    print("First run downloads the model (about 90 MB). After that it is cached.\n")

    encoder = load_encoder()

    started = time.time()
    vectors = embed_texts(
        [chunk.text for chunk in chunks],
        encoder=encoder,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - started

    save_vectors(vectors, config.VECTORS_FILE)

    index = build_index(vectors)
    save_index(index, config.INDEX_FILE)
    save_index_meta(
        config.INDEX_META_FILE,
        model_name=config.EMBEDDING_MODEL,
        dim=int(vectors.shape[1]),
        vector_count=len(chunks),
        chunks_file=str(args.chunks_file),
    )

    print("\nSummary")
    print(f"  chunks encoded     {len(chunks)}")
    print(f"  vector shape       {vectors.shape}")
    print(f"  time taken         {elapsed:.1f}s ({len(chunks) / max(elapsed, 0.001):.0f}/s)")
    print(f"  index size         {index.ntotal} vectors")
    print(f"  written to         {config.INDEX_DIR}")
    print('\nTry it:  python -m src.search "your question here"')


if __name__ == "__main__":
    main()
