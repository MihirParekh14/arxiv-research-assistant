"""Split papers into overlapping chunks for retrieval.

A whole paper cannot be embedded in one piece. The embedding model truncates
at 256 tokens, and even if it did not, a single vector for a 10,000-word paper
would average away everything specific about it -- which is exactly what search
needs to find.

So papers are cut into passages of a fixed word count, with an overlap between
neighbours so a sentence sitting on a boundary is still findable from either
side.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from src import config
from src.data import Paper


@dataclass(frozen=True)
class Chunk:
    """One searchable passage, with enough metadata to cite it later.

    ``paper_id`` and ``title`` travel with the text so a retrieved chunk can be
    shown to the user with its source. A retrieval system that cannot say where
    an answer came from is not much better than guessing.
    """

    chunk_id: str
    paper_id: str
    title: str
    text: str
    chunk_index: int
    word_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def split_words(
    text: str,
    chunk_words: int = config.CHUNK_WORDS,
    overlap_words: int = config.CHUNK_OVERLAP_WORDS,
    min_words: int = config.MIN_CHUNK_WORDS,
) -> list[str]:
    """Split ``text`` into overlapping passages, measured in words.

    The step between chunk starts is ``chunk_words - overlap_words``. If the
    overlap were allowed to equal or exceed the chunk size that step would be
    zero or negative and the loop would never advance, so it is rejected here
    rather than hanging later.

    A trailing fragment shorter than ``min_words`` is dropped. The content is
    not lost: the overlap means those words already appear at the end of the
    previous chunk.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= chunk_words:
        raise ValueError(
            f"overlap_words ({overlap_words}) must be smaller than "
            f"chunk_words ({chunk_words}), otherwise chunking never advances"
        )

    words = text.split()
    if len(words) < min_words:
        return []

    step = chunk_words - overlap_words
    chunks: list[str] = []

    for start in range(0, len(words), step):
        window = words[start : start + chunk_words]

        # Only the final window can fall short. Dropping it avoids indexing
        # stubs; earlier windows are always full length by construction.
        if len(window) < min_words:
            break

        chunks.append(" ".join(window))

        # The window already reached the end of the text, so another step
        # would only re-emit a suffix of what was just added.
        if start + chunk_words >= len(words):
            break

    return chunks


def chunk_paper(paper: Paper, **kwargs) -> list[Chunk]:
    """Cut one paper into :class:`Chunk` records."""
    passages = split_words(paper.text, **kwargs)
    return [
        Chunk(
            chunk_id=f"{paper.paper_id}::{index}",
            paper_id=paper.paper_id,
            title=paper.title,
            text=passage,
            chunk_index=index,
            word_count=len(passage.split()),
        )
        for index, passage in enumerate(passages)
    ]


def chunk_papers(papers: Iterable[Paper], **kwargs) -> Iterator[Chunk]:
    """Cut many papers, yielding chunks as they are produced."""
    for paper in papers:
        yield from chunk_paper(paper, **kwargs)


def save_chunks(chunks: Iterable[Chunk], path: str | Path) -> int:
    """Write chunks to a JSONL file, one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_chunks(path: str | Path) -> list[Chunk]:
    """Read chunks back, in the order they were written.

    Order is not a detail here. The embedding step writes one vector per chunk
    in this same order, and search looks a result up by its row number. If the
    two ever disagreed, every answer would cite the wrong paper while looking
    perfectly convincing. A list is returned rather than an iterator so that
    order is fixed and the file is read exactly once.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.ingest --limit 1000` first."
        )

    chunks = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                chunks.append(Chunk(**json.loads(line)))
    return chunks


def chunk_stats(chunks: Iterable[Chunk]) -> dict:
    """Summarise a set of chunks, for logging after an ingest run."""
    counts = [chunk.word_count for chunk in chunks]
    if not counts:
        return {"chunks": 0, "min_words": 0, "max_words": 0, "mean_words": 0.0}

    return {
        "chunks": len(counts),
        "min_words": min(counts),
        "max_words": max(counts),
        "mean_words": round(sum(counts) / len(counts), 1),
    }
