"""Download arXiv papers and write them to disk in one consistent shape.

The raw dataset is streamed rather than downloaded in full. Streaming means
pulling 1,000 papers costs a few seconds and a few megabytes instead of a
multi-gigabyte download, which keeps the first run of this project fast.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from src import config


@dataclass(frozen=True)
class Paper:
    """One paper, in the shape the rest of the pipeline expects.

    Normalising into this record early means a change of data source later
    touches only this module.
    """

    paper_id: str
    title: str
    abstract: str
    text: str
    authors: str
    published: str
    url: str

    def word_count(self) -> int:
        return len(self.text.split())


# Fields differ between arXiv dumps, so each one is looked up by a list of
# candidate names rather than a single hard-coded key.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "paper_id": ("id", "paper_id", "arxiv_id"),
    "title": ("title",),
    "abstract": ("abstract", "summary"),
    "text": ("markdown", "text", "full_text", "body"),
    "authors": ("authors", "author"),
    "published": ("published_date", "published", "date", "update_date"),
    "url": ("link", "url", "abs_url"),
}


def _first_present(record: dict, keys: Iterable[str]) -> str:
    """Return the first non-empty value among ``keys``, or an empty string."""
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


# Markdown syntax that carries no meaning once the text is a flat passage.
# Applied while line breaks still exist, because headings and rules are
# anchored to the start of a line.
_HORIZONTAL_RULE = re.compile(r"^\s{0,3}(?:-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)
_HEADING_MARKER = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def strip_markdown(text: str) -> str:
    """Remove markdown syntax that is not part of the text's meaning.

    The source papers are markdown, so a passage can arrive carrying ``######``
    before a heading or a figure path inside image syntax. Those characters are
    not words. They spend tokens inside the embedding model's 256-token budget
    and add nothing a search could match on.

    Heading *text* is kept on purpose -- knowing a passage came from
    "Introduction" rather than "Results" is useful context. Only the markers go.

    Emphasis markers (``*`` and ``_``) are deliberately left alone: ``x_1`` and
    ``\\sigma_t`` are variable names in these papers, and stripping underscores
    would corrupt them.
    """
    if not text:
        return ""
    text = _HORIZONTAL_RULE.sub(" ", text)
    text = _HEADING_MARKER.sub("", text)
    text = _IMAGE.sub(" ", text)
    return _LINK.sub(r"\1", text)


def clean_text(text: str) -> str:
    """Strip markdown syntax, then collapse whitespace.

    Markdown from the source carries hard line wraps, repeated blank lines and
    tabs. Left alone these inflate the word count and split chunks in odd
    places, so they are flattened to single spaces before anything downstream
    counts words.

    Order matters: markdown is stripped first, while line breaks still mark
    where headings and horizontal rules begin. After the collapse that
    information is gone.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", strip_markdown(text)).strip()


def normalize_record(record: dict) -> Paper:
    """Turn one raw dataset row into a :class:`Paper`.

    Missing fields become empty strings rather than raising. A paper with no
    recorded author is still worth indexing; a crash halfway through an ingest
    is not.
    """
    values = {
        field: clean_text(_first_present(record, aliases))
        for field, aliases in _FIELD_ALIASES.items()
    }

    # Fall back to the abstract when full text is absent, so the paper still
    # carries something searchable instead of being silently dropped.
    if not values["text"]:
        values["text"] = values["abstract"]

    return Paper(**values)


def is_usable(paper: Paper) -> bool:
    """Reject papers too short to produce even one meaningful chunk."""
    return bool(paper.paper_id) and paper.word_count() >= config.MIN_CHUNK_WORDS


def fetch_papers(limit: int, streaming: bool = True) -> Iterator[Paper]:
    """Stream ``limit`` usable papers from the configured dataset.

    ``datasets`` is imported inside the function so the rest of this module --
    and its tests -- stay importable without the heavy dependency installed.
    """
    from datasets import load_dataset  # noqa: PLC0415

    dataset = load_dataset(config.HF_DATASET, split=config.HF_SPLIT, streaming=streaming)

    kept = 0
    for record in dataset:
        paper = normalize_record(dict(record))
        if not is_usable(paper):
            continue
        yield paper
        kept += 1
        if kept >= limit:
            return


def save_papers(papers: Iterable[Paper], path: str | Path) -> int:
    """Write papers to a JSONL file, one JSON object per line.

    JSONL is used instead of a single JSON array so the file can be read back
    line by line without holding every paper in memory at once.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(asdict(paper), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_papers(path: str | Path) -> Iterator[Paper]:
    """Read papers back from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.ingest --limit 1000` first."
        )

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield Paper(**json.loads(line))
