# arXiv Research Assistant

A retrieval-augmented question answering system over arXiv papers.

Ask a question in plain English, and the system searches a corpus of papers,
retrieves the passages that actually bear on the question, and answers from
those passages — with citations back to the papers it used.

This is a rebuild of a graduate deep learning project. The original had a
classifier and a summarizer but no retrieval, so it could only work on a paper
you handed it directly. It also had an evaluation bug: training and evaluation
ran on overlapping data, which inflated every score. Both are addressed here.

---

## Status

This is being built in stages. Each stage is tested before the next begins.

| Stage | What it does | Status |
|---|---|---|
| 1. Ingestion | Download papers, clean, chunk with overlap | **Done** |
| 2. Embeddings | Encode chunks, build a FAISS index | Next |
| 3. Retrieval + answers | Search, assemble a prompt, generate with citations | Planned |
| 4. Classifier | Multi-label subject tagging (fine-tuned RoBERTa) | Planned |
| 5. Summarizer | Abstractive summaries (google/pegasus-arxiv) | Planned |
| 6. Evaluation | Held-out splits, retrieval and answer metrics | Planned |

---

## Quick start

```bash
git clone https://github.com/MihirParekh14/arxiv-research-assistant.git
cd arxiv-research-assistant

py -3.11 -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS and Linux

pip install -r requirements-dev.txt
```

Run the tests:

```bash
pytest
```

Download and chunk 1,000 papers:

```bash
python -m src.ingest --limit 1000
```

Re-chunk what you already downloaded, without downloading again:

```bash
python -m src.ingest --reuse-papers
```

Output lands in `data/`, which is gitignored. Nothing large is committed.

A run of 1,000 papers takes about 20 seconds and produces roughly 49,000 chunks.

### Known warning on Windows

After the summary prints, Windows may show this:

```
[WinError 10038] An operation was attempted on something that is not a socket
Retrying in 1s [Retry 1/5].
```

It is harmless and the data is already written — note that it appears *after*
both output files are saved. The cause is stopping the stream early: once the
limit is reached the HTTP connection is abandoned half-open, Windows closes the
socket, and `huggingface_hub`'s retry thread then finds it gone. Linux and macOS
tear the same connection down without complaint.

---

## How the ingestion works

```
arXiv dataset  ──►  normalise  ──►  clean  ──►  chunk  ──►  chunks.jsonl
   (streamed)        one shape     whitespace   overlap
```

**Streaming, not downloading.** The source dataset holds 138,000 papers. Pulling
all of it to try 1,000 would waste several gigabytes, so papers are streamed and
the loop stops once it has enough.

**One record shape.** Different arXiv dumps name their fields differently —
`id` or `paper_id`, `abstract` or `summary`, `markdown` or `full_text`. Each
field is resolved against a list of candidate names and normalised into a single
`Paper` record. Swapping the data source later touches one module instead of
five.

**Missing fields do not crash an ingest.** A paper with no recorded author is
still worth indexing. Absent fields become empty strings; only papers with no
usable text at all are dropped.

**Markdown syntax is stripped, heading text is not.** The source is markdown, so
a raw passage arrives looking like this:

```
# Image Completion via Dual-Path Cooperative Filtering ###### Abstract Given ...
```

`#` and `######` are not words. They spend tokens inside the embedding model's
256-token budget and no search query will ever match them. Figure paths inside
image syntax are the same. Both are removed.

Heading *text* is kept deliberately — knowing a passage came from
"Introduction" rather than "Results" is real context for retrieval. Emphasis
markers are also left alone, because `x_1` and `\sigma_t` are variable names in
these papers and stripping underscores would corrupt the maths.

Order matters: markdown is stripped while line breaks still mark where headings
begin, and only then is whitespace collapsed.

---

## Why the chunks are 150 words

Chunk size is decided by the embedding model, not by preference.

`all-MiniLM-L6-v2` truncates its input at **256 tokens**. Anything past that is
silently discarded — so an oversized chunk would be indexed with its tail
missing, and that tail could never be retrieved. English runs roughly 1.3 tokens
per word, which puts 150 words near 195 tokens: inside the limit, with room for
unusually long words.

**Overlap is 30 words.** Without overlap, a sentence that lands on a boundary is
cut in half and neither half retrieves well. With too much overlap the index
fills with near-duplicates and the same passage crowds out other results. A
fifth of the chunk is a reasonable starting point, and it is a parameter worth
tuning once the evaluation in stage 6 exists to measure it.

**Fragments under 25 words are dropped.** A leftover stub from a section heading
adds noise to search without adding meaning. Nothing is lost: the overlap means
those words already appear at the end of the previous chunk.

All three values live in `src/config.py`.

---

## Testing

```bash
pytest --cov=src --cov-report=term-missing
ruff check src tests
```

42 tests. CI runs both on Python 3.10, 3.11 and 3.12 on every push.

The tests never touch the network and never need a dataset or an API key. Every
test builds its own records, so the suite runs anywhere in under a second.

The chunking tests get the most attention, because chunking is where retrieval
systems quietly break. An off-by-one in the step size either drops text between
chunks or repeats it forever, and neither failure is visible from the output —
you just get worse answers. So the tests pin down the boundaries directly: that
consecutive chunks share exactly the configured overlap, that chunk starts
advance by exactly the step size, that every word of the input appears in at
least one chunk, and that an overlap greater than or equal to the chunk size is
rejected rather than looping forever.

---

## Project layout

```
src/
  config.py      all tunable values in one place
  data.py        download, normalise, save and load papers
  chunking.py    split papers into overlapping passages
  ingest.py      command line entry point
tests/           pytest suite, no network required
notebooks/       exploration
data/            downloaded and generated files (gitignored)
```

---

## Licence

MIT
