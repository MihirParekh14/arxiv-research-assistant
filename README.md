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
| 2. Search | Encode chunks, build a FAISS index, retrieve by meaning | **Done** |
| 3. Answers | Assemble a prompt from retrieved passages, generate with citations | **Done** |
| 3.5 Reranking | Cross-encoder rerank over the top 20 (see known limitation) | Next |
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

Encode every chunk and build the search index:

```bash
python -m src.build_index
```

Encoding writes a block to disk every 8,192 chunks, so stopping it is safe --
re-running the same command continues from the last finished block. Add
`--fresh` to re-encode everything, or `--limit N` for a quick trial run.

Search:

```bash
python -m src.search "how do diffusion models handle image inpainting"
python -m src.search "contrastive loss" --top-k 10
```

Ask a question and get an answer with citations:

```bash
python -m src.ask "how do diffusion models handle image inpainting"
python -m src.ask "what is speculative decoding" --backend ollama
python -m src.ask "how does LoRA work" --backend anthropic
python -m src.ask "anything" --show-prompt     # see exactly what the model is sent
```

The default backend is `extractive`: it returns the retrieved passages
themselves and calls no model, so the project runs with no API key and the
retrieval half can be judged on its own. `anthropic` needs `ANTHROPIC_API_KEY`
in a `.env` file; `ollama` needs `ollama serve` running locally.

Output lands in `data/`, which is gitignored. Nothing large is committed.

A run of 1,000 papers takes about 20 seconds and produces roughly 49,000
chunks. Encoding them takes about 40 minutes on a laptop CPU.

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
   (streamed)        one shape     whitespace   overlap         │
                                                                ▼
          results  ◄──  FAISS  ◄──  index  ◄──  encode  ◄──  150-word
         (ranked)       search                  (MiniLM)      passages
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

## How the search works

Each chunk is encoded into 384 numbers. Chunks about the same thing land near
each other in that space, even when they share no words — "image inpainting"
and "filling masked regions" end up close because the model learned they mean
the same thing. A query is encoded the same way, and search is then just a
question of which stored vectors point most nearly in the same direction.

**Vectors are normalised to length 1.** For unit vectors the dot product is
exactly the cosine of the angle between them, so FAISS's inner-product search
gives cosine similarity with no extra work. Skipping this step would let long
vectors win on size rather than relevance, and vector length tracks things like
passage length, not meaning.

**The index is flat, not approximate.** `IndexFlatIP` compares the query
against all 253,000 vectors — no approximation, no training step, exact results.
Approximate indexes like IVF and HNSW trade recall for speed and earn their
keep at millions of vectors. At this size an exact scan is about 19 million
multiply-adds, which takes a few milliseconds, so approximation would cost
accuracy and buy nothing.

**Encoding runs in blocks, and resumes.** The first version handed every
passage to the encoder in a single call. That was fine at 49,000 and fell over
at 253,000: holding all the inputs and outputs at once pushed the machine into
swap, and throughput dropped from 3 seconds per batch to 213 — a factor of
sixty, not a slowdown you can wait out. A job that was three hours of real work
was reporting eight days remaining.

So each block of 8,192 passages is encoded, written to disk, and released.
Memory stays flat whatever the corpus size, and an interrupted run continues
from the last finished block instead of starting again. Blocks are written to a
temporary name and renamed, so a run killed mid-write leaves nothing half
finished, and a block whose length does not match is re-encoded rather than
trusted. A manifest records the corpus size, block size and model, and blocks
from a different corpus are discarded rather than stitched onto the new one.

**Row order is the contract.** Line *i* of `chunks.jsonl`, row *i* of
`vectors.npy` and position *i* in the index are the same passage. That is the
only link between a vector and the text it came from, and it fails silently:
if they ever drifted apart, every answer would cite a real passage from the
wrong paper and look entirely convincing. So the index records the model and
corpus size it was built from, and search refuses to run against a mismatch
instead of answering from stale rows.

---

## How the answering works

Retrieval finds passages. Answering decides whether they are good enough to
answer from, assembles them into a prompt, and asks a model for an answer that
cites them. Three decisions shape it, and two of them came from measurements
rather than from judgement.

### Growing the corpus, and what it showed

The corpus went from 1,000 papers to 5,000 — 49,334 passages to 253,405. The
same two queries were run before and after.

| Query | | 1,000 papers | 5,000 papers |
|---|---|---|---|
| image inpainting | top score | 0.627 | **0.729** |
| | spread | 0.018 | 0.043 |
| transformer attention | top score | 0.657 | 0.677 |
| | spread | 0.086 | **0.025** |

The inpainting query improved exactly as expected: more papers, better
coverage, and the top result became a paper titled *Gradpaint: Gradient-Guided
Inpainting with Diffusion Models*. A false match from the smaller corpus — a
depth-estimation paper that matched on "regions with missing values" —
disappeared.

The spread is the interesting column, because it moved in opposite directions.

### The confidence signal, and the idea that did not survive

The first plan was to treat a narrow spread between the top and bottom scores
as a sign that the corpus had nothing to offer, and refuse to answer. The table
above kills that idea. Inpainting had almost no good matches, so everything sat
low together and the spread was narrow; adding papers lifted the top and the
spread widened. Transformer attention had one excellent match and then a cliff;
adding papers lifted the *tail* until all five were strong, and the spread
narrowed.

A narrow spread means either *nothing matches* or *everything matches well*,
and nothing distinguishes the two. The top score separated good answers from
bad ones in all four runs, so it is what the confidence gate uses: below
`MIN_ANSWER_SCORE` the system refuses and never builds a prompt, between that
and `WEAK_ANSWER_SCORE` it answers with a warning. Both numbers are provisional
starting points; stage 6 measures them against a held-out question set.

### Capping how much one paper can contribute

On the larger index, the inpainting query returned **four of its five results
from a single paper**. Better retrieval made this worse, not better: once a
strongly matching paper exists, its chunks crowd out everything else.

For search that is only untidy. For a generated answer it is a defect, because
those passages are the entire material the answer is written from — an answer
drawn from one paper cites one paper and sounds exactly as confident as one
drawn from five. So no paper may supply more than `MAX_CHUNKS_PER_PAPER`
passages, and the index is asked for five times as many results as are wanted
so that capping replaces results rather than losing them.

### A known limitation: the query is one vector

Two queries about the same thing, run against the same index, expose the
ceiling of this design.

**Query A — "how does LoRA work"** returned three passages genuinely about
Low-Rank Adaptation, and two about **LoRa**, the long-range radio protocol —
chirp demodulation, spreading factors, LoRaWAN jamming. Two unrelated
technologies, nearly the same spelling, and a bi-encoder cannot separate them.

**Query B — "how does low-rank adaptation work for fine-tuning language
models"** removed the radio papers and made the result set worse: only one of
five was about LoRA. The rest were about fine-tuning in general — annotating
data for rankers, Arabic machine translation, multilingual vocabularies. The
added words are common in this corpus, so averaging them into one vector
diluted the one distinctive term until the query pointed at generic NLP
adaptation.

| Query | Passages about LoRA | Top score |
|---|---|---|
| "how does LoRA work" | 3 of 5 | 0.645 |
| "how does low-rank adaptation work for fine-tuning language models" | 1 of 5 | **0.680** |

Note which one scored higher. Similarity to the query vector is not answer
quality, and here the two point in opposite directions. The confidence gate is
still worth having — it catches the case where nothing matches at all — but it
grades the top match, not the correctness of the list.

Neither failure is fixed by a larger corpus; more papers means more radio
papers. Both have the same cause: a bi-encoder embeds the query and the passage
separately and never compares them directly.

The standard fix is a **cross-encoder reranker** — retrieve twenty passages
with the fast vector search, then score each one with a model that reads the
question and the passage together. Reading both at once is what allows
"spreading factor" to be recognised as irrelevant to a fine-tuning question.
That is the next thing this project needs, ahead of anything in stages 4 or 5.

Also worth recording: the best explanation of LoRA in the whole corpus came
from a passage inside *"Hate speech detection in algerian dialect using deep
learning"*, a paper whose title gives no sign of it. Indexing whole papers
would have missed it; indexing passages found it.

### Refusing is a feature

Below the score floor, no prompt is built and no model is called. The prompt
itself also tells the model to answer only from the passages and to say so when
they do not cover the question.

A system that always produces fluent text is worse than one that sometimes
declines, because a wrong answer with a citation attached is harder to catch
than no answer at all. Both paths are tested: that a weak result set returns a
refusal, and that it reaches no generator.

---

## Testing

```bash
pytest --cov=src --cov-report=term-missing
ruff check src tests
```

117 tests. CI runs both on Python 3.10, 3.11 and 3.12 on every push.

The tests never touch the network, never download a model and never need an API
key. A stand-in encoder returns fixed vectors, so the whole suite runs anywhere
in well under a second.

Most of the attention goes to the two places a retrieval system breaks quietly.

**Chunking.** An off-by-one in the step size either drops text between chunks
or repeats it forever, and neither failure is visible from the output — you
just get worse answers. The tests pin the boundaries down directly: consecutive
chunks share exactly the configured overlap, chunk starts advance by exactly
the step size, every word of the input appears in at least one chunk, and an
overlap greater than or equal to the chunk size is rejected rather than looping
forever.

**The link between a vector and its text.** Everything between a query and an
answer is numbers, so a result pointing at the wrong row looks exactly like one
pointing at the right row. A keyword-counting stand-in encoder makes this
testable without a model: a query for "protein" must come back with the passage
about proteins, not merely with *a* passage. The guards against a stale index —
wrong corpus size, wrong model — are tested too, because both of those produce
confident nonsense rather than an error.

**What the model is allowed to see.** No test calls a model; the generator is
injected. What is covered instead is everything around it: that the prompt
contains every passage numbered from one, that those numbers match the source
list so a citation can be checked, that the prompt forbids outside knowledge
and permits refusal, that a weak result set returns a refusal *and reaches no
generator at all*, and that one paper cannot take the whole result set.

---

## Project layout

```
src/
  config.py        all tunable values in one place
  data.py          download, normalise, save and load papers
  chunking.py      split papers into overlapping passages
  embed.py         turn text into unit vectors, in resumable blocks
  index.py         build, save and query the FAISS index
  ingest.py        command line: download and chunk
  build_index.py   command line: encode and index
  search.py        command line: search
  answer.py        confidence, prompt assembly, generator backends
  ask.py           command line: ask a question
tests/             pytest suite, no network or model required
data/              downloaded and generated files (gitignored)
```

---

## Licence

MIT
