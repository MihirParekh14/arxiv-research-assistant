"""Shared settings for the research assistant.

Every tunable number lives here so experiments change one file, not five.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

PAPERS_FILE = RAW_DIR / "papers.jsonl"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
VECTORS_FILE = INDEX_DIR / "vectors.npy"
BLOCKS_DIR = INDEX_DIR / "blocks"
INDEX_FILE = INDEX_DIR / "index.faiss"
INDEX_META_FILE = INDEX_DIR / "index_meta.json"

# --- Data source -----------------------------------------------------------

# 138k arXiv papers with full markdown text, not just abstracts.
# Full text matters: a retrieval system that only sees abstracts can answer
# "what is this paper about" but not "which papers used a contrastive loss".
HF_DATASET = "neuralwork/arxiver"
HF_SPLIT = "train"

# --- Chunking --------------------------------------------------------------

# Chunk size is set by the embedding model, not by taste.
#
# all-MiniLM-L6-v2 truncates its input at 256 tokens. Anything longer is
# silently cut off, so the tail of an oversized chunk would never be searchable.
# English averages roughly 1.3 tokens per word, so 150 words lands near
# 195 tokens -- comfortably inside the limit with room for rare long words.
CHUNK_WORDS = 150

# Overlap keeps a sentence that straddles a boundary findable from either side.
# Too little and answers get cut in half; too much and the index bloats with
# near-duplicates. A fifth of the chunk is a common starting point.
CHUNK_OVERLAP_WORDS = 30

# Chunks shorter than this are dropped. A 12-word fragment left over from a
# section heading adds noise to search results without adding meaning.
MIN_CHUNK_WORDS = 25

# --- Models ----------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
EMBEDDING_MAX_TOKENS = 256

# How many chunks the encoder processes at once. Larger batches are faster but
# hold more in memory at the same time. 64 keeps a laptop CPU busy without
# making the machine unusable while it runs.
EMBED_BATCH_SIZE = 64

# How many chunks are encoded before the results are written to disk and the
# memory is released.
#
# This is not a speed knob, it is a survival one. Handing a quarter of a million
# passages to the encoder in a single call holds every input string and every
# output tensor in memory at once; past the point where that fits, the machine
# starts paging to disk and throughput collapses by a factor of sixty rather
# than degrading gently. Encoding in blocks keeps memory flat no matter how
# large the corpus grows, and it means an interrupted run resumes from the last
# finished block instead of starting again.
EMBED_BLOCK_SIZE = 8192

SUMMARIZER_MODEL = "google/pegasus-arxiv"
CLASSIFIER_BASE_MODEL = "roberta-base"

# --- Retrieval -------------------------------------------------------------

TOP_K = 5

# At most this many passages from any one paper may appear in a result set.
#
# Measured, not guessed. On a 5,000-paper index, "how do diffusion models
# handle image inpainting" returned four of its five results from a single
# paper. For search that is only untidy; for a generated answer it is a real
# defect, because all five passages become the material the answer is written
# from, and an answer drawn from one source cites one source and sounds just as
# confident either way.
MAX_CHUNKS_PER_PAPER = 2

# The index is asked for this multiple of TOP_K before the per-paper cap is
# applied, so capping still leaves enough results to fill the list.
RETRIEVAL_OVERSAMPLE = 5

# --- Answer confidence -----------------------------------------------------
#
# The top result's score decides whether an answer is attempted. An earlier
# idea -- treating a narrow spread between the top and bottom scores as low
# confidence -- was tested on a corpus five times larger and did not survive:
# a narrow spread turns out to mean either "nothing matches" or "everything
# matches well", and nothing distinguishes the two. The top score tracked
# answer quality in every run, so it is what is used.
#
# Both thresholds are provisional starting points. Stage 6 measures them
# against a held-out question set rather than leaving them to judgement.
MIN_ANSWER_SCORE = 0.35
WEAK_ANSWER_SCORE = 0.50

# --- Generation ------------------------------------------------------------

ANSWER_BACKEND = "extractive"
ANTHROPIC_MODEL = "claude-sonnet-4-5"
OLLAMA_MODEL = "llama3.2"
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_ANSWER_TOKENS = 700

# --- Reproducibility -------------------------------------------------------

RANDOM_SEED = 42


def ensure_dirs() -> None:
    """Create the data directories if they do not exist yet."""
    for directory in (RAW_DIR, PROCESSED_DIR, INDEX_DIR):
        directory.mkdir(parents=True, exist_ok=True)
