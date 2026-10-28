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

SUMMARIZER_MODEL = "google/pegasus-arxiv"
CLASSIFIER_BASE_MODEL = "roberta-base"

# --- Retrieval -------------------------------------------------------------

TOP_K = 5

# --- Reproducibility -------------------------------------------------------

RANDOM_SEED = 42


def ensure_dirs() -> None:
    """Create the data directories if they do not exist yet."""
    for directory in (RAW_DIR, PROCESSED_DIR, INDEX_DIR):
        directory.mkdir(parents=True, exist_ok=True)
