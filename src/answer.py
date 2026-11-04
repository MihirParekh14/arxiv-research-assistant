"""Turn retrieved passages into an answer with citations.

Retrieval finds passages. This module decides whether they are good enough to
answer from, assembles them into a prompt, and asks a model to write an answer
that cites them.

The important part is not the model. It is the two rules around it: answer only
from the passages, and refuse when the passages do not support an answer. A
retrieval system that always produces fluent text is worse than one that
sometimes says it does not know, because a wrong answer with a citation
attached is harder to catch than no answer at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.search import SearchResult

BACKENDS = ("extractive", "anthropic", "ollama")

INSTRUCTIONS = """\
Answer the question using only the numbered passages below.

Rules:
- Use only what the passages say. Do not add anything you know from elsewhere.
- After each claim, cite the passages it came from, like [1] or [2][4].
- If the passages do not answer the question, say so plainly and stop. Do not
  guess, and do not fill the gap with general knowledge.
- Be concise. Three to six sentences is usually enough.
"""


@dataclass(frozen=True)
class Answer:
    """An answer, the passages behind it, and how much to trust it."""

    text: str
    results: list[SearchResult]
    confidence: str  # "none", "weak" or "ok"
    backend: str

    @property
    def top_score(self) -> float:
        """The score the confidence gate actually read."""
        return self.results[0].relevance if self.results else 0.0

    @property
    def scale(self) -> str:
        """Which scale ``top_score`` is on, since the two are not comparable."""
        if self.results and self.results[0].was_reranked:
            return "cross-encoder"
        return "cosine"


def confidence_for(results: list[SearchResult]) -> str:
    """Grade a result set by its best score.

    The top score is used rather than the spread between top and bottom. An
    earlier version used the spread, on the theory that a flat set of scores
    meant the corpus had nothing to offer. Re-running the same queries against
    a corpus five times larger disproved it: a flat spread appeared both when
    nothing matched and when everything matched well, and the two are
    indistinguishable from the spread alone. The top score separated good
    answers from bad ones in every run.

    Which thresholds apply depends on where the top score came from. Cosine
    similarity runs 0 to 1; a cross-encoder logit runs roughly -11 to +11.
    Grading a logit against the cosine thresholds would pass every reranked
    answer, including the ones that exist to be refused.
    """
    if not results:
        return "none"

    top = results[0]
    floor, weak = thresholds_for(results)

    if top.relevance < floor:
        return "none"
    if top.relevance < weak:
        return "weak"
    return "ok"


def thresholds_for(results: list[SearchResult]) -> tuple[float, float]:
    """The (floor, weak) pair that applies to how these results were scored."""
    if results and results[0].was_reranked:
        return config.MIN_RERANK_SCORE, config.WEAK_RERANK_SCORE
    return config.MIN_ANSWER_SCORE, config.WEAK_ANSWER_SCORE


def usable(results: list[SearchResult]) -> list[SearchResult]:
    """Drop passages that score below the floor.

    Ranking a bad passage last is not enough. Everything that reaches the
    prompt is material the model is invited to answer from, and a passage the
    reranker scored at -6 is not weak evidence, it is the wrong subject --
    LoRa the radio protocol sitting underneath a question about fine-tuning.
    Leaving it in the context and hoping the model ignores it is a worse
    design than removing it.
    """
    floor, _ = thresholds_for(results)
    return [result for result in results if result.relevance >= floor]


def build_prompt(question: str, results: list[SearchResult]) -> str:
    """Assemble the passages and the question into one prompt.

    Passages are numbered from 1 and carry their title and paper id, so the
    model can cite them and a reader can check the citation against the source
    list printed underneath the answer. The numbering here and the numbering in
    :func:`format_sources` are the same ordering, which is what makes a
    citation verifiable rather than decorative.
    """
    blocks = []
    for number, result in enumerate(results, start=1):
        chunk = result.chunk
        blocks.append(
            f"[{number}] {chunk.title} (arXiv:{chunk.paper_id})\n{chunk.text}"
        )

    return (
        f"{INSTRUCTIONS}\n"
        f"Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n\nAnswer:"
    )


def format_sources(results: list[SearchResult]) -> str:
    """List the passages behind an answer, numbered to match its citations."""
    lines = []
    for number, result in enumerate(results, start=1):
        chunk = result.chunk
        scores = (f"rerank {result.rerank_score:+.2f}  retrieval {result.score:.3f}"
                  if result.was_reranked else f"score {result.score:.3f}")
        lines.append(
            f"[{number}] {chunk.title}\n"
            f"    arXiv:{chunk.paper_id}  chunk {chunk.chunk_index}  {scores}\n"
            f"    https://arxiv.org/abs/{chunk.paper_id}"
        )
    return "\n".join(lines)


# --- generator backends ----------------------------------------------------


def _load_env_file(path: str | Path = ".env") -> None:
    """Read KEY=value lines from a .env file into the environment.

    Six lines instead of a dependency. Values already in the environment win,
    so an exported key is not overwritten by a stale file.
    """
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def generate_extractive(prompt: str, results: list[SearchResult]) -> str:
    """Return the retrieved passages themselves, with no model involved.

    This is the default so the project runs with no API key and no local model,
    and so the retrieval half can be judged on its own. It is honest about what
    it is: these are the passages, not a written answer.
    """
    if not results:
        return "No passages matched the question."

    lines = ["Retrieved passages (no model used -- this is extractive output):", ""]
    for number, result in enumerate(results, start=1):
        snippet = " ".join(result.chunk.text.split()[:60])
        lines.append(f"[{number}] {snippet} ...")
        lines.append("")
    return "\n".join(lines).rstrip()


def generate_anthropic(prompt: str, model: str = config.ANTHROPIC_MODEL) -> str:
    """Generate with the Anthropic API. Needs ANTHROPIC_API_KEY."""
    _load_env_file()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Put it in a .env file "
            "(see .env.example) or use --backend extractive."
        )

    try:
        import anthropic  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "The anthropic package is not installed. Run `pip install anthropic`, "
            "or use --backend ollama or --backend extractive."
        ) from exc

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=config.MAX_ANSWER_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if block.type == "text").strip()


def generate_ollama(prompt: str, model: str = config.OLLAMA_MODEL) -> str:
    """Generate with a model served locally by Ollama.

    Uses urllib from the standard library rather than adding a dependency for
    one POST request.
    """
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": config.MAX_ANSWER_TOKENS},
    }).encode("utf-8")

    request = urllib.request.Request(
        config.OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read()).get("response", "").strip()
    except urllib.error.URLError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            f"Could not reach Ollama at {config.OLLAMA_URL}. Is `ollama serve` running, "
            f"and have you pulled the model with `ollama pull {model}`?"
        ) from exc


def generate(prompt: str, results: list[SearchResult], backend: str) -> str:
    """Send the prompt to the chosen backend."""
    if backend == "extractive":
        return generate_extractive(prompt, results)
    if backend == "anthropic":
        return generate_anthropic(prompt)
    if backend == "ollama":
        return generate_ollama(prompt)
    raise ValueError(f"unknown backend {backend!r}. Choose one of {', '.join(BACKENDS)}.")


# --- the whole thing -------------------------------------------------------


def answer_question(
    question: str,
    results: list[SearchResult],
    backend: str = config.ANSWER_BACKEND,
    generator=generate,
) -> Answer:
    """Produce an :class:`Answer` from retrieved passages.

    When the best passage scores below the floor, no prompt is built and no
    model is called. Refusing early costs nothing and is the only thing
    standing between a bad retrieval and a fluent, well-cited, wrong answer.

    ``generator`` is injectable so the assembly logic can be tested without a
    model, an API key or a network.
    """
    confidence = confidence_for(results)
    # Only passages above the floor reach the prompt, and only those are shown
    # as sources, so the citations list matches what the answer was built from.
    kept = usable(results) if confidence != "none" else results

    if confidence == "none":
        return Answer(
            text=(
                "I could not find passages in this corpus that answer the question. "
                "The closest matches were too weak to answer from, so rather than "
                "assembling something plausible out of them, here is nothing."
            ),
            results=results,
            confidence=confidence,
            backend=backend,
        )

    prompt = build_prompt(question, kept)
    text = generator(prompt, kept, backend)

    if confidence == "weak":
        text = (
            "Note: the best passage scored low, so treat this as a partial answer "
            "and check the sources.\n\n" + text
        )

    return Answer(text=text, results=kept, confidence=confidence, backend=backend)
