"""Tests for assembling retrieved passages into a cited answer.

No model is called anywhere here. The generator is injected, so what these
tests cover is the part that decides what the model is allowed to see and
whether it should be asked at all -- which is where a retrieval system
produces its most convincing mistakes.
"""

from __future__ import annotations

import pytest

from src import config
from src.answer import (
    INSTRUCTIONS,
    Answer,
    answer_question,
    build_prompt,
    confidence_for,
    format_sources,
    generate,
    generate_extractive,
)
from src.chunking import Chunk
from src.search import SearchResult


def result(score: float, paper_id: str = "2309.00001", index: int = 0,
           text: str = "a passage about diffusion models") -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=f"{paper_id}::{index}",
            paper_id=paper_id,
            title=f"Paper {paper_id}",
            text=text,
            chunk_index=index,
            word_count=len(text.split()),
        ),
        score=score,
    )


def spy_generator(calls: list):
    def generator(prompt, results, backend):
        calls.append((prompt, results, backend))
        return "a generated answer [1]"
    return generator


# --- confidence ------------------------------------------------------------


def test_a_strong_top_score_is_ok():
    assert confidence_for([result(0.73), result(0.70)]) == "ok"


def test_a_middling_top_score_is_weak():
    assert confidence_for([result(0.42)]) == "weak"


def test_a_low_top_score_is_none():
    assert confidence_for([result(0.20)]) == "none"


def test_no_results_is_none():
    assert confidence_for([]) == "none"


def test_confidence_reads_the_top_score_not_the_spread():
    """The spread was the first idea and it did not survive a bigger corpus:
    a narrow spread means either nothing matches or everything does."""
    narrow_and_strong = [result(0.73), result(0.72), result(0.71)]
    narrow_and_weak = [result(0.30), result(0.29), result(0.28)]

    assert confidence_for(narrow_and_strong) == "ok"
    assert confidence_for(narrow_and_weak) == "none"


# --- prompt assembly -------------------------------------------------------


def test_prompt_contains_every_passage_numbered_from_one():
    results = [result(0.7, "a", 0, "first passage"), result(0.6, "b", 0, "second passage")]
    prompt = build_prompt("why?", results)

    assert "[1] Paper a" in prompt
    assert "[2] Paper b" in prompt
    assert "first passage" in prompt
    assert "second passage" in prompt


def test_prompt_contains_the_question():
    assert "why does it work?" in build_prompt("why does it work?", [result(0.7)])


def test_prompt_tells_the_model_to_use_only_the_passages():
    """Without this the model answers from training data and cites a passage
    that never said it."""
    prompt = build_prompt("q", [result(0.7)])
    assert "only what the passages say" in prompt
    assert INSTRUCTIONS in prompt


def test_prompt_tells_the_model_it_may_refuse():
    assert "do not answer" in build_prompt("q", [result(0.7)]).lower() or \
           "say so plainly" in build_prompt("q", [result(0.7)])


def test_prompt_carries_the_paper_id_so_a_citation_can_be_checked():
    prompt = build_prompt("q", [result(0.7, paper_id="2309.09614")])
    assert "arXiv:2309.09614" in prompt


def test_prompt_numbering_matches_the_source_list_numbering():
    """A citation is only verifiable if [2] means the same passage in both."""
    results = [result(0.7, "aaa"), result(0.6, "bbb"), result(0.5, "ccc")]

    prompt = build_prompt("q", results)
    sources = format_sources(results)

    for number, paper_id in enumerate(["aaa", "bbb", "ccc"], start=1):
        assert f"[{number}] Paper {paper_id}" in prompt
        assert f"[{number}] Paper {paper_id}" in sources


def test_sources_include_a_link_to_each_paper():
    assert "https://arxiv.org/abs/2309.09614" in format_sources(
        [result(0.7, paper_id="2309.09614")]
    )


# --- the refusal path ------------------------------------------------------


def test_a_weak_corpus_match_produces_a_refusal_and_calls_no_model():
    calls = []
    answer = answer_question("q", [result(0.10)], generator=spy_generator(calls))

    assert answer.confidence == "none"
    assert calls == []  # no prompt built, no model called, no tokens spent
    assert "could not find" in answer.text


def test_no_results_at_all_produces_a_refusal():
    answer = answer_question("q", [], generator=spy_generator([]))
    assert answer.confidence == "none"


def test_a_borderline_match_answers_but_warns():
    calls = []
    answer = answer_question("q", [result(0.42)], generator=spy_generator(calls))

    assert answer.confidence == "weak"
    assert len(calls) == 1
    assert "treat this as a partial answer" in answer.text
    assert "a generated answer" in answer.text


def test_a_strong_match_answers_without_a_warning():
    answer = answer_question("q", [result(0.73)], generator=spy_generator([]))

    assert answer.confidence == "ok"
    assert "partial answer" not in answer.text


# --- plumbing --------------------------------------------------------------


def test_the_generator_receives_the_assembled_prompt_and_the_backend():
    calls = []
    results = [result(0.8, "xyz", 0, "the passage text")]
    answer_question("the question", results, backend="ollama",
                    generator=spy_generator(calls))

    prompt, passed_results, backend = calls[0]
    assert "the question" in prompt
    assert "the passage text" in prompt
    assert passed_results == results
    assert backend == "ollama"


def test_the_answer_keeps_the_passages_it_was_built_from():
    results = [result(0.8, "a"), result(0.7, "b")]
    answer = answer_question("q", results, generator=spy_generator([]))

    assert answer.results == results
    assert answer.top_score == pytest.approx(0.8)


def test_an_unknown_backend_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown backend"):
        generate("prompt", [result(0.8)], backend="gpt-9")


def test_the_extractive_backend_needs_no_model_and_says_so():
    text = generate_extractive("prompt", [result(0.8, text="some retrieved words")])

    assert "no model used" in text
    assert "some retrieved words" in text
    assert "[1]" in text


def test_the_extractive_backend_handles_an_empty_result_set():
    assert "No passages matched" in generate_extractive("prompt", [])


def test_answer_is_immutable():
    """An answer and its citations should not drift apart after the fact."""
    answer = Answer(text="t", results=[], confidence="none", backend="extractive")
    with pytest.raises(AttributeError):
        answer.text = "something else"


# --- the thresholds themselves ---------------------------------------------


def test_the_thresholds_are_ordered():
    assert 0 < config.MIN_ANSWER_SCORE < config.WEAK_ANSWER_SCORE < 1
