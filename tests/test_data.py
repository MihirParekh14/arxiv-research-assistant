"""Tests for loading and normalising paper records.

None of these touch the network. Every test builds its own records, so the
suite runs in CI with no dataset download and no API key.
"""

from __future__ import annotations

import pytest

from src.data import (
    Paper,
    clean_text,
    is_usable,
    load_papers,
    normalize_record,
    save_papers,
)


def make_paper(text: str = "word " * 100, paper_id: str = "2401.00001") -> Paper:
    return Paper(
        paper_id=paper_id,
        title="A Study of Things",
        abstract="We study things.",
        text=text.strip(),
        authors="A. Researcher",
        published="2024-01-01",
        url="https://arxiv.org/abs/2401.00001",
    )


# --- clean_text ------------------------------------------------------------


def test_clean_text_collapses_all_whitespace():
    assert clean_text("a\n\nb\t\tc   d") == "a b c d"


def test_clean_text_strips_edges():
    assert clean_text("  padded  ") == "padded"


def test_clean_text_handles_empty_input():
    assert clean_text("") == ""


# --- strip_markdown --------------------------------------------------------


def test_heading_markers_are_removed_but_heading_text_is_kept():
    """"Abstract" locates the passage in the paper; "######" is just syntax."""
    assert clean_text("# A Title\n\n###### Abstract\n\nBody.") == "A Title Abstract Body."


def test_heading_marker_is_only_stripped_at_the_start_of_a_line():
    """A `#` inside a sentence is content, not a heading."""
    assert clean_text("we test C# and F#") == "we test C# and F#"


def test_horizontal_rules_are_removed():
    assert clean_text("Section one.\n\n---\n\nSection two.") == "Section one. Section two."


def test_image_syntax_is_removed_entirely():
    """A figure path is noise: nothing a search query could ever match."""
    assert clean_text("See ![fig 1](images/plot_01.png) for results.") == "See for results."


def test_link_syntax_keeps_the_text_and_drops_the_url():
    assert clean_text("see [our earlier work](https://arxiv.org/abs/1) here") == (
        "see our earlier work here"
    )


def test_subscript_underscores_in_variable_names_survive():
    """Stripping emphasis markers would corrupt the maths in these papers."""
    assert clean_text("we set x_1 and \\sigma_t") == "we set x_1 and \\sigma_t"


def test_bracketed_citations_without_a_url_are_untouched():
    assert clean_text("as shown in [12] and [13]") == "as shown in [12] and [13]"


# --- normalize_record ------------------------------------------------------


def test_normalize_record_reads_standard_fields():
    paper = normalize_record(
        {
            "id": "2401.00001",
            "title": "A Title",
            "abstract": "An abstract.",
            "markdown": "Full body text.",
            "authors": "A. Researcher",
            "published_date": "2024-01-01",
            "link": "https://arxiv.org/abs/2401.00001",
        }
    )
    assert paper.paper_id == "2401.00001"
    assert paper.text == "Full body text."


def test_normalize_record_accepts_alternative_field_names():
    """A different arXiv dump uses `paper_id`, `summary` and `full_text`."""
    paper = normalize_record(
        {
            "paper_id": "2401.00002",
            "title": "Another Title",
            "summary": "An abstract.",
            "full_text": "Body.",
        }
    )
    assert paper.paper_id == "2401.00002"
    assert paper.abstract == "An abstract."
    assert paper.text == "Body."


def test_normalize_record_falls_back_to_abstract_when_no_full_text():
    paper = normalize_record({"id": "x", "abstract": "Only the abstract."})
    assert paper.text == "Only the abstract."


def test_normalize_record_missing_fields_become_empty_strings():
    """A missing author must not crash an ingest of 10,000 papers."""
    paper = normalize_record({"id": "x", "abstract": "Text."})
    assert paper.authors == ""
    assert paper.url == ""


def test_normalize_record_ignores_blank_values_when_choosing_a_field():
    paper = normalize_record({"id": "x", "markdown": "   ", "abstract": "Real text."})
    assert paper.text == "Real text."


# --- is_usable -------------------------------------------------------------


def test_is_usable_accepts_a_normal_paper():
    assert is_usable(make_paper())


def test_is_usable_rejects_paper_with_no_id():
    assert not is_usable(make_paper(paper_id=""))


def test_is_usable_rejects_text_too_short_to_chunk():
    assert not is_usable(make_paper(text="three words only"))


# --- save and load ---------------------------------------------------------


def test_save_then_load_returns_identical_papers(tmp_path):
    original = [make_paper(paper_id="a"), make_paper(paper_id="b")]
    path = tmp_path / "papers.jsonl"

    written = save_papers(original, path)
    assert written == 2

    assert list(load_papers(path)) == original


def test_save_papers_creates_missing_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "papers.jsonl"
    save_papers([make_paper()], path)
    assert path.exists()


def test_load_papers_raises_a_helpful_error_when_file_is_absent(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.ingest"):
        list(load_papers(tmp_path / "nothing.jsonl"))


def test_saved_file_has_one_json_object_per_line(tmp_path):
    path = tmp_path / "papers.jsonl"
    save_papers([make_paper(paper_id="a"), make_paper(paper_id="b")], path)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 2
