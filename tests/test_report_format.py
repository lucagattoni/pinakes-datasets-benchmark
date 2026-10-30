from __future__ import annotations

from pnkbench import report


def test_a_tiny_p_value_is_shown_as_a_bound_and_never_as_zero() -> None:
    assert report.pvalue(None) == "n/a"
    assert report.pvalue(0.00009999) == "<0.0001"
    assert report.pvalue(0.0) == "<0.0001"
    assert report.pvalue(0.0001) == "0.0001"
    assert report.pvalue(0.03125) == "0.0312"  # 2 * (1/32) / 2: the round-half-even of 0.03125
    assert report.pvalue(1.0) == "1.0000"


def _filters(mismatches: int) -> dict:
    paired = {
        "filtered_better": 3,
        "filtered_worse": 1,
        "both_found": 5,
        "neither_found": 1,
        "sign_test_p": 0.3125,
    }
    proportion = {"successes": 5, "n": 10, "value": 0.5, "ci95": [0.24, 0.76]}
    return {
        "f-cat-tag": {
            "n": 10,
            "mean_admitted_fraction": 0.25,
            "soundness_violations": 0,
            "applied_before_retrieval_violations": 0,
            "all_evidence_at_10_docs_unfiltered": proportion,
            "all_evidence_at_10_docs_filtered": proportion,
            "paired": paired,
        },
        "equivalence_tag_vs_path_prefix": {
            "pairs": 4,
            "mismatches": mismatches,
            "mismatch_ids": ["mhr-0001"] if mismatches else [],
        },
    }


def test_the_equivalence_sentence_ends_cleanly_with_and_without_mismatches() -> None:
    clean = report._filters_section(_filters(0))
    assert clean.endswith("**0 returned different passages or scores**.")
    broken = report._filters_section(_filters(1))
    assert broken.endswith("**1 returned different passages or scores** (e.g. ['mhr-0001']).")


def test_the_abstention_section_does_not_call_the_scores_logits() -> None:
    ab = {
        "auroc_answerable_vs_null": {
            s: {"auroc": 0.9, "ci95": [0.8, 0.95], "n_pos": 5, "n_neg": 3}
            for s in ("all", "eval", "fit")
        },
        "auroc_answered_correctly_vs_null": {
            "auroc": 0.9,
            "ci95": [0.8, 0.95],
            "n_pos": 5,
            "n_neg": 3,
        },
        "auroc_null_without_style_tells_vs_answerable": {
            "auroc": None,
            "ci95": None,
            "n_pos": 5,
            "n_neg": 0,
        },
        "style_tells": {"asks-for-a-letter": {"answerable": 0, "null": 2}, "null_with_no_tell": 1},
        "top1_score_quantiles": {
            side: {k: 0.5 for k in ("min", "p10", "p25", "median", "p75", "p90", "max")}
            for side in ("answerable", "null")
        },
        "labels": None,
    }
    text = report._abstention_section(ab, None)
    assert "logit" not in text.lower()
    assert "probabilities between 0 and 1" in text


def test_the_comparison_shows_max_tokens_and_the_set_aside_reading() -> None:
    paired = {
        "n": 4,
        "a_rate": 0.5,
        "b_rate": 0.25,
        "b_found_a_missed": 0,
        "a_found_b_missed": 1,
        "both_found": 1,
        "neither_found": 2,
        "sign_test_p": 1.0,
    }
    side = {"run_id": "r", "config": "c", "chunking_metadata": "off", "pinakes": "p"}
    result = {
        "a": dict(side, chunking_max_tokens=None),
        "b": dict(side, chunking_max_tokens=459),
        "questions": {"a": 4, "b": 4, "compared": 4},
        "identical_searches": {"of": 4, "same_passages_and_scores": 2},
        "answerable": {"all_evidence_in_first_8_passages": paired},
        "by_question_type": {"comparison_query": paired},
        "null_top1_auroc": {"a": 0.9, "b": 0.9},
        "without_set_aside_articles": {
            "articles": 11,
            "questions_kept": 3,
            "questions_set_aside": 1,
            "answerable": {"all_evidence_in_first_8_passages": paired},
        },
    }
    text = report.render_compare(result)
    assert "| chunking max_tokens | default | 459 |" in text
    assert "## Read without the articles the two configurations chunk differently" in text
    assert "splits 11 articles differently" in text and "leaving 3" in text
    assert "set aside" not in report.render_compare(
        {k: v for k, v in result.items() if k != "without_set_aside_articles"}
    )


def test_articles_chunked_differently_finds_exactly_the_resplit_ones(tmp_path) -> None:
    import sqlite3

    from pnkbench import compare

    def build(name: str, spans: dict[str, list[tuple[int, int]]], *, deleted: tuple[str, ...] = ()):
        path = tmp_path / name
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE documents (id TEXT, path TEXT, state TEXT)")
        db.execute("CREATE TABLE chunks (doc_id TEXT, char_start INT, char_end INT)")
        for number, (doc, found) in enumerate(spans.items()):
            db.execute(
                "INSERT INTO documents VALUES (?, ?, ?)",
                (f"id{number}", doc, "deleted" if doc in deleted else "active"),
            )
            db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?)", [(f"id{number}", s, e) for s, e in found]
            )
        db.commit()
        db.close()
        return path

    one = build(
        "a.db", {"docs/a.txt": [(0, 10), (10, 20)], "docs/b.txt": [(0, 5)], "docs/c.txt": [(0, 7)]}
    )
    two = build(
        "b.db",
        {"docs/a.txt": [(0, 10), (10, 20)], "docs/b.txt": [(0, 3), (3, 5)], "docs/c.txt": [(0, 7)]},
    )
    index = {"docs/a.txt": 0, "docs/b.txt": 1, "docs/c.txt": 2}
    assert compare.articles_chunked_differently(one, two, index) == frozenset({1})
    assert compare.articles_chunked_differently(one, one, index) == frozenset()
    other = build("c.db", {"docs/a.txt": [(0, 10)], "docs/z.txt": [(0, 1)]})
    import pytest

    with pytest.raises(compare.CompareError, match="different documents"):
        compare.articles_chunked_differently(one, other, index)
    gone = build("d.db", {"docs/a.txt": [(0, 10)], "docs/b.txt": [(0, 1)]}, deleted=("docs/b.txt",))
    assert set(compare.chunk_spans(gone)) == {"docs/a.txt"}  # a deleted document is not compared


def test_the_header_names_the_commit_that_scored_the_run_when_it_is_recorded() -> None:
    header = {
        "pinakes": {"version": "0.32.8", "source": "pypi", "commit": None, "code_sha256": "a" * 64},
        "config": {"name": "baseline", "description": "d"},
        "settings": {
            "embedding": {"model": "m", "revision": None, "dim": 3, "provider": "p"},
            "rerank": {"model": "r", "revision": None, "provider": "p"},
            "chunking": {"max_tokens": 510},
            "retrieval": {"final_k": 8},
        },
        "dataset": {"source_repo": "o/r", "revision": "b" * 40, "articles": 1, "queries": 1},
        "items": {"total": 2, "by_variant": {"base": 2}},
        "execution": {"workers": 1, "passages_per_search": 20, "wall_seconds": 5},
        "environment": {"python": "3.13", "torch": "2", "platform": "x", "devices": {}},
        "benchmark": {"version": "0.1.0", "commit": "c" * 40, "dirty": False},
    }
    without = report._header_section(header)
    assert "Scored by" not in without
    scored = {
        "benchmark": {"version": "0.1.0", "commit": "d" * 40, "dirty": True},
        "scored_utc": "2026-10-30T20:40:00+00:00",
    }
    with_row = report._header_section(header, scored)
    assert (
        "| Scored by | v0.1.0, commit `dddddddddd`, 2026-10-30T20:40:00+00:00, **working tree was dirty** |"
        in with_row
    )
    assert "commit `cccccccccc`" in with_row  # the searching commit is still shown
