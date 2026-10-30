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
