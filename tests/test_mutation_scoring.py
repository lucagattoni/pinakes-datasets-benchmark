"""Tests written against the mutants of `scoring.py` and `admission.py` that an independent review
found still surviving the suite (the suite stayed green with the code deliberately broken).

Every expected value comes from a module docstring, the README's "What is measured", a statistic's
textbook definition, or arithmetic done by hand beside the assertion. Where a number could only
come from running the code, an independent re-implementation of the definition stands in for it.
The fit/eval membership of the toy queries was computed with `shasum`, not with the code under test:
eval is mhr-0000, 0001, 0004, 0007, 0008, 0009; fit is mhr-0002, 0003, 0005, 0006.
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import replace
from typing import Any

import pytest
from toy import make_dataset

from pnkbench import items, layout, scoring
from pnkbench.admission import FilterSpec, admits
from pnkbench.multihop_rag import Dataset, Evidence, Query


def _row(
    item_id: str,
    passages: list[list[Any]],
    *,
    variant: str = "base",
    base: str | None = None,
    considered: int = 20,
    ms: int = 5,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "base_id": base or item_id.split(":")[0],
        "variant": variant,
        "conf": "unknown",
        "considered": considered,
        "p": passages,
        "ms": ms,
    }


def _ps(*indices: int, top: float = 9.0) -> list[list[Any]]:
    """Passages from the given article indices, in that rank order, scores falling from `top`."""
    return [[index, top - 0.01 * rank] for rank, index in enumerate(indices)]


def _info(
    *,
    answerable: bool = True,
    gold: tuple[int, ...] = (1, 2),
    split: str = "eval",
    tells: tuple[str, ...] = (),
    qtype: str = "comparison_query",
) -> scoring.QueryInfo:
    return scoring.QueryInfo(
        qtype=qtype, answerable=answerable, gold=frozenset(gold), split=split, tells=tells
    )


def _brute_auroc(pos: list[float], neg: list[float]) -> float:
    """The definition: P(positive outscores negative), a tie counting half, over all pairs."""
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


# --- proportions and the bootstrap interval -----------------------------------------------


def test_an_empty_proportion_has_no_value_and_no_interval() -> None:
    assert scoring.proportion(0, 0) == {"successes": 0, "n": 0, "value": None, "ci95": None}


def test_a_proportion_carries_its_wilson_interval_as_a_list() -> None:
    p = scoring.proportion(3, 4)
    assert (p["successes"], p["n"], p["value"]) == (3, 4, 0.75)
    # Wilson, z = 1.96, 3 of 4, by hand: denominator 1 + 3.8416/4 = 1.9604, centre
    # (0.75 + 0.4802) / 1.9604 = 0.62753, half-width 1.96 * sqrt(0.046875 + 0.060025) / 1.9604
    # = 0.32689, so (0.3006, 0.9544): the textbook interval for 3/4.
    assert isinstance(p["ci95"], list)
    assert p["ci95"] == pytest.approx([0.3006, 0.9544], abs=2e-4)


def _asymmetric_sample() -> tuple[list[float], list[float]]:
    """13 positives and 19 negatives, two decimals so that ties occur."""
    rng = random.Random(11)
    return (
        [round(rng.gauss(0.8, 1.0), 2) for _ in range(13)],
        [round(rng.gauss(0.0, 1.0), 2) for _ in range(19)],
    )


def _bootstrap_oracle(pos: list[float], neg: list[float], resamples: int, seed: int) -> list[float]:
    """The documented procedure written out independently: resample each side with replacement at
    its own size (positives first), score every resample by pairs, take the 2.5th and 97.5th
    percentile of the sorted draws at index int(q * (resamples - 1))."""
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        p = rng.choices(pos, k=len(pos))
        n = rng.choices(neg, k=len(neg))
        draws.append(_brute_auroc(p, n))
    draws.sort()
    return [draws[int(0.025 * (resamples - 1))], draws[int(0.975 * (resamples - 1))]]


def test_the_bootstrap_interval_is_the_documented_procedure_with_500_resamples_and_the_seed() -> (
    None
):
    pos, neg = _asymmetric_sample()
    got = scoring.auroc_with_ci(pos, neg)
    assert got["auroc"] == pytest.approx(_brute_auroc(pos, neg))
    assert (got["n_pos"], got["n_neg"]) == (13, 19)
    # README: "bootstrap (500 resamples, fixed seed)"; both literals are spelled out here on
    # purpose, so a changed constant changes every committed interval and fails this test.
    assert got["ci95"] == pytest.approx(_bootstrap_oracle(pos, neg, 500, 20261030), abs=1e-12)


def test_the_readme_states_500_resamples_and_the_fixed_seed_is_pinned() -> None:
    # An interval built on 499 or 501 draws can land on the same order statistics as one built
    # on 500, so the count is pinned as well as exercised.
    assert scoring.BOOTSTRAP_RESAMPLES == 500
    assert scoring.BOOTSTRAP_SEED == 20261030


# 41 and 81 are the counts where 0.025 * (n - 1) is a whole number (1 and 2), the edge of the rule
@pytest.mark.parametrize("resamples", [40, 41, 81, 200, 1000])
def test_the_bootstrap_percentile_index_rule_holds_for_other_resample_counts(
    monkeypatch: pytest.MonkeyPatch, resamples: int
) -> None:
    pos, neg = _asymmetric_sample()
    monkeypatch.setattr(scoring, "BOOTSTRAP_RESAMPLES", resamples)
    got = scoring.auroc_with_ci(pos, neg)["ci95"]
    assert got == pytest.approx(_bootstrap_oracle(pos, neg, resamples, 20261030), abs=1e-12)


def test_the_bootstrap_interval_of_a_perfect_separation_is_a_point() -> None:
    assert scoring.auroc_with_ci([5.0, 6.0, 7.0], [1.0, 2.0]) == {
        "auroc": 1.0,
        "ci95": [1.0, 1.0],
        "n_pos": 3,
        "n_neg": 2,
    }


def test_the_bootstrap_interval_of_an_empty_side_is_none_but_keeps_the_counts() -> None:
    assert scoring.auroc_with_ci([], [1.0, 2.0]) == {
        "auroc": None,
        "ci95": None,
        "n_pos": 0,
        "n_neg": 2,
    }
    assert scoring.auroc_with_ci([1.0], []) == {
        "auroc": None,
        "ci95": None,
        "n_pos": 1,
        "n_neg": 0,
    }


# --- quantiles ----------------------------------------------------------------------------


def test_quantiles_of_eleven_evenly_spaced_values_sit_on_their_positions() -> None:
    # position = fraction * (n - 1) = fraction * 10, so each quantile is just that position
    q = scoring._quantiles([float(n) for n in range(11)])
    assert list(q) == ["min", "p10", "p25", "median", "p75", "p90", "max"]
    assert q == pytest.approx(
        {"min": 0.0, "p10": 1.0, "p25": 2.5, "median": 5.0, "p75": 7.5, "p90": 9.0, "max": 10.0}
    )


def test_quantiles_sort_their_input_and_interpolate_linearly() -> None:
    # numpy.percentile([1, 2, 3, 4, 5], [10, 25, 50, 75, 90]) is [1.4, 2, 3, 4, 4.6]
    q = scoring._quantiles([5.0, 1.0, 4.0, 2.0, 3.0])
    assert q == pytest.approx(
        {"min": 1.0, "p10": 1.4, "p25": 2.0, "median": 3.0, "p75": 4.0, "p90": 4.6, "max": 5.0}
    )


def test_quantiles_of_one_value_are_that_value_and_of_nothing_are_none() -> None:
    assert set(scoring._quantiles([7.5]).values()) == {7.5}  # type: ignore[union-attr]
    assert scoring._quantiles([]) is None


# --- the answerable measures --------------------------------------------------------------


def _five_answerable_rows() -> list[dict[str, Any]]:
    """Gold sets (toy): mhr-0000 {0,3}, mhr-0008 {10,11}, mhr-0003 {5,8}, mhr-0001 {1,2,3},
    mhr-0006 {0,4}. Each row is built so its outcome at every k is readable off the passages."""
    return [
        _row("mhr-0000", _ps(0, 3)),  # both gold at once
        _row("mhr-0008", _ps(1, 2, 4, 5, 6, 7, 10, 11)),  # gold at passages and docs 7 and 8
        _row("mhr-0003", _ps(5, 5, 5, 7, 8)),  # three passages of 5, then 7, then 8
        _row("mhr-0001", _ps(9, 1)),  # one of three gold articles, at rank 2
        _row("mhr-0006", _ps(7)),  # a miss
    ]


def test_every_answerable_measure_on_five_hand_built_rows() -> None:
    info = scoring.query_info(make_dataset())
    m = scoring.answerable_metrics(_five_answerable_rows(), info)
    assert m["n"] == 5

    # all gold within the first k distinct articles: A and C by 3 (C's docs are [5, 7, 8]), B only
    # by 10 (its gold are the 7th and 8th article), D and E never
    assert list(m["all_evidence_at_docs"]) == ["3", "5", "10"]
    assert [m["all_evidence_at_docs"][k]["successes"] for k in ("3", "5", "10")] == [2, 2, 3]
    assert all(m["all_evidence_at_docs"][k]["n"] == 5 for k in ("3", "5", "10"))
    assert m["all_evidence_at_docs"]["10"]["value"] == pytest.approx(0.6)

    # share of gold found: at 5 (A 1, B 0, C 1, D 1/3, E 0) and at 10 (A 1, B 1, C 1, D 1/3, E 0)
    assert list(m["coverage_at_docs"]) == ["5", "10"]
    assert m["coverage_at_docs"]["5"] == pytest.approx((1 + 0 + 1 + 1 / 3 + 0) / 5)
    assert m["coverage_at_docs"]["10"] == pytest.approx((1 + 1 + 1 + 1 / 3 + 0) / 5)

    # all gold within the first k passages: B's gold are passages 7 and 8, C's are passages 1 and 5
    assert list(m["all_evidence_at_passages"]) == ["5", "8", "20"]
    assert [m["all_evidence_at_passages"][k]["successes"] for k in ("5", "8", "20")] == [2, 3, 3]

    # any gold within the first k passages: B's first gold is passage 7, D's is passage 2
    assert list(m["any_evidence_at_passages"]) == ["5", "8", "20"]
    assert [m["any_evidence_at_passages"][k]["successes"] for k in ("5", "8", "20")] == [3, 4, 4]

    # first gold at article rank 1, 7, 1, 2 and never: a miss contributes 0, not 1
    assert m["mrr_first_gold"] == pytest.approx((1 + 1 / 7 + 1 + 1 / 2 + 0) / 5)

    # nDCG@10 over articles, gain 1/log2(rank + 1); the ideal places min(|gold|, 10) hits first
    def dcg(ranks: list[int]) -> float:
        return sum(1 / math.log2(rank + 1) for rank in ranks)

    ndcgs = [
        dcg([1, 2]) / dcg([1, 2]),  # A
        dcg([7, 8]) / dcg([1, 2]),  # B
        dcg([1, 3]) / dcg([1, 2]),  # C: docs [5, 7, 8], gold {5, 8}
        dcg([2]) / dcg([1, 2, 3]),  # D: gold has three articles
        0.0,  # E
    ]
    assert m["ndcg_at_10_docs"] == pytest.approx(sum(ndcgs) / 5)

    # distinct articles in each row's passages: 2, 8, 3, 2, 1 -> sorted [1, 2, 2, 3, 8]
    assert m["distinct_articles_in_20_passages"] == pytest.approx(
        {"min": 1.0, "p10": 1.4, "p25": 2.0, "median": 2.0, "p75": 3.0, "p90": 6.0, "max": 8.0}
    )


def test_the_ten_article_window_is_exactly_ten() -> None:
    info = scoring.query_info(make_dataset())
    rows = [
        # gold {0, 3} at article ranks 9 and 10: inside
        _row("mhr-0000", _ps(1, 2, 4, 5, 6, 7, 8, 9, 0, 3)),
        # gold {0, 11} at article ranks 10 and 11: the second one is outside
        _row("mhr-0009", _ps(1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 11)),
    ]
    m = scoring.answerable_metrics(rows, info)
    assert [m["all_evidence_at_docs"][k]["successes"] for k in ("3", "5", "10")] == [0, 0, 1]
    assert m["coverage_at_docs"]["5"] == 0.0
    assert m["coverage_at_docs"]["10"] == pytest.approx((1.0 + 0.5) / 2)
    # nDCG@10: the first row earns the gains of ranks 9 and 10, the second only that of rank 10
    # (its rank 11 is outside); the ideal for two gold articles is ranks 1 and 2
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    first = (1 / math.log2(10) + 1 / math.log2(11)) / ideal
    second = (1 / math.log2(11)) / ideal
    assert m["ndcg_at_10_docs"] == pytest.approx((first + second) / 2)


def test_a_passage_window_of_k_does_not_reach_passage_k_plus_one_for_any_evidence() -> None:
    gold = frozenset({4})
    row = _row("a", _ps(9, 9, 9, 4))  # the gold article is the 4th passage
    assert not scoring.found_any_passages(row, gold, 3)
    assert scoring.found_any_passages(row, gold, 4)


def test_an_empty_answerable_set_has_no_values_rather_than_zeros() -> None:
    m = scoring.answerable_metrics([], scoring.query_info(make_dataset()))
    assert m["n"] == 0
    assert m["mrr_first_gold"] is None and m["ndcg_at_10_docs"] is None
    assert m["distinct_articles_in_20_passages"] is None
    assert set(m["coverage_at_docs"].values()) == {None}
    for block in ("all_evidence_at_docs", "all_evidence_at_passages", "any_evidence_at_passages"):
        assert all(p["value"] is None and p["ci95"] is None for p in m[block].values())


def test_grouped_orders_its_groups_by_key_and_keeps_each_groups_row_order() -> None:
    rows = [{"k": "temporal", "n": 1}, {"k": "comparison", "n": 2}, {"k": "temporal", "n": 3}]
    groups = scoring.grouped(rows, lambda r: r["k"])
    assert list(groups) == ["comparison", "temporal"]
    assert [r["n"] for r in groups["temporal"]] == [1, 3]
    assert list(scoring.grouped([{"s": 4}, {"s": 2}, {"s": 3}], lambda r: r["s"])) == [2, 3, 4]
    # ordered by the key, not by what the group holds: the rows are written so that their text
    # sorts the other way round ("b" holds the row that sorts first)
    by_key = scoring.grouped([{"n": 1, "k": "b"}, {"n": 2, "k": "a"}], lambda r: r["k"])
    assert list(by_key) == ["a", "b"]


# --- style tells --------------------------------------------------------------------------


def test_the_letter_and_opening_tells_ignore_case_but_the_outlet_does_not() -> None:
    import re

    pattern = re.compile(r"\bThe Verge\b")
    assert scoring.TELL_LETTER in scoring.style_tells("Which LETTER is it? The Verge asks", pattern)
    assert scoring.TELL_LETTER in scoring.style_tells("Letter from The Verge", pattern)
    assert scoring.TELL_OPENING in scoring.style_tells(
        "CONSIDERING THE INFORMATION given, The Verge", pattern
    )
    # a name has to match as a whole word, with its capitals
    assert scoring.TELL_NO_OUTLET in scoring.style_tells("The Verger said so", pattern)
    assert scoring.style_tells("The Verge said so", pattern) == ()
    # the opening tell is the whole phrase, not its first words
    assert scoring.style_tells("Considering the weather, did The Verge cover it?", pattern) == ()
    assert scoring.style_tells("Considering the information, did The Verge cover it?", pattern) == (
        scoring.TELL_OPENING,
    )


def test_query_info_reads_type_gold_split_and_tells_from_the_dataset() -> None:
    dataset = make_dataset()
    info = scoring.query_info(dataset)
    assert info["mhr-0003"].qtype == "comparison_query"
    assert info["mhr-0003"].gold == frozenset({5, 8})  # article 5 is listed twice
    assert info["mhr-0003"].answerable is True and info["mhr-0005"].answerable is False
    assert info["mhr-0005"].gold == frozenset()
    assert info["mhr-0000"].split == items.EVAL and info["mhr-0002"].split == items.FIT


def test_an_outlet_name_with_a_regex_metacharacter_matches_only_itself() -> None:
    dataset = make_dataset()
    articles = (replace(dataset.articles[0], source="Fast.Co"), *dataset.articles[1:])
    queries = (
        replace(dataset.queries[0], text="What did Fast.Co report?"),
        replace(dataset.queries[1], text="What did FastXCo report?"),
        *dataset.queries[2:],
    )
    info = scoring.query_info(replace(dataset, articles=articles, queries=queries))
    assert info["mhr-0000"].tells == ()
    assert info["mhr-0001"].tells == (scoring.TELL_NO_OUTLET,)


# --- the abstention report ----------------------------------------------------------------


def _abstention_fixture() -> tuple[list[dict[str, Any]], dict[str, scoring.QueryInfo]]:
    """Four answerable questions (gold {1, 2}) and three null ones, each with a known top-1 score.

    a1 eval  5.0  found at article ranks 1, 2
    a2 fit  -10.0 miss
    a3 eval  6.0  gold at article ranks 9 and 10: found within 10, not within 9 or 5
    a4 fit   4.5  gold at article ranks 10 and 11: not found within 10 (found within 11)
    n1 eval  0.0, n2 fit 5.5, n3 eval no results at all
    """
    fillers = list(range(3, 12))  # nine articles that are nobody's gold here
    rows = [
        _row("a1", _ps(1, 2, top=5.0)),
        _row("a2", _ps(9, top=-10.0)),
        _row("a3", _ps(3, 4, 5, 6, 7, 8, 9, 10, 1, 2, top=6.0)),
        _row("a4", _ps(*fillers, 1, 2, top=4.5)),
        _row("n1", _ps(0, top=0.0)),
        _row("n2", _ps(0, top=5.5)),
        _row("n3", []),
    ]
    info = {
        "a1": _info(split="eval"),
        "a2": _info(split="fit", tells=(scoring.TELL_NO_OUTLET,)),
        "a3": _info(split="eval"),
        "a4": _info(split="fit", tells=(scoring.TELL_NO_OUTLET,)),
        "n1": _info(answerable=False, split="eval"),
        "n2": _info(
            answerable=False,
            split="fit",
            tells=(scoring.TELL_LETTER, scoring.TELL_OPENING),
        ),
        "n3": _info(answerable=False, split="eval"),
    }
    return rows, info


def test_the_auroc_blocks_pair_the_right_scores_on_the_right_sides() -> None:
    rows, info = _abstention_fixture()
    report = scoring.abstention_report(rows, info, None)

    # answerable [5, -10, 6, 4.5] against null [0.0, 5.5] (n3 has no score): wins 1+0+2+1 of 8
    everyone = report["auroc_answerable_vs_null"]["all"]
    assert everyone["auroc"] == pytest.approx(4 / 8)
    assert (everyone["n_pos"], everyone["n_neg"]) == (4, 2)
    # eval: [5, 6] against [0.0]; fit: [-10, 4.5] against [5.5]
    assert report["auroc_answerable_vs_null"]["eval"]["auroc"] == 1.0
    assert report["auroc_answerable_vs_null"]["fit"]["auroc"] == 0.0
    assert report["auroc_answerable_vs_null"]["eval"]["n_pos"] == 2


def test_the_answered_correctly_auroc_uses_the_questions_found_within_ten_articles() -> None:
    rows, info = _abstention_fixture()
    block = scoring.abstention_report(rows, info, None)["auroc_answered_correctly_vs_null"]
    # found within 10 articles: a1 (5.0) and a3 (6.0, gold at ranks 9 and 10); a4 (ranks 10 and
    # 11) is not. Against null [0.0, 5.5]: 5.0 beats 0.0 only, 6.0 beats both -> 3 of 4.
    assert block["auroc"] == pytest.approx(3 / 4)
    assert (block["n_pos"], block["n_neg"]) == (2, 2)


def test_the_style_controlled_auroc_keeps_only_null_questions_with_no_tell() -> None:
    rows, info = _abstention_fixture()
    block = scoring.abstention_report(rows, info, None)[
        "auroc_null_without_style_tells_vs_answerable"
    ]
    # null with no tell: n1 (0.0) and n3 (no score). Answerable [5, -10, 6, 4.5] against [0.0]:
    # 5 wins, -10 loses, 6 wins, 4.5 wins -> 3 of 4.
    assert block["auroc"] == pytest.approx(3 / 4)
    assert (block["n_pos"], block["n_neg"]) == (4, 1)


def test_the_style_tell_counts_are_split_by_answerable_and_null() -> None:
    rows, info = _abstention_fixture()
    tells = scoring.abstention_report(rows, info, None)["style_tells"]
    assert tells[scoring.TELL_LETTER] == {"answerable": 0, "null": 1}
    assert tells[scoring.TELL_OPENING] == {"answerable": 0, "null": 1}
    assert tells[scoring.TELL_NO_OUTLET] == {"answerable": 2, "null": 0}
    assert tells["null_with_no_tell"] == 2  # n1 and n3


def test_no_results_counts_the_rows_that_returned_nothing_and_scores_skip_them() -> None:
    rows, info = _abstention_fixture()
    report = scoring.abstention_report(rows, info, None)
    assert report["no_results"] == 1  # n3 only
    quantiles = report["top1_score_quantiles"]
    # answerable sorted [-10, 4.5, 5, 6]: median is midway between 4.5 and 5.0
    assert quantiles["answerable"]["min"] == -10.0 and quantiles["answerable"]["max"] == 6.0
    assert quantiles["answerable"]["median"] == pytest.approx(4.75)
    # null [0.0, 5.5]: n3 contributes no score
    assert quantiles["null"]["min"] == 0.0 and quantiles["null"]["max"] == 5.5
    assert quantiles["null"]["median"] == pytest.approx(2.75)
    assert report["labels"] is None and "thresholds" not in report


def _label_fixture() -> tuple[list[dict[str, Any]], dict[str, scoring.QueryInfo]]:
    """Thresholds -3 and 1; gold {1, 2} throughout.

    eval answerable (6): e1 hit and low, e2 low but its only gold article is the 6th passage,
    e3 low with one of two gold articles first, e4 found and medium, e5 found and low, e6 low
    with one of two gold articles as the 5th passage.
    eval null (7): scores 2.0 (high), 1.0 and -3.0 (both medium: the rule is strict), no
    results (unknown), -7.0 (low), -2.0 and -2.5 (medium).
    fit answerable (3): f1 found and high, f2 low with one of two gold articles, f3 found, medium.
    fit null (2): g1 high, g2 low.
    The two sides have different numbers of answerable and null questions on purpose, so a
    denominator taken from the wrong group shows.
    """
    rows = [
        _row("e1", _ps(1, 2, top=-5.0)),
        _row("e2", _ps(7, 8, 9, 10, 11, 1, top=-5.0)),
        _row("e3", _ps(1, top=-5.0)),
        _row("e4", [[1, 0.0], [2, -1.0]]),
        _row("e5", _ps(2, 1, top=-4.0)),
        _row("e6", _ps(7, 8, 9, 10, 1, top=-5.0)),
        _row("z1", _ps(0, top=2.0)),
        _row("z2", _ps(0, top=1.0)),
        _row("z3", _ps(0, top=-3.0)),
        _row("z4", []),
        _row("z5", _ps(0, top=-7.0)),
        _row("z6", _ps(0, top=-2.0)),
        _row("z7", _ps(0, top=-2.5)),
        _row("f1", _ps(1, 2, top=2.0)),
        _row("f2", _ps(1, top=-5.0)),
        _row("f3", _ps(1, 2, top=0.0)),
        _row("g1", _ps(0, top=3.0)),
        _row("g2", _ps(0, top=-9.0)),
    ]
    info = {}
    for name in ("e1", "e2", "e3", "e4", "e5", "e6"):
        info[name] = _info(split="eval")
    for name in ("z1", "z2", "z3", "z4", "z5", "z6", "z7"):
        info[name] = _info(answerable=False, split="eval")
    for name in ("f1", "f2", "f3"):
        info[name] = _info(split="fit")
    for name in ("g1", "g2"):
        info[name] = _info(answerable=False, split="fit")
    return rows, info


_THRESHOLDS = {"low_below": -3, "high_above": 1}


def test_the_label_counts_follow_pinakes_strict_thresholds_on_each_side() -> None:
    rows, info = _label_fixture()
    report = scoring.abstention_report(rows, info, _THRESHOLDS)
    assert report["thresholds"] == {"low_below": -3.0, "high_above": 1.0}
    eval_side, fit_side = report["labels"]["eval"], report["labels"]["fit"]
    assert eval_side["answerable"] == {"high": 0, "medium": 1, "low": 5, "unknown": 0}
    assert eval_side["null"] == {"high": 1, "medium": 4, "low": 1, "unknown": 1}
    assert fit_side["answerable"] == {"high": 1, "medium": 1, "low": 1, "unknown": 0}
    assert fit_side["null"] == {"high": 1, "medium": 0, "low": 1, "unknown": 0}


def test_false_confidence_counts_only_null_questions_labelled_high() -> None:
    rows, info = _label_fixture()
    labels = scoring.abstention_report(rows, info, _THRESHOLDS)["labels"]
    # eval null (7): only z1 (2.0) is high; z2 and z3 sit on the thresholds, z4 is unknown
    assert labels["eval"]["false_confidence_null_labelled_high"] == scoring.proportion(1, 7)
    assert labels["fit"]["false_confidence_null_labelled_high"] == scoring.proportion(1, 2)


def test_false_abstain_by_pinakes_definition_needs_a_hit_in_the_first_five_passages() -> None:
    rows, info = _label_fixture()
    labels = scoring.abstention_report(rows, info, _THRESHOLDS)["labels"]
    # eval: e1, e3, e5 and e6 are low and have a gold article within 5 passages (e6's is exactly
    # the 5th); e2's is the 6th passage (outside) and e4 is medium. Over all six answerable.
    assert labels["eval"]["false_abstain_pinakes_definition"] == scoring.proportion(4, 6)
    # fit: f2 is low with a hit; f1 is high, f3 is medium. Over its three answerable questions.
    assert labels["fit"]["false_abstain_pinakes_definition"] == scoring.proportion(1, 3)


def test_false_abstain_on_all_evidence_needs_both_the_find_and_the_low_label() -> None:
    rows, info = _label_fixture()
    labels = scoring.abstention_report(rows, info, _THRESHOLDS)["labels"]
    # eval: only e1 and e5 found every gold article within 10 articles AND are low
    # (e2, e3, e6 are low but not found; e4 is found but medium)
    assert labels["eval"]["false_abstain_all_evidence_found_at_10_docs"] == scoring.proportion(2, 6)
    # fit: f1 found but high, f2 low but only one of two gold, f3 found but medium
    assert labels["fit"]["false_abstain_all_evidence_found_at_10_docs"] == scoring.proportion(0, 3)


# --- filters ------------------------------------------------------------------------------


def _filter_fixture(
    overrides: dict[str, list[list[Any]]] | None = None,
    dataset: Dataset | None = None,
) -> tuple[Any, dict[str, scoring.QueryInfo], list[items.RunItem], dict[str, dict[str, Any]]]:
    """Every derived item gets a row: gold first for a base item, the admitted gold for a variant.
    `overrides` replaces the passages of named rows."""
    dataset = dataset or make_dataset()
    info = scoring.query_info(dataset)
    derived = items.derive_items(dataset)
    rows: dict[str, dict[str, Any]] = {}
    for item in derived:
        gold = sorted(info[item.base_id].gold)
        if item.variant == items.BASE:
            rows[item.id] = _row(item.id, _ps(*gold))
            continue
        admitted = {
            a.index
            for a in dataset.articles
            if admits(item.filters, a, layout.tags_for(a), layout.doc_relpath(a))
        }
        rows[item.id] = _row(
            item.id,
            _ps(*[g for g in gold if g in admitted]),
            variant=item.variant,
            base=item.base_id,
        )
    for item_id, passages in (overrides or {}).items():
        rows[item_id] = {**rows[item_id], "p": passages}
    return dataset, info, derived, rows


def _report(dataset, info, derived, rows, *, fusion_top_k: int = 3) -> dict[str, Any]:
    return scoring.filter_report(
        rows,
        derived,
        dataset,
        info,
        relpath_of=layout.doc_relpath,
        tags_of=layout.tags_for,
        fusion_top_k=fusion_top_k,
    )


# Six f-date questions, each with a known pair of outcomes. Gold sets and the articles each
# window admits: mhr-0000 {0,3} in 0..3, mhr-0001 {1,2,3} in 1..3, mhr-0002 {1,4} in 1..4,
# mhr-0003 {5,8} in 5..8, mhr-0006 {0,4} in 0..4, mhr-0008 {10,11} in 10..11.
_DATE_SCENARIO = {
    "mhr-0000": _ps(0),  # base: half the gold          filtered (default): both -> a win
    "mhr-0001:f-date": _ps(1),  # filtered: one of three      base (default): all three -> a loss
    "mhr-0003": _ps(9),  # base: a miss                 filtered (default): both -> a win
    "mhr-0008": _ps(0),  # base: a miss                 filtered (default): both -> a win
}  # mhr-0002 and mhr-0006 are found by both; no question is missed by both


def test_the_date_variant_pairs_each_question_with_its_unfiltered_twin() -> None:
    entry = _report(*_filter_fixture(_DATE_SCENARIO))[items.F_DATE]
    assert entry["n"] == 6
    # wins: mhr-0000, 0003, 0008. losses: mhr-0001. both: mhr-0002, 0006. neither: none.
    # (3 wins against 1 loss: n = 4, tail = (1 + 4) / 16, doubled to 10 / 16)
    assert entry["paired"] == {
        "filtered_better": 3,
        "filtered_worse": 1,
        "both_found": 2,
        "neither_found": 0,
        "sign_test_p": pytest.approx(0.625),
    }
    # filtered found: mhr-0000, 0002, 0003, 0006, 0008; unfiltered found: mhr-0001, 0002, 0006
    assert entry["all_evidence_at_10_docs_filtered"] == scoring.proportion(5, 6)
    assert entry["all_evidence_at_10_docs_unfiltered"] == scoring.proportion(3, 6)


def _docs_with_gold_at(gold: tuple[int, ...], first_rank: int) -> list[list[Any]]:
    """Passages for distinct articles in which the two gold articles hold ranks first_rank and
    first_rank + 1; every earlier rank is some other article."""
    fillers = [a for a in range(12) if a not in gold][: first_rank - 1]
    return _ps(*fillers, *gold)


def test_the_ten_article_window_of_the_filter_comparison_is_exactly_ten() -> None:
    boundary = {
        # gold {0,3}: unfiltered at ranks 9 and 10 (found); filtered at ranks 10 and 11 (not)
        "mhr-0000": _docs_with_gold_at((0, 3), 9),
        "mhr-0000:f-date": _docs_with_gold_at((0, 3), 10),
        # gold {1,4}: unfiltered at ranks 11 and 12 (not); filtered at ranks 9 and 10 (found)
        "mhr-0002": _docs_with_gold_at((1, 4), 11),
        "mhr-0002:f-date": _docs_with_gold_at((1, 4), 9),
        # gold {0,4}: unfiltered at ranks 10 and 11 (not); the filtered default finds both
        "mhr-0006": _docs_with_gold_at((0, 4), 10),
    }
    report = _report(*_filter_fixture(boundary))[items.F_DATE]
    # (the oversized filtered rows are unsound on purpose; only the pairing is read here)
    assert report["paired"] == {
        "filtered_better": 2,  # mhr-0002, mhr-0006
        "filtered_worse": 1,  # mhr-0000
        "both_found": 3,  # mhr-0001, 0003, 0008
        "neither_found": 0,
        "sign_test_p": 1.0,  # (2, 1): n = 3, tail 4 / 8, doubled
    }
    # coverage within 10 articles, filtered - unfiltered: mhr-0000 1/2 - 1 = -1/2;
    # mhr-0002 1 - 0 = +1 (its unfiltered gold sit at 11 and 12); mhr-0006 1 - 1/2 = +1/2
    assert report["mean_coverage_at_10_docs_change"] == pytest.approx((-0.5 + 1.0 + 0.5) / 6)


def _same_category_dataset(queries: int) -> Dataset:
    toy = make_dataset()
    articles = toy.articles
    # articles n and n + 3 share a category (index % 3), so every query gets a category variant
    return Dataset(
        articles=articles,
        queries=tuple(
            Query(
                f"mhr-{n:04d}",
                n,
                f"q{n}",
                "x",
                "comparison_query",
                tuple(Evidence(articles[i].title, articles[i].url, "a fact") for i in (n, n + 3)),
            )
            for n in range(queries)
        ),
        sha256={},
    )


def test_only_the_first_five_disagreeing_pairs_are_named() -> None:
    dataset = _same_category_dataset(7)
    overrides = {f"mhr-{n:04d}:f-cat-path": _ps(9) for n in range(7)}
    report = _report(*_filter_fixture(overrides, dataset))
    assert report["equivalence_tag_vs_path_prefix"] == {
        "pairs": 7,
        "mismatches": 7,
        "mismatch_ids": [f"mhr-{n:04d}" for n in range(5)],
    }


def test_a_question_both_runs_miss_is_counted_as_neither() -> None:
    both_miss = {"mhr-0003": _ps(9), "mhr-0003:f-date": _ps(6)}  # 6 is admitted but is not gold
    entry = _report(*_filter_fixture(both_miss))[items.F_DATE]
    assert entry["paired"]["neither_found"] == 1
    assert entry["paired"]["both_found"] == 5 and entry["paired"]["filtered_better"] == 0


def test_the_mean_coverage_change_is_filtered_minus_unfiltered() -> None:
    entry = _report(*_filter_fixture(_DATE_SCENARIO))[items.F_DATE]
    # coverage filtered - base: mhr-0000 1 - 1/2 = +1/2; mhr-0001 1/3 - 1 = -2/3;
    # mhr-0003 1 - 0 and mhr-0008 1 - 0 = +1 each; mhr-0002 and 0006 are 0.
    # Mean over six: (1/2 - 2/3 + 1 + 1) / 6 = (11/6) / 6.
    assert entry["mean_coverage_at_10_docs_change"] == pytest.approx(11 / 36)


def test_the_mean_admitted_fraction_is_a_share_of_the_whole_corpus() -> None:
    report = _report(*_filter_fixture(_DATE_SCENARIO))
    # twelve articles. f-date windows admit 4, 3, 4, 4, 5, 2: mean 22 / 6 / 12
    assert report[items.F_DATE]["mean_admitted_fraction"] == pytest.approx(22 / 72)
    # a category holds 4 of the 12 articles, a source 3 of them
    assert report[items.F_CAT_TAG]["mean_admitted_fraction"] == pytest.approx(4 / 12)
    assert report[items.F_CAT_PATH]["mean_admitted_fraction"] == pytest.approx(4 / 12)
    assert report[items.F_SRC_TAG]["mean_admitted_fraction"] == pytest.approx(3 / 12)


def test_the_selectivity_terciles_of_the_date_variant() -> None:
    terciles = _report(*_filter_fixture(_DATE_SCENARIO))[items.F_DATE]["by_selectivity_tercile"]
    assert [t["tercile"] for t in terciles] == ["narrowest", "middle", "widest"]
    assert [t["n"] for t in terciles] == [2, 2, 2]
    # ordered by (admitted share, id): mhr-0008 (2) mhr-0001 (3) | mhr-0000 (4) mhr-0002 (4)
    # | mhr-0003 (4) mhr-0006 (5), each over 12 articles
    narrow, middle, wide = terciles
    assert narrow["mean_admitted_fraction"] == pytest.approx((2 / 12 + 3 / 12) / 2)
    assert middle["mean_admitted_fraction"] == pytest.approx(4 / 12)
    assert wide["mean_admitted_fraction"] == pytest.approx((4 / 12 + 5 / 12) / 2)
    # filtered found: narrow mhr-0008 only; middle both; wide both. Unfiltered found: narrow
    # mhr-0001; middle mhr-0002; wide mhr-0006.
    assert [t["all_evidence_at_10_docs_filtered"] for t in terciles] == [0.5, 1.0, 1.0]
    assert [t["all_evidence_at_10_docs_unfiltered"] for t in terciles] == [0.5, 0.5, 0.5]


@pytest.mark.parametrize(
    ("n", "sizes"),
    [(1, [1]), (2, [1, 1]), (3, [1, 1, 1]), (6, [2, 2, 2]), (7, [2, 2, 3]), (10, [3, 3, 4])],
)
def test_terciles_are_equal_count_thirds_cut_at_n_over_three_and_two_n_over_three(
    n: int, sizes: list[int]
) -> None:
    effects = [(i / 100, f"i{i:02d}", True, False) for i in range(n)]
    names = ["narrowest", "middle", "widest"]
    got = scoring._terciles(effects)
    assert sum(t["n"] for t in got) == n
    # an empty third is left out, so a short list drops the narrowest (then the middle) first
    assert [t["tercile"] for t in got] == names[3 - len(sizes) :]
    assert [t["n"] for t in got] == sizes


def test_terciles_of_nothing_are_nothing() -> None:
    assert scoring._terciles([]) == []


def test_filter_report_stops_when_an_item_claims_a_wrong_admitted_count() -> None:
    dataset, info, derived, rows = _filter_fixture()
    tampered = [
        replace(item, allowed=item.allowed + 1) if item.id == "mhr-0000:f-date" else item
        for item in derived
    ]
    with pytest.raises(RuntimeError, match=r"mhr-0000:f-date: derived 5 admitted, recomputed 4"):
        _report(dataset, info, tampered, rows)


def test_soundness_violations_are_all_counted_but_only_five_ids_are_listed() -> None:
    # article 9 is outside every f-date window, so adding it makes all six rows unsound
    dataset, info, derived, rows = _filter_fixture()
    for item in derived:
        if item.variant == items.F_DATE:
            rows[item.id]["p"] = [*rows[item.id]["p"], [9, 0.1]]
    entry = _report(dataset, info, derived, rows)[items.F_DATE]
    assert entry["soundness_violations"] == 6
    assert entry["soundness_violation_ids"] == [f"mhr-{n:04d}:f-date" for n in (0, 1, 2, 3, 6)]


# --- admission ----------------------------------------------------------------------------


def test_a_path_prefix_must_start_the_path_not_merely_occur_in_it() -> None:
    article = make_dataset().articles[0]
    tags = layout.tags_for(article)
    path = layout.doc_relpath(article)  # docs/sports/the-verge/...
    assert admits(FilterSpec(path_prefix="docs/sports/"), article, tags, path)
    assert not admits(FilterSpec(path_prefix="sports/"), article, tags, path)
    assert not admits(FilterSpec(path_prefix="the-verge/"), article, tags, path)


def test_a_bound_of_zero_is_a_bound_and_an_unset_filter_admits_everything() -> None:
    article = make_dataset().articles[0]
    tags, path = layout.tags_for(article), layout.doc_relpath(article)
    assert not admits(FilterSpec(modified_before=0.0), article, tags, path)
    assert admits(FilterSpec(modified_after=0.0), article, tags, path)
    assert admits(FilterSpec(), article, tags, path) and not FilterSpec().any_set()
    assert FilterSpec(modified_before=0.0).any_set() and FilterSpec(modified_after=0.0).any_set()
    assert FilterSpec(tags=("x",)).any_set() and FilterSpec(path_prefix="docs/").any_set()


# --- the whole run ------------------------------------------------------------------------


def _run_rows(dataset, derived) -> list[dict[str, Any]]:
    """Base rows: an eval-half answerable question is found (top score 2.0), a fit-half one is
    missed (0.5), a null one scores -5.0. Variant rows hold the admitted gold. ms is 100 for a base
    row and 300 for a variant row."""
    info = scoring.query_info(dataset)
    rows = []
    for item in derived:
        known = info[item.base_id]
        gold = sorted(known.gold)
        if item.variant == items.BASE:
            if not known.answerable:
                passages = _ps(9, top=-5.0)
            elif known.split == items.EVAL:
                passages = _ps(*gold, top=2.0)
            else:
                passages = _ps(9, top=0.5)
            rows.append(_row(item.id, passages, ms=100))
            continue
        admitted = {
            a.index
            for a in dataset.articles
            if admits(item.filters, a, layout.tags_for(a), layout.doc_relpath(a))
        }
        passages = _ps(*[g for g in gold if g in admitted])
        rows.append(_row(item.id, passages, variant=item.variant, base=item.base_id, ms=300))
    return rows


def _score(rows, derived, dataset, **overrides) -> dict[str, Any]:
    kwargs = {
        "relpath_of": layout.doc_relpath,
        "tags_of": layout.tags_for,
        "thresholds": None,
        "fusion_top_k": 20,
        "passages_per_search": 20,
    }
    return scoring.score_run(rows, derived, dataset, **{**kwargs, **overrides})


def test_score_run_counts_rows_variants_and_null_questions() -> None:
    dataset = make_dataset()
    derived = items.derive_items(dataset)
    out = _score(_run_rows(dataset, derived), derived, dataset)
    # ten queries: eight answerable, two null. Variants by hand: a category variant pair for
    # mhr-0000, 0002, 0003 (one category each); a source variant for mhr-0006 only; a date
    # variant for every answerable query except mhr-0004 and mhr-0009 (their windows span the corpus)
    assert out["counts"] == {
        "rows": 23,
        "base": 10,
        "answerable": 8,
        "null": 2,
        "by_variant": {"base": 10, "f-cat-tag": 3, "f-cat-path": 3, "f-src-tag": 1, "f-date": 6},
    }


def test_score_run_reports_the_eval_half_and_each_group_on_hand_derived_counts() -> None:
    dataset = make_dataset()
    derived = items.derive_items(dataset)
    answerable = _score(_run_rows(dataset, derived), derived, dataset)["answerable"]

    def found(block: dict[str, Any]) -> tuple[int, int]:
        return block["all_evidence_at_docs"]["10"]["successes"], block["n"]

    # eval-half answerable questions are found, fit-half ones missed (see _run_rows):
    # eval 0000, 0001, 0004, 0008, 0009; fit 0002, 0003, 0006
    assert found(answerable["all"]) == (5, 8)
    assert found(answerable["eval_half"]) == (5, 5)
    assert list(answerable["by_question_type"]) == [
        "comparison_query",
        "inference_query",
        "temporal_query",
    ]
    # comparison: 0000 yes, 0003 no, 0006 no, 0009 yes. inference: 0001, 0004 both eval.
    # temporal: 0002 fit, 0008 eval.
    assert {k: found(v) for k, v in answerable["by_question_type"].items()} == {
        "comparison_query": (2, 4),
        "inference_query": (2, 2),
        "temporal_query": (1, 2),
    }
    # gold of two articles: 0000, 0002, 0003 (5 twice, so {5,8}), 0006, 0008, 0009; of three: 0001;
    # of four: 0004
    assert list(answerable["by_gold_size"]) == ["2", "3", "4"]
    assert {k: found(v) for k, v in answerable["by_gold_size"].items()} == {
        "2": (3, 6),
        "3": (1, 1),
        "4": (1, 1),
    }


def test_score_run_reads_the_abstention_report_from_base_rows_only() -> None:
    dataset = make_dataset()
    derived = items.derive_items(dataset)
    rows = _run_rows(dataset, derived)
    out = _score(rows, derived, dataset, thresholds={"low_below": -1.0, "high_above": 1.0})
    abstention = out["abstention"]
    # the eight answerable top-1 scores are 2.0 (five) and 0.5 (three), both above the two nulls
    assert abstention["auroc_answerable_vs_null"]["all"]["auroc"] == 1.0
    assert abstention["auroc_answerable_vs_null"]["all"]["n_pos"] == 8
    assert abstention["no_results"] == 0
    eval_side, fit_side = abstention["labels"]["eval"], abstention["labels"]["fit"]
    assert eval_side["answerable"] == {"high": 5, "medium": 0, "low": 0, "unknown": 0}
    assert fit_side["answerable"] == {"high": 0, "medium": 3, "low": 0, "unknown": 0}
    assert eval_side["null"] == {"high": 0, "medium": 0, "low": 1, "unknown": 0}  # mhr-0007
    assert fit_side["null"] == {"high": 0, "medium": 0, "low": 1, "unknown": 0}  # mhr-0005


def test_score_run_wires_the_filter_report_and_times_every_row() -> None:
    dataset = make_dataset()
    derived = items.derive_items(dataset)
    out = _score(_run_rows(dataset, derived), derived, dataset)
    assert set(out["filters"]) == {
        "f-cat-tag",
        "f-cat-path",
        "f-src-tag",
        "f-date",
        "equivalence_tag_vs_path_prefix",
    }
    # f-date: mhr-0000, 0001, 0008 are found both ways (eval half); mhr-0002, 0003, 0006 only
    # when filtered (the unfiltered fit-half rows miss). Three wins, no loss: p = 2 * (1/8).
    assert out["filters"]["f-date"]["paired"] == {
        "filtered_better": 3,
        "filtered_worse": 0,
        "both_found": 3,
        "neither_found": 0,
        "sign_test_p": 0.25,
    }
    assert all(
        out["filters"][v]["soundness_violations"] == 0
        for v in ("f-cat-tag", "f-cat-path", "f-src-tag", "f-date")
    )
    # latency is over every row, variants included: 10 rows at 100 ms and 13 at 300 ms
    latency = out["latency_ms"]
    assert latency["min"] == 100.0 and latency["max"] == 300.0
    assert latency["median"] == 300.0  # sorted: 10 x 100 then 13 x 300; position 11 is a 300


def test_score_run_refuses_a_window_one_passage_too_short() -> None:
    dataset = make_dataset()
    derived = items.derive_items(dataset)
    rows = _run_rows(dataset, derived)
    with pytest.raises(ValueError, match="at most 19 passages"):
        _score(rows, derived, dataset, passages_per_search=19)
    assert _score(rows, derived, dataset, passages_per_search=20)["counts"]["base"] == 10
