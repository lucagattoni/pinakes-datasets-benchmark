from __future__ import annotations

import itertools
import math
import random
from typing import Any

import pytest
from toy import make_dataset

from pnkbench import items, layout, scoring

# --- statistics ---------------------------------------------------------------------------


def test_wilson_known_values() -> None:
    low, high = scoring.wilson(50, 100)  # type: ignore[misc]
    assert (round(low, 4), round(high, 4)) == (0.4038, 0.5962)
    low, high = scoring.wilson(0, 10)  # type: ignore[misc]
    assert low == 0.0 and round(high, 4) == 0.2775
    low, high = scoring.wilson(10, 10)  # type: ignore[misc]
    assert round(low, 4) == 0.7225 and high == 1.0
    assert scoring.wilson(0, 0) is None


def test_auroc_hand_computed_cases() -> None:
    assert scoring.auroc([3, 4], [1, 2]) == 1.0
    assert scoring.auroc([1, 2], [3, 4]) == 0.0
    assert scoring.auroc([5, 5], [5, 5]) == 0.5
    assert scoring.auroc([1, 2, 3], [2]) == 0.5  # lose, tie, win
    assert scoring.auroc([3, 4], [1, 2, 3]) == pytest.approx(5.5 / 6)
    assert scoring.auroc([], [1]) is None and scoring.auroc([1], []) is None


def test_auroc_matches_brute_force_on_random_data_with_ties() -> None:
    rng = random.Random(7)
    for _ in range(25):
        pos = [rng.randint(0, 6) for _ in range(rng.randint(1, 12))]
        neg = [rng.randint(0, 6) for _ in range(rng.randint(1, 12))]
        brute = sum(
            1.0 if p > n else 0.5 if p == n else 0.0 for p, n in itertools.product(pos, neg)
        )
        assert scoring.auroc(pos, neg) == pytest.approx(brute / (len(pos) * len(neg)))


def test_auroc_ci_brackets_the_point_estimate_and_is_reproducible() -> None:
    rng = random.Random(1)
    pos = [rng.gauss(1, 1) for _ in range(80)]
    neg = [rng.gauss(0, 1) for _ in range(60)]
    first = scoring.auroc_with_ci(pos, neg)
    assert first == scoring.auroc_with_ci(pos, neg)
    assert first["ci95"][0] < first["auroc"] < first["ci95"][1]


def test_sign_test_exact_values() -> None:
    assert scoring.sign_test_p(0, 0) is None
    assert scoring.sign_test_p(5, 0) == pytest.approx(2 / 32)
    assert scoring.sign_test_p(3, 3) == 1.0
    assert scoring.sign_test_p(10, 2) == pytest.approx(2 * 79 / 4096)
    assert scoring.sign_test_p(2, 10) == scoring.sign_test_p(10, 2)


# --- one row ------------------------------------------------------------------------------


def row(
    item_id: str,
    passages: list[list[Any]],
    *,
    variant: str = "base",
    base: str | None = None,
    conf: str = "unknown",
    considered: int = 20,
    ms: int = 5,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "base_id": base or item_id.split(":")[0],
        "variant": variant,
        "conf": conf,
        "considered": considered,
        "p": passages,
        "ms": ms,
    }


def test_docs_in_rank_order_keeps_first_appearance() -> None:
    assert scoring.docs_in_rank_order([[5, 0.1], [5, 0.0], [2, 0.0], [5, -1.0]]) == [5, 2]
    assert scoring.docs_in_rank_order([]) == []


def test_all_evidence_needs_every_gold_article() -> None:
    gold = frozenset({1, 3})
    docs = [5, 1, 9, 3]
    assert not scoring.found_all(docs, gold, 3)
    assert scoring.found_all(docs, gold, 4)
    assert scoring.coverage(docs, gold, 3) == 0.5
    assert scoring.first_gold_rank(docs, gold) == 2
    assert scoring.first_gold_rank([7, 8], gold) is None


def test_passage_and_doc_counting_differ_when_one_article_fills_the_slots() -> None:
    gold = frozenset({1, 3})
    passages = [[1, 0.9], [1, 0.8], [1, 0.7], [3, 0.6]]
    r = row("a", passages)
    assert scoring.found_all(scoring.docs_in_rank_order(passages), gold, 2)  # 2 distinct articles
    assert not scoring.found_all_passages(r, gold, 2)  # but only 2 passages, both article 1
    assert scoring.found_all_passages(r, gold, 4)
    assert scoring.found_any_passages(r, gold, 1)


def test_ndcg_matches_hand_calculation() -> None:
    gold = frozenset({1, 3})
    gain = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert scoring.ndcg([5, 1, 9, 3], gold, 4) == pytest.approx(gain / ideal)
    assert scoring.ndcg([1, 3], gold, 10) == pytest.approx(1.0)
    assert scoring.ndcg([7], gold, 10) == 0.0


def test_confidence_label_follows_pinakes_strict_inequalities() -> None:
    assert scoring.confidence_label(-5.0, -4.0, 1.0) == "low"
    assert scoring.confidence_label(-4.0, -4.0, 1.0) == "medium"  # not below
    assert scoring.confidence_label(1.0, -4.0, 1.0) == "medium"  # not above
    assert scoring.confidence_label(1.01, -4.0, 1.0) == "high"
    assert scoring.confidence_label(None, -4.0, 1.0) == "unknown"


# --- aggregates over the toy dataset ------------------------------------------------------


def test_answerable_metrics_on_hand_built_rows() -> None:
    info = scoring.query_info(make_dataset())
    rows = [
        row("mhr-0000", [[0, 3.0], [3, 2.0]]),  # gold {0,3}: found at once
        row("mhr-0003", [[5, 3.0], [1, 2.0], [8, 1.0]]),  # gold {5,8}: both within 3 passages
        row("mhr-0008", [[2, 3.0], [10, 2.0]]),  # gold {10,11}: only one found
    ]
    m = scoring.answerable_metrics(rows, info)
    assert m["n"] == 3
    assert m["all_evidence_at_passages"]["8"]["successes"] == 2
    assert m["all_evidence_at_passages"]["5"]["successes"] == 2
    assert m["any_evidence_at_passages"]["5"]["successes"] == 3
    assert m["all_evidence_at_docs"]["3"]["successes"] == 2
    # first-gold ranks are 1, 1 and 2
    assert m["mrr_first_gold"] == pytest.approx((1 + 1 + 0.5) / 3)


def test_abstention_report_separates_and_labels_by_split() -> None:
    dataset = make_dataset()
    info = scoring.query_info(dataset)
    rows = []
    for q in dataset.queries:
        score = -8.0 if not q.answerable else 2.0
        gold_first = [[min(info[q.id].gold), score]] if q.answerable else [[0, score]]
        rows.append(row(q.id, gold_first))
    report = scoring.abstention_report(rows, info, {"low_below": -3.0, "high_above": 1.0})
    assert report["auroc_answerable_vs_null"]["all"]["auroc"] == 1.0
    for side in ("eval", "fit"):
        labels = report["labels"][side]
        assert labels["false_confidence_null_labelled_high"]["successes"] == 0
        assert sum(labels["null"].values()) == labels["false_confidence_null_labelled_high"]["n"]
    assert scoring.abstention_report(rows, info, None)["labels"] is None


def test_false_abstain_uses_pinakes_definition_over_all_answerable() -> None:
    dataset = make_dataset()
    info = scoring.query_info(dataset)
    answerable = [q for q in dataset.queries if q.answerable]
    # every answerable question is a hit (its first gold article, rank 1) but scored low
    rows = [row(q.id, [[min(info[q.id].gold), -9.0]]) for q in dataset.queries if q.answerable]
    rows += [row(q.id, [[0, -9.0]]) for q in dataset.queries if not q.answerable]
    report = scoring.abstention_report(rows, info, {"low_below": -3.0, "high_above": 1.0})
    for side in ("eval", "fit"):
        sample = report["labels"][side]["false_abstain_pinakes_definition"]
        in_side = [q for q in answerable if info[q.id].split == side]
        assert sample["n"] == len(in_side) and sample["successes"] == len(in_side)


# --- filters ------------------------------------------------------------------------------


def _filter_setup(fusion_top_k: int = 3):
    dataset = make_dataset()
    info = scoring.query_info(dataset)
    derived = items.derive_items(dataset)
    return dataset, info, derived, fusion_top_k


def _all_rows(dataset, derived, *, mutate=None, considered=3):
    """Base rows that find their gold; variant rows restricted to what the filter admits."""
    info = scoring.query_info(dataset)
    out = {}
    for item in derived:
        gold = sorted(info[item.base_id].gold)
        if item.variant == items.BASE:
            out[item.id] = row(item.id, [[g, 1.0] for g in gold], considered=considered)
        else:
            admitted = [
                a.index
                for a in dataset.articles
                if scoring.admits(item.filters, a, layout.tags_for(a), layout.doc_relpath(a))
            ]
            out[item.id] = row(
                item.id,
                [[g, 1.0] for g in gold if g in admitted],
                variant=item.variant,
                base=item.base_id,
                considered=considered,
            )
    if mutate:
        mutate(out)
    return out


def test_filter_report_is_clean_on_a_correct_run() -> None:
    dataset, info, derived, k = _filter_setup()
    rows = _all_rows(dataset, derived)
    report = scoring.filter_report(
        rows,
        derived,
        dataset,
        info,
        relpath_of=layout.doc_relpath,
        tags_of=layout.tags_for,
        fusion_top_k=k,
    )
    for variant in items.FILTER_VARIANTS:
        if variant in report:
            assert report[variant]["soundness_violations"] == 0
            assert report[variant]["applied_before_retrieval_violations"] == 0
    assert report["equivalence_tag_vs_path_prefix"]["mismatches"] == 0
    assert report["equivalence_tag_vs_path_prefix"]["pairs"] > 0


def test_filter_report_catches_an_article_outside_the_filter() -> None:
    dataset, info, derived, k = _filter_setup()

    def leak(rows):  # a `sports` filter that returns a technology article
        rows["mhr-0000:f-cat-tag"]["p"].append([1, 0.1])

    rows = _all_rows(dataset, derived, mutate=leak)
    report = scoring.filter_report(
        rows,
        derived,
        dataset,
        info,
        relpath_of=layout.doc_relpath,
        tags_of=layout.tags_for,
        fusion_top_k=k,
    )
    assert report[items.F_CAT_TAG]["soundness_violations"] == 1
    assert report[items.F_CAT_TAG]["soundness_violation_ids"] == ["mhr-0000:f-cat-tag"]


def test_filter_report_catches_a_post_filter_shortfall() -> None:
    dataset, info, derived, k = _filter_setup(fusion_top_k=3)

    def starve(rows):  # fewer candidates than the pool though 4 articles are admitted
        rows["mhr-0000:f-cat-tag"]["considered"] = 1

    rows = _all_rows(dataset, derived, mutate=starve)
    report = scoring.filter_report(
        rows,
        derived,
        dataset,
        info,
        relpath_of=layout.doc_relpath,
        tags_of=layout.tags_for,
        fusion_top_k=k,
    )
    assert report[items.F_CAT_TAG]["applied_before_retrieval_violations"] == 1


def test_filter_report_catches_tag_and_path_disagreement() -> None:
    dataset, info, derived, k = _filter_setup()

    def diverge(rows):
        rows["mhr-0000:f-cat-path"]["p"] = [[0, 0.5]]

    rows = _all_rows(dataset, derived, mutate=diverge)
    report = scoring.filter_report(
        rows,
        derived,
        dataset,
        info,
        relpath_of=layout.doc_relpath,
        tags_of=layout.tags_for,
        fusion_top_k=k,
    )
    eq = report["equivalence_tag_vs_path_prefix"]
    assert eq["mismatches"] == 1 and eq["mismatch_ids"] == ["mhr-0000"]


def test_admits_matches_pinakes_filter_semantics() -> None:
    article = make_dataset().articles[0]  # sports, The Verge, 2023-10-01 09:30 UTC
    path, tags = layout.doc_relpath(article), layout.tags_for(article)
    f = items.FilterSpec
    assert scoring.admits(f(tags=("sports",)), article, tags, path)
    assert not scoring.admits(f(tags=("sports", "business")), article, tags, path)  # tags AND
    assert scoring.admits(f(path_prefix="docs/sports/"), article, tags, path)
    assert not scoring.admits(f(path_prefix="docs/business/"), article, tags, path)
    stamp = article.published_at.timestamp()
    assert scoring.admits(
        f(modified_after=stamp, modified_before=stamp), article, tags, path
    )  # inclusive
    assert not scoring.admits(f(modified_after=stamp + 1), article, tags, path)
    assert not scoring.admits(f(modified_before=stamp - 1), article, tags, path)


def test_style_tells_flag_the_template_and_the_missing_outlet() -> None:
    import re

    pattern = re.compile(r"\bThe Verge\b|\bTechCrunch\b")
    plain = "According to TechCrunch, which company raised money?"
    assert scoring.style_tells(plain, pattern) == ()
    templated = "Considering the information from an article, which single letter is it?"
    assert scoring.style_tells(templated, pattern) == (
        scoring.TELL_LETTER,
        scoring.TELL_OPENING,
        scoring.TELL_NO_OUTLET,
    )
    assert scoring.style_tells("Does The Verge agree?", pattern) == ()


def test_abstention_report_gives_the_style_controlled_auroc_and_counts() -> None:
    dataset = make_dataset()
    info = scoring.query_info(dataset)
    rows = [row(q.id, [[0, -8.0 if not q.answerable else 2.0]]) for q in dataset.queries]
    report = scoring.abstention_report(rows, info, None)
    assert report["style_tells"]["null_with_no_tell"] == 0  # toy queries name no outlet
    assert report["style_tells"][scoring.TELL_NO_OUTLET]["null"] == 2
    controlled = report["auroc_null_without_style_tells_vs_answerable"]
    assert controlled["auroc"] is None and controlled["n_neg"] == 0


# --- gaps a mutation pass found (each test below killed a mutant that had survived) ---------


def test_the_passage_window_is_exactly_k() -> None:
    gold = frozenset({1, 3})
    r = row("a", [[1, 0.9], [2, 0.8], [3, 0.7]])
    assert not scoring.found_all_passages(
        r, gold, 2
    )  # the third passage is outside a window of two
    assert scoring.found_all_passages(r, gold, 3)


def test_ndcg_ideal_is_limited_by_k_when_there_are_more_gold_articles_than_slots() -> None:
    assert scoring.ndcg([1, 2, 3], frozenset({1, 2, 3}), 2) == pytest.approx(1.0)


def test_top1_is_the_first_passage_as_pinakes_ranks_it() -> None:
    assert scoring.top1_score(row("a", [[1, 5.0], [2, 9.0], [3, 1.0]])) == 5.0
    assert scoring.top1_score(row("a", [])) is None


def test_the_shortfall_check_applies_when_exactly_as_many_articles_are_admitted_as_the_pool() -> (
    None
):
    dataset, info, derived, _ = _filter_setup()

    def starve(rows):  # `sports` admits exactly 4 articles
        rows["mhr-0000:f-cat-tag"]["considered"] = 3

    rows = _all_rows(dataset, derived, mutate=starve, considered=4)
    report = scoring.filter_report(
        rows,
        derived,
        dataset,
        info,
        relpath_of=layout.doc_relpath,
        tags_of=layout.tags_for,
        fusion_top_k=4,
    )
    assert report[items.F_CAT_TAG]["applied_before_retrieval_violations"] == 1


def test_paired_filter_comparison_counts_each_direction_separately() -> None:
    dataset, info, derived, k = _filter_setup()

    def base_misses(rows):
        rows["mhr-0000"]["p"] = [[9, 0.1]]  # article 9 is nobody's gold

    def filtered_misses(rows):
        rows["mhr-0000:f-cat-tag"]["p"] = [[9, 0.1]]

    for mutate, better, worse in ((base_misses, 1, 0), (filtered_misses, 0, 1)):
        rows = _all_rows(dataset, derived, mutate=mutate)
        paired = scoring.filter_report(
            rows,
            derived,
            dataset,
            info,
            relpath_of=layout.doc_relpath,
            tags_of=layout.tags_for,
            fusion_top_k=k,
        )[items.F_CAT_TAG]["paired"]
        assert (paired["filtered_better"], paired["filtered_worse"]) == (better, worse)


def test_false_abstain_counts_a_hit_anywhere_in_the_first_five_passages() -> None:
    dataset = make_dataset()
    info = scoring.query_info(dataset)
    rows = []
    for q in dataset.queries:
        if (
            q.answerable
        ):  # three filler passages from an article that is nobody's gold, then the gold
            rows.append(row(q.id, [[9, -9.0], [9, -9.1], [9, -9.2], [min(info[q.id].gold), -9.3]]))
        else:
            rows.append(row(q.id, [[9, -9.0]]))
    report = scoring.abstention_report(rows, info, {"low_below": -3.0, "high_above": 1.0})
    for side in ("eval", "fit"):
        sample = report["labels"][side]["false_abstain_pinakes_definition"]
        assert sample["n"] > 0 and sample["successes"] == sample["n"]


# --- fixes made after an independent review -------------------------------------------------


def test_quantiles_interpolate_between_ranks() -> None:
    q = scoring._quantiles([1.0, 2.0, 3.0, 4.0])
    assert q["median"] == pytest.approx(2.5)  # round-half-to-even used to give 3.0
    assert q["min"] == 1.0 and q["max"] == 4.0
    assert q["p25"] == pytest.approx(1.75) and q["p75"] == pytest.approx(3.25)
    assert scoring._quantiles([1.0, 2.0])["median"] == pytest.approx(1.5)
    assert scoring._quantiles([7.0])["p90"] == 7.0
    assert scoring._quantiles([]) is None
    eleven = scoring._quantiles([float(n) for n in range(11)])
    assert eleven["p10"] == pytest.approx(1.0) and eleven["p90"] == pytest.approx(9.0)


def test_terciles_are_cut_by_id_inside_a_tie_never_by_outcome() -> None:
    """Sorting on the outcome as well put every miss into the narrowest third of a tied group."""
    # nine items with one admitted fraction; found in the middle three by id, missed elsewhere
    effects = [(0.28, f"i{n}", n in (3, 4, 5), n in (3, 4, 5)) for n in range(9)]
    terciles = {t["tercile"]: t for t in scoring._terciles(effects)}
    assert [t["n"] for t in terciles.values()] == [3, 3, 3]
    assert terciles["narrowest"]["all_evidence_at_10_docs_filtered"] == 0.0
    assert terciles["middle"]["all_evidence_at_10_docs_filtered"] == 1.0
    assert terciles["widest"]["all_evidence_at_10_docs_filtered"] == 0.0
    shuffled = [effects[i] for i in (7, 2, 5, 0, 8, 3, 1, 6, 4)]
    assert scoring._terciles(shuffled) == scoring._terciles(effects)


def test_terciles_order_by_admitted_fraction_first() -> None:
    effects = [(0.9, "a", True, False), (0.1, "b", False, False), (0.5, "c", True, True)]
    names = {t["tercile"]: t["mean_admitted_fraction"] for t in scoring._terciles(effects)}
    assert names == {"narrowest": 0.1, "middle": 0.5, "widest": 0.9}


def _toy_rows_and_items():
    dataset, _, derived, _ = _filter_setup()
    return dataset, derived, list(_all_rows(dataset, derived, considered=20).values())


def test_score_run_refuses_rows_too_short_for_the_longest_window() -> None:
    dataset, derived, rows = _toy_rows_and_items()
    kwargs = {
        "relpath_of": layout.doc_relpath,
        "tags_of": layout.tags_for,
        "thresholds": None,
        "fusion_top_k": 20,
    }
    with pytest.raises(ValueError, match="at most 8 passages"):
        scoring.score_run(rows, derived, dataset, passages_per_search=8, **kwargs)
    assert (
        scoring.score_run(rows, derived, dataset, passages_per_search=20, **kwargs)["counts"][
            "base"
        ]
        == 10
    )


def test_score_run_counts_what_the_toy_dataset_contains() -> None:
    dataset, derived, rows = _toy_rows_and_items()
    out = scoring.score_run(
        rows,
        derived,
        dataset,
        relpath_of=layout.doc_relpath,
        tags_of=layout.tags_for,
        thresholds=None,
        fusion_top_k=20,
        passages_per_search=20,
    )
    counts = out["counts"]
    assert (counts["base"], counts["answerable"], counts["null"]) == (10, 8, 2)
    assert counts["rows"] == len(derived)
    # q4 (articles 0, 4, 8, 11) and q9 (0, 11) span the first and the last day, so a publication
    # window around their gold admits the whole corpus and no f-date variant exists for them:
    # 8 answerable queries - 2 = 6. The other 6 answerable queries each get one.
    assert counts["by_variant"]["base"] == 10 and counts["by_variant"]["f-date"] == 6
    assert out["answerable"]["all"]["n"] == 8
    eval_answerable = [
        q for q in dataset.queries if q.answerable and items.split_of(q.id) == items.EVAL
    ]
    # the pinned split table in test_layout_and_items puts 0, 1, 4, 8 and 9 in the eval half (7 is null)
    assert out["answerable"]["eval_half"]["n"] == len(eval_answerable) == 5
    assert {k: v["n"] for k, v in out["answerable"]["by_question_type"].items()} == {
        "comparison_query": 4,
        "inference_query": 2,
        "temporal_query": 2,
    }
    # gold sizes by hand: q1 {1,2,3} is 3; q4 {0,4,8,11} is 4; q3 lists article 5 twice so {5,8} is 2;
    # the other five are pairs
    assert {k: v["n"] for k, v in out["answerable"]["by_gold_size"].items()} == {
        "2": 6,
        "3": 1,
        "4": 1,
    }
