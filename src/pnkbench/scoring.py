"""Every metric, computed from the rows file alone: no index, no model, no network.

**Why a scorer of our own.** Pinakes' harness counts a question as found when *any* expected
document is retrieved (`eval.py`: `hit_rank` is the first rank at which a retrieved path is in
`expect`). MultiHop-RAG questions need two to four distinct articles, so the harness number says
whether Pinakes found one of them. The primary metric here is **all-evidence recall at k**: every
gold article is among the first k documents. The harness's any-of number is reported as well, under
its own name, so the two are never confused.

Two units of k, kept apart because they answer different questions:

* `@k docs`: the first k *distinct articles* in rank order (passages from one article collapse).
* `@k passages`: the first k passages, which is what Pinakes' harness and `final_k` count.

Nothing here assumes a score range: reranker scores are raw logits (calibrate.py measured about
-0.28 for a relevant passage and -7.85 for an irrelevant one).
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pnkbench import items as items_module
from pnkbench.items import F_CAT_PATH, F_CAT_TAG, FILTER_VARIANTS, FilterSpec, RunItem
from pnkbench.multihop_rag import Article, Dataset

K_DOCS = (3, 5, 10)
K_COVERAGE = (5, 10)
K_PASSAGES = (5, 8, 20)
HIGH, MEDIUM, LOW, UNKNOWN = "high", "medium", "low", "unknown"
BOOTSTRAP_RESAMPLES = 500
BOOTSTRAP_SEED = 20261030

Row = Mapping[str, Any]


TELL_LETTER = "asks-for-a-letter"
TELL_OPENING = "opens-with-considering-the-information"
TELL_NO_OUTLET = "names-no-outlet"


@dataclass(frozen=True, slots=True)
class QueryInfo:
    qtype: str
    answerable: bool
    gold: frozenset[int]
    split: str
    tells: tuple[str, ...] = ()
    """Surface features of the wording that separate null queries from the rest.

    Measured on the full dataset (20261030): 47.8% of null queries mention a "letter" against 0% of
    answerable ones, 61.1% contain "Considering the information" against 0.1%, and 70% name no
    news outlet against 0.6%. A score that separates the two groups may be reading these, not the
    content, so the abstention report also gives the AUROC on null queries with none of them.
    """


def style_tells(text: str, outlet_pattern: re.Pattern[str]) -> tuple[str, ...]:
    lowered = text.lower()
    tells = []
    if "letter" in lowered:
        tells.append(TELL_LETTER)
    if "considering the information" in lowered:
        tells.append(TELL_OPENING)
    if not outlet_pattern.search(text):
        tells.append(TELL_NO_OUTLET)
    return tuple(tells)


def query_info(dataset: Dataset) -> dict[str, QueryInfo]:
    """Per base id: question type, the gold article indices, the fit/eval side, style tells."""
    outlets = sorted({a.source for a in dataset.articles}, key=len, reverse=True)
    pattern = re.compile("|".join(rf"\b{re.escape(o)}\b" for o in outlets))
    return {
        q.id: QueryInfo(
            qtype=q.qtype,
            answerable=q.answerable,
            gold=frozenset(a.index for a in dataset.gold(q)),
            split=items_module.split_of(q.id),
            tells=style_tells(q.text, pattern),
        )
        for q in dataset.queries
    }


# --- statistics ---------------------------------------------------------------------------


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """The Wilson score interval for a proportion; `None` for an empty sample."""
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def proportion(successes: int, n: int) -> dict[str, Any]:
    interval = wilson(successes, n)
    return {
        "successes": successes,
        "n": n,
        "value": successes / n if n else None,
        "ci95": list(interval) if interval else None,
    }


def auroc(positives: Sequence[float], negatives: Sequence[float]) -> float | None:
    """P(a positive outscores a negative), ties counting half; `None` if either side is empty."""
    if not positives or not negatives:
        return None
    tagged = sorted([(s, 1) for s in positives] + [(s, 0) for s in negatives])
    rank_sum = 0.0
    i = 0
    while i < len(tagged):
        j = i
        while j < len(tagged) and tagged[j][0] == tagged[i][0]:
            j += 1
        average_rank = (i + 1 + j) / 2  # ranks are 1-based: i+1 .. j
        rank_sum += average_rank * sum(tag for _, tag in tagged[i:j])
        i = j
    n_pos, n_neg = len(positives), len(negatives)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auroc_with_ci(positives: Sequence[float], negatives: Sequence[float]) -> dict[str, Any]:
    point = auroc(positives, negatives)
    if point is None:
        return {"auroc": None, "ci95": None, "n_pos": len(positives), "n_neg": len(negatives)}
    rng = random.Random(BOOTSTRAP_SEED)
    draws = sorted(
        value
        for _ in range(BOOTSTRAP_RESAMPLES)
        if (
            value := auroc(
                rng.choices(positives, k=len(positives)), rng.choices(negatives, k=len(negatives))
            )
        )
        is not None
    )
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(0.975 * (len(draws) - 1))]
    return {"auroc": point, "ci95": [low, high], "n_pos": len(positives), "n_neg": len(negatives)}


def sign_test_p(wins: int, losses: int) -> float | None:
    """Exact two-sided sign test on the discordant pairs; `None` when there are none."""
    n = wins + losses
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


# --- one row ------------------------------------------------------------------------------


def docs_in_rank_order(passages: Sequence[Sequence[Any]]) -> list[int]:
    """Distinct article indices, in the order each first appears among the passages."""
    seen: list[int] = []
    for index, _score in passages:
        if index not in seen:
            seen.append(index)
    return seen


def top1_score(row: Row) -> float | None:
    passages = row["p"]
    return passages[0][1] if passages else None


def found_all(docs: Sequence[int], gold: frozenset[int], k: int) -> bool:
    return gold <= set(docs[:k])


def coverage(docs: Sequence[int], gold: frozenset[int], k: int) -> float:
    return len(gold & set(docs[:k])) / len(gold)


def found_all_passages(row: Row, gold: frozenset[int], k: int) -> bool:
    """Every gold article appears among the first k passages (what `pnk search` shows a user)."""
    return gold <= {index for index, _ in row["p"][:k]}


def found_any_passages(row: Row, gold: frozenset[int], k: int) -> bool:
    """Pinakes' own notion of a hit: any gold article among the first k *passages*."""
    return any(index in gold for index, _ in row["p"][:k])


def first_gold_rank(docs: Sequence[int], gold: frozenset[int]) -> int | None:
    for rank, index in enumerate(docs, start=1):
        if index in gold:
            return rank
    return None


def ndcg(docs: Sequence[int], gold: frozenset[int], k: int) -> float:
    gain = sum(1 / math.log2(rank + 1) for rank, i in enumerate(docs[:k], start=1) if i in gold)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
    return gain / ideal


def confidence_label(score: float | None, low: float, high: float) -> str:
    """Pinakes' rule (`search.confidence_of`) applied to a stored top-1 rerank score."""
    if score is None:
        return UNKNOWN
    if score < low:
        return LOW
    if score > high:
        return HIGH
    return MEDIUM


# --- answerable questions -----------------------------------------------------------------


def answerable_metrics(rows: Sequence[Row], info: Mapping[str, QueryInfo]) -> dict[str, Any]:
    n = len(rows)
    prepared = [
        (docs_in_rank_order(r["p"]), info[r["base_id"]].gold, r) for r in rows if r["p"] is not None
    ]
    reciprocal = [1 / rank if (rank := first_gold_rank(d, g)) else 0.0 for d, g, _ in prepared]
    return {
        "n": n,
        "all_evidence_at_docs": {
            str(k): proportion(sum(found_all(d, g, k) for d, g, _ in prepared), n) for k in K_DOCS
        },
        "coverage_at_docs": {
            str(k): (sum(coverage(d, g, k) for d, g, _ in prepared) / n if n else None)
            for k in K_COVERAGE
        },
        "all_evidence_at_passages": {
            str(k): proportion(sum(found_all_passages(r, g, k) for _, g, r in prepared), n)
            for k in K_PASSAGES
        },
        "distinct_articles_in_20_passages": _quantiles([float(len(d)) for d, _, _ in prepared]),
        "any_evidence_at_passages": {
            str(k): proportion(sum(found_any_passages(r, g, k) for _, g, r in prepared), n)
            for k in K_PASSAGES
        },
        "mrr_first_gold": sum(reciprocal) / n if n else None,
        "ndcg_at_10_docs": sum(ndcg(d, g, 10) for d, g, _ in prepared) / n if n else None,
    }


def grouped(rows: Iterable[Row], key: Callable[[Row], Any]) -> dict[Any, list[Row]]:
    groups: dict[Any, list[Row]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return dict(sorted(groups.items(), key=lambda kv: str(kv[0])))


# --- abstention ---------------------------------------------------------------------------


def abstention_report(
    base_rows: Sequence[Row],
    info: Mapping[str, QueryInfo],
    thresholds: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """How well the top-1 rerank score separates unanswerable from answerable questions.

    The AUROC needs no thresholds, so it is the number to read first. Labels (high/medium/low)
    exist only once thresholds were fitted, and are reported on each side of the fit/eval split:
    the eval half is held out, the fit half shows how optimistic fitting on the scored set is.
    """
    answerable = [r for r in base_rows if info[r["base_id"]].answerable]
    unanswerable = [r for r in base_rows if not info[r["base_id"]].answerable]

    def scores(rows: Iterable[Row]) -> list[float]:
        return [s for r in rows if (s := top1_score(r)) is not None]

    def found(row: Row) -> bool:
        return found_all(docs_in_rank_order(row["p"]), info[row["base_id"]].gold, 10)

    report: dict[str, Any] = {
        "auroc_answerable_vs_null": {
            side: auroc_with_ci(
                scores(r for r in answerable if side == "all" or info[r["base_id"]].split == side),
                scores(
                    r for r in unanswerable if side == "all" or info[r["base_id"]].split == side
                ),
            )
            for side in ("all", items_module.EVAL, items_module.FIT)
        },
        "auroc_answered_correctly_vs_null": auroc_with_ci(
            scores(r for r in answerable if found(r)), scores(unanswerable)
        ),
        "auroc_null_without_style_tells_vs_answerable": auroc_with_ci(
            scores(answerable), scores(r for r in unanswerable if not info[r["base_id"]].tells)
        ),
        "style_tells": {
            tell: {
                "answerable": sum(1 for r in answerable if tell in info[r["base_id"]].tells),
                "null": sum(1 for r in unanswerable if tell in info[r["base_id"]].tells),
            }
            for tell in (TELL_LETTER, TELL_OPENING, TELL_NO_OUTLET)
        }
        | {"null_with_no_tell": sum(1 for r in unanswerable if not info[r["base_id"]].tells)},
        "top1_score_quantiles": {
            "answerable": _quantiles(scores(answerable)),
            "null": _quantiles(scores(unanswerable)),
        },
        "no_results": sum(1 for r in base_rows if not r["p"]),
    }
    if thresholds is None:
        report["labels"] = None
        return report

    low, high = float(thresholds["low_below"]), float(thresholds["high_above"])
    report["thresholds"] = {"low_below": low, "high_above": high}
    report["labels"] = {}
    for side in (items_module.EVAL, items_module.FIT):
        ans = [r for r in answerable if info[r["base_id"]].split == side]
        nul = [r for r in unanswerable if info[r["base_id"]].split == side]
        labelled = {
            "null": _label_counts(nul, low, high),
            "answerable": _label_counts(ans, low, high),
        }
        labelled["false_confidence_null_labelled_high"] = proportion(
            sum(1 for r in nul if confidence_label(top1_score(r), low, high) == HIGH), len(nul)
        )
        # Pinakes' own definition (eval.score_rows): an answerable question that was a hit and
        # was labelled low, over *all* answerable questions. `hit` is any-of over 5 passages.
        labelled["false_abstain_pinakes_definition"] = proportion(
            sum(
                1
                for r in ans
                if found_any_passages(r, info[r["base_id"]].gold, 5)
                and confidence_label(top1_score(r), low, high) == LOW
            ),
            len(ans),
        )
        labelled["false_abstain_all_evidence_found_at_10_docs"] = proportion(
            sum(1 for r in ans if found(r) and confidence_label(top1_score(r), low, high) == LOW),
            len(ans),
        )
        report["labels"][side] = labelled
    return report


def _label_counts(rows: Sequence[Row], low: float, high: float) -> dict[str, int]:
    counts = {HIGH: 0, MEDIUM: 0, LOW: 0, UNKNOWN: 0}
    for r in rows:
        counts[confidence_label(top1_score(r), low, high)] += 1
    return counts


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return ordered[round(fraction * (len(ordered) - 1))]

    return {
        name: at(f)
        for name, f in (
            ("min", 0),
            ("p10", 0.1),
            ("p25", 0.25),
            ("median", 0.5),
            ("p75", 0.75),
            ("p90", 0.9),
            ("max", 1),
        )
    }


# --- filters ------------------------------------------------------------------------------


def admits(spec: FilterSpec, article: Article, tags: Sequence[str], relpath: str) -> bool:
    """Whether Pinakes' filter semantics admit `article`: tags AND together, prefix, mtime window."""
    if any(tag not in tags for tag in spec.tags):
        return False
    if spec.path_prefix and not relpath.startswith(spec.path_prefix):
        return False
    stamp = article.published_at.timestamp()
    if spec.modified_after is not None and stamp < spec.modified_after:
        return False
    return not (spec.modified_before is not None and stamp > spec.modified_before)


def filter_report(
    rows_by_id: Mapping[str, Row],
    items: Sequence[RunItem],
    dataset: Dataset,
    info: Mapping[str, QueryInfo],
    *,
    relpath_of: Callable[[Article], str],
    tags_of: Callable[[Article], Sequence[str]],
    fusion_top_k: int,
) -> dict[str, Any]:
    """Oracle-filter variants against their unfiltered base rows.

    Four checks, each answering a different way a filter can be wrong:

    * **soundness**: every returned article satisfies the filter;
    * **applied before retrieval**: a post-filter returns fewer than `fusion_top_k` candidates
      when the filter is narrow, so `considered` must reach that count whenever at least that
      many articles are admitted;
    * **equivalence**: the tag and the path-prefix spelling of one category admit the same set,
      so they must return identical passages and scores;
    * **effect**: all-evidence recall at 10 docs, paired with the unfiltered question.
    """
    total = len(dataset.articles)
    articles = dataset.articles
    admitted_cache: dict[str, frozenset[int]] = {}

    def admitted(item: RunItem) -> frozenset[int]:
        if item.id not in admitted_cache:
            admitted_cache[item.id] = frozenset(
                a.index for a in articles if admits(item.filters, a, tags_of(a), relpath_of(a))
            )
        return admitted_cache[item.id]

    report: dict[str, Any] = {}
    for variant in FILTER_VARIANTS:
        subset = [i for i in items if i.variant == variant]
        if not subset:
            continue
        unsound = 0
        unsound_ids: list[str] = []
        shortfall = 0
        wins = losses = both = neither = 0
        coverage_delta = 0.0
        recall_filtered = 0
        recall_base = 0
        effects: list[tuple[float, bool, bool]] = []
        for item in subset:
            row = rows_by_id[item.id]
            base = rows_by_id[item.base_id]
            allowed_set = admitted(item)
            if len(allowed_set) != item.allowed:
                raise RuntimeError(
                    f"{item.id}: derived {item.allowed} admitted, recomputed {len(allowed_set)}"
                )
            if any(index not in allowed_set for index, _ in row["p"]):
                unsound += 1
                unsound_ids.append(item.id)
            if len(allowed_set) >= fusion_top_k and row["considered"] < fusion_top_k:
                shortfall += 1
            gold = info[item.base_id].gold
            f_docs, b_docs = docs_in_rank_order(row["p"]), docs_in_rank_order(base["p"])
            f_hit, b_hit = found_all(f_docs, gold, 10), found_all(b_docs, gold, 10)
            recall_filtered += f_hit
            recall_base += b_hit
            wins += f_hit and not b_hit
            losses += b_hit and not f_hit
            both += f_hit and b_hit
            neither += (not f_hit) and (not b_hit)
            coverage_delta += coverage(f_docs, gold, 10) - coverage(b_docs, gold, 10)
            effects.append((item.allowed / total, f_hit, b_hit))
        n = len(subset)
        entry: dict[str, Any] = {
            "n": n,
            "mean_admitted_fraction": sum(i.allowed for i in subset) / n / total,
            "soundness_violations": unsound,
            "soundness_violation_ids": unsound_ids[:5],
            "applied_before_retrieval_violations": shortfall,
            "all_evidence_at_10_docs_filtered": proportion(recall_filtered, n),
            "all_evidence_at_10_docs_unfiltered": proportion(recall_base, n),
            "paired": {
                "filtered_better": wins,
                "filtered_worse": losses,
                "both_found": both,
                "neither_found": neither,
                "sign_test_p": sign_test_p(wins, losses),
            },
            "mean_coverage_at_10_docs_change": coverage_delta / n,
            "by_selectivity_tercile": _terciles(effects),
        }
        report[variant] = entry

    pair = [i for i in items if i.variant == F_CAT_TAG]
    mismatched = [
        i.base_id
        for i in pair
        if (path_id := items_module.item_id(i.base_id, F_CAT_PATH)) in rows_by_id
        and rows_by_id[i.id]["p"] != rows_by_id[path_id]["p"]
    ]
    report["equivalence_tag_vs_path_prefix"] = {
        "pairs": len(pair),
        "mismatches": len(mismatched),
        "mismatch_ids": mismatched[:5],
    }
    return report


def _terciles(effects: Sequence[tuple[float, bool, bool]]) -> list[dict[str, Any]]:
    ordered = sorted(effects)
    n = len(ordered)
    out: list[dict[str, Any]] = []
    for name, lo, hi in (
        ("narrowest", 0, n // 3),
        ("middle", n // 3, 2 * n // 3),
        ("widest", 2 * n // 3, n),
    ):
        group = ordered[lo:hi]
        if not group:
            continue
        out.append(
            {
                "tercile": name,
                "n": len(group),
                "mean_admitted_fraction": sum(f for f, _, _ in group) / len(group),
                "all_evidence_at_10_docs_filtered": sum(h for _, h, _ in group) / len(group),
                "all_evidence_at_10_docs_unfiltered": sum(b for _, _, b in group) / len(group),
            }
        )
    return out


# --- the whole run ------------------------------------------------------------------------


def score_run(
    rows: Sequence[Row],
    items: Sequence[RunItem],
    dataset: Dataset,
    *,
    relpath_of: Callable[[Article], str],
    tags_of: Callable[[Article], Sequence[str]],
    thresholds: Mapping[str, Any] | None,
    fusion_top_k: int,
) -> dict[str, Any]:
    info = query_info(dataset)
    rows_by_id = {r["id"]: r for r in rows}
    base = [r for r in rows if r["variant"] == items_module.BASE]
    answerable = [r for r in base if info[r["base_id"]].answerable]
    return {
        "counts": {
            "rows": len(rows),
            "base": len(base),
            "answerable": len(answerable),
            "null": len(base) - len(answerable),
            "by_variant": {
                v: sum(1 for r in rows if r["variant"] == v)
                for v in (items_module.BASE, *FILTER_VARIANTS)
            },
        },
        "answerable": {
            "all": answerable_metrics(answerable, info),
            "eval_half": answerable_metrics(
                [r for r in answerable if info[r["base_id"]].split == items_module.EVAL], info
            ),
            "by_question_type": {
                qtype: answerable_metrics(group, info)
                for qtype, group in grouped(answerable, lambda r: info[r["base_id"]].qtype).items()
            },
            "by_gold_size": {
                str(size): answerable_metrics(group, info)
                for size, group in grouped(
                    answerable, lambda r: len(info[r["base_id"]].gold)
                ).items()
            },
        },
        "abstention": abstention_report(base, info, thresholds),
        "filters": filter_report(
            rows_by_id,
            items,
            dataset,
            info,
            relpath_of=relpath_of,
            tags_of=tags_of,
            fusion_top_k=fusion_top_k,
        ),
        "latency_ms": _quantiles([float(r["ms"]) for r in rows]),
    }
