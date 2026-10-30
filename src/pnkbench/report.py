"""Markdown from a run directory's JSON. Nothing is computed here: every number is read.

The report states what was measured and how; what the dataset *is* and what Pinakes *does* are
linked, not restated (`datasets/multihop-rag/README.md`, Pinakes' own docs).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PINAKES_DOCS = "https://github.com/lucagattoni/pinakes/blob/main/docs"


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100 * value:.{digits}f}%"


def prop(entry: Mapping[str, Any]) -> str:
    """`41.2% (38.9–43.5, n=2255)` from a `scoring.proportion` dict."""
    if entry["value"] is None:
        return "n/a"
    low, high = entry["ci95"]
    return f"{pct(entry['value'])} ({100 * low:.1f}–{100 * high:.1f}, n={entry['n']})"


def num(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _read(path: Path) -> Any | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def render_report(run_dir: Path) -> str:
    header = _read(run_dir / "run.json")
    summary = _read(run_dir / "summary.json")
    calibration = _read(run_dir / "calibration.json")
    harness = _read(run_dir / "harness.json")
    if header is None or summary is None:
        raise FileNotFoundError(f"{run_dir} has no run.json and summary.json; run `pnkbench score`")

    parts = [f"# Run `{header['run_id']}`", _header_section(header)]
    parts.append(_answerable_section(summary["answerable"]))
    parts.append(_abstention_section(summary["abstention"], calibration))
    parts.append(_filters_section(summary["filters"]))
    if harness is not None:
        parts.append(_harness_section(harness))
    parts.append(_latency_section(summary["latency_ms"], header))
    parts.append(_reading_section())
    return "\n\n".join(parts) + "\n"


def _header_section(h: Mapping[str, Any]) -> str:
    p, s = h["pinakes"], h["settings"]
    emb, rr = s["embedding"], s["rerank"]
    commit = f", commit `{p['commit'][:10]}`" if p.get("commit") else ""
    rows = [
        (
            "Pinakes",
            f"{p['version']} from {p['source']}{commit}; installed-code hash `{p['code_sha256'][:16]}`",
        ),
        ("Configuration", f"`{h['config']['name']}`: {h['config']['description']}"),
        (
            "Embedding",
            f"`{emb['model']}@{(emb.get('revision') or 'unpinned')[:10]}`, {emb['dim']} dims, {emb['provider']}",
        ),
        (
            "Reranker",
            f"`{rr['model']}@{(rr.get('revision') or 'unpinned')[:10]}`, {rr['provider']}"
            if rr
            else "none",
        ),
        ("Chunking", ", ".join(f"{k} = {v}" for k, v in s["chunking"].items())),
        ("Retrieval", ", ".join(f"{k} = {v}" for k, v in s["retrieval"].items())),
        (
            "Dataset",
            f"`{h['dataset']['source_repo']}` @ `{h['dataset']['revision'][:10]}`: {h['dataset']['articles']} articles, {h['dataset']['queries']} queries",
        ),
        (
            "Items run",
            f"{h['items']['total']} ({', '.join(f'{v} {n}' for v, n in h['items']['by_variant'].items())})",
        ),
        (
            "Execution",
            f"{h['execution']['workers']} worker(s), {h['execution'].get('wall_seconds', '?')} s wall, {h['execution']['passages_per_search']} passages per search; devices {h['environment']['devices']}",
        ),
        ("Reproducibility", _reproducibility(h.get("reproducibility"))),
        (
            "Benchmark repo",
            f"v{h['benchmark']['version']}, commit `{(h['benchmark']['commit'] or '?')[:10]}`"
            + (", **working tree was dirty**" if h["benchmark"]["dirty"] else ""),
        ),
        (
            "Environment",
            f"Python {h['environment']['python']}, torch {h['environment']['torch']}, {h['environment']['platform']}",
        ),
    ]
    return "## What produced this\n\n" + table(["", ""], [list(r) for r in rows])


def _reproducibility(r: Mapping[str, Any] | None) -> str:
    if not r:
        return "not checked"
    return (
        f"{r['checked']} items searched again in one process: {r['ranking_mismatches']} rankings "
        f"differed, largest score difference {r['max_abs_score_difference']:.2e}"
    )


def _answerable_section(a: Mapping[str, Any]) -> str:
    allm = a["all"]
    rows = []
    for k, label in (
        ("8", "first 8 passages (what `pnk search` returns by default)"),
        ("5", "first 5 passages"),
        ("20", "first 20 passages (the reranker's whole pool)"),
    ):
        rows.append(
            [f"every gold article in the {label}", prop(allm["all_evidence_at_passages"][k])]
        )
    for k in ("3", "5", "10"):
        rows.append(
            [
                f"every gold article in the first {k} distinct articles",
                prop(allm["all_evidence_at_docs"][k]),
            ]
        )
    for k in ("5", "8"):
        rows.append(
            [
                f"*any* gold article in the first {k} passages (Pinakes' harness definition)",
                prop(allm["any_evidence_at_passages"][k]),
            ]
        )
    rows.append(
        [
            "mean share of gold articles found in the first 10 articles",
            pct(allm["coverage_at_docs"]["10"]),
        ]
    )
    rows.append(["MRR of the first gold article", num(allm["mrr_first_gold"])])
    rows.append(["nDCG@10 (articles, binary)", num(allm["ndcg_at_10_docs"])])
    q = allm["distinct_articles_in_20_passages"]
    rows.append(
        [
            "distinct articles among the 20 passages (median, p10–p90)",
            f"{q['median']:.0f} ({q['p10']:.0f}–{q['p90']:.0f})",
        ]
    )

    def by(group: Mapping[str, Any], label: str) -> str:
        body = [
            [
                name,
                str(m["n"]),
                prop(m["all_evidence_at_passages"]["8"]),
                prop(m["all_evidence_at_passages"]["20"]),
                prop(m["any_evidence_at_passages"]["5"]),
                num(m["mrr_first_gold"]),
            ]
            for name, m in group.items()
        ]
        return table(
            [
                label,
                "n",
                "all in first 8 passages",
                "all in first 20 passages",
                "any in first 5 passages",
                "MRR",
            ],
            body,
        )

    return (
        "## Answerable questions\n\n"
        "The primary measure is **all-evidence recall**: a question counts only when *every* gold "
        "article is retrieved. Pinakes' harness counts it when *any* is.\n\n"
        + table(["", "result (95% Wilson interval)"], rows)
        + "\n\n### By question type\n\n"
        + by(a["by_question_type"], "type")
        + "\n\n### By number of gold articles\n\n"
        + by(a["by_gold_size"], "gold articles")
    )


def _abstention_section(ab: Mapping[str, Any], calibration: Mapping[str, Any] | None) -> str:
    def au(entry: Mapping[str, Any]) -> str:
        if entry["auroc"] is None:
            return "n/a"
        low, high = entry["ci95"]
        return f"{entry['auroc']:.3f} ({low:.3f}–{high:.3f}; {entry['n_pos']} vs {entry['n_neg']})"

    rows = [
        ["answerable vs null, all questions", au(ab["auroc_answerable_vs_null"]["all"])],
        ["answerable vs null, eval half", au(ab["auroc_answerable_vs_null"]["eval"])],
        ["answerable vs null, fit half", au(ab["auroc_answerable_vs_null"]["fit"])],
        [
            "null vs answerable *found* (all gold in the first 10 articles)",
            au(ab["auroc_answered_correctly_vs_null"]),
        ],
        [
            "**null without any style tell** vs answerable, all questions",
            au(ab["auroc_null_without_style_tells_vs_answerable"]),
        ],
    ]
    tells = ab["style_tells"]
    tell_rows = [
        [name.replace("-", " "), str(v["answerable"]), str(v["null"])]
        for name, v in tells.items()
        if isinstance(v, dict)
    ] + [["null queries with none of the three", "n/a", str(tells["null_with_no_tell"])]]
    quant = ab["top1_score_quantiles"]
    q_rows = [
        [side]
        + [num(quant[side][k], 2) for k in ("min", "p10", "p25", "median", "p75", "p90", "max")]
        for side in ("answerable", "null")
    ]
    out = (
        "## Abstention: can the score tell a null question from an answerable one?\n\n"
        "The AUROC is the probability that a random answerable question's top rerank score beats a "
        "random null question's (0.5 = no separation, 1.0 = perfect). It needs no thresholds.\n\n"
        + table(["comparison", "AUROC (95% bootstrap interval; n)"], rows)
        + "\n\n**Read the first rows with care.** The null queries differ from the answerable ones in "
        "their wording (see [the dataset notes](datasets/multihop-rag/README.md#caveats)), so a score "
        "gap may be reading style. The bold row keeps only null queries with none of these tells:\n\n"
        + table(["tell", "answerable queries", "null queries"], tell_rows)
        + "\n\nTop-1 rerank score (raw logits):\n\n"
        + table(["", "min", "p10", "p25", "median", "p75", "p90", "max"], q_rows)
    )
    if ab["labels"] is None:
        return (
            out
            + "\n\nNo thresholds were fitted for this run, so Pinakes reports `confidence: unknown`."
        )
    t = ab["thresholds"]
    lab = ab["labels"]
    split = calibration["split"] if calibration else {}
    out += (
        f"\n\n### Confidence labels\n\nThresholds fitted by `pinakes.calibrate` on the **fit half** "
        f"({split.get('fit_null', '?')} null questions): `low_below = {t['low_below']}`, "
        f"`high_above = {t['high_above']}`. The **eval half** was not used in the fit.\n\n"
    )
    rows = []
    for side in ("eval", "fit"):
        s = lab[side]
        rows.append(
            [
                f"{side} half" + (" (held out)" if side == "eval" else " (fitted on)"),
                prop(s["false_confidence_null_labelled_high"]),
                prop(s["false_abstain_pinakes_definition"]),
                prop(s["false_abstain_all_evidence_found_at_10_docs"]),
            ]
        )
    out += table(
        [
            "",
            "false confidence: null labelled high",
            "false abstain (Pinakes' definition: any gold in first 5 passages, labelled low)",
            "false abstain (all gold in first 10 articles, labelled low)",
        ],
        rows,
    )
    if calibration:
        c = calibration["cross_check_against_stored_scores"]
        p = calibration["parity_with_pinakes_labels"]
        out += (
            f"\n\nChecks: the printed thresholds were re-derived from {c['null_scores']} stored scores "
            f"(agree: {c['agrees']}); {p['checked']} searches repeated with the thresholds in the manifest "
            f"gave {p['label_mismatches']} label mismatches against ours."
        )
    return out


def _filters_section(f: Mapping[str, Any]) -> str:
    if not any(k != "equivalence_tag_vs_path_prefix" for k in f):
        return "## Filters\n\nNo filter variants in this run."
    rows = []
    for name, v in f.items():
        if name == "equivalence_tag_vs_path_prefix":
            continue
        p = v["paired"]
        rows.append(
            [
                f"`{name}`",
                str(v["n"]),
                pct(v["mean_admitted_fraction"]),
                str(v["soundness_violations"]),
                str(v["applied_before_retrieval_violations"]),
                prop(v["all_evidence_at_10_docs_unfiltered"]),
                prop(v["all_evidence_at_10_docs_filtered"]),
                f"+{p['filtered_better']} / -{p['filtered_worse']}",
                num(p["sign_test_p"], 4),
            ]
        )
    eq = f["equivalence_tag_vs_path_prefix"]
    return (
        "## Filters (oracle variants)\n\n"
        "Each variant filters on the gold articles' own metadata, so it cannot exclude a gold "
        "article. It tests the filter machinery, not whether a user would have chosen the filter "
        "([how they are derived](datasets/multihop-rag/README.md#filter-variants)).\n\n"
        + table(
            [
                "variant",
                "n",
                "share of corpus admitted",
                "soundness violations",
                "not applied before retrieval",
                "all gold in first 10 articles, unfiltered",
                "same, filtered",
                "filtered better / worse",
                "sign test p",
            ],
            rows,
        )
        + f"\n\nTag and path-prefix spellings of the same category: {eq['pairs']} pairs, "
        f"**{eq['mismatches']} returned different passages or scores** "
        + (f"(e.g. {eq['mismatch_ids']})" if eq["mismatches"] else "")
        + "."
    )


def _harness_section(h: Mapping[str, Any]) -> str:
    m = h["metrics"]
    rows = [
        [k, num(v, 4) if isinstance(v, float) else str(v)] for k, v in m.items() if k != "by_kind"
    ]
    kinds = [[k, num(v, 4)] for k, v in m.get("by_kind", {}).items()]
    return (
        f"## Pinakes' own harness (`python -m pinakes.eval`, k = {h['k']})\n\n"
        f"{h['questions']} questions ({h['by_variant']}). {h['note']}.\n\n"
        + table(["metric", "value"], rows)
        + "\n\nBy kind:\n\n"
        + table(["kind", "recall@k"], kinds)
    )


def _latency_section(q: Mapping[str, float] | None, header: Mapping[str, Any]) -> str:
    if not q:
        return ""
    return (
        "## Latency\n\nPer search, in milliseconds, with "
        f"{header['execution']['workers']} worker(s) sharing one GPU (so under contention): "
        + ", ".join(f"{k} {v:.0f}" for k, v in q.items())
        + "."
    )


def _reading_section() -> str:
    return (
        "## Reading this\n\n"
        "The dataset's own caveats (LLM-generated queries, templated null questions, yes/no "
        "answers) are in [its README](datasets/multihop-rag/README.md#caveats). Pinakes chunks "
        f"plain text by paragraph by design ([DESIGN §4.6]({PINAKES_DOCS}/DESIGN.md)), which is why "
        "twenty passages can come from few articles."
    )


def render_compare(result: Mapping[str, Any]) -> str:
    a, b = result["a"], result["b"]
    rows = [
        [
            name.replace("_", " "),
            pct(v["a_rate"]),
            pct(v["b_rate"]),
            str(v["b_found_a_missed"]),
            str(v["a_found_b_missed"]),
            num(v["sign_test_p"], 4),
        ]
        for name, v in result["answerable"].items()
    ]
    type_rows = [
        [
            t,
            str(v["n"]),
            pct(v["a_rate"]),
            pct(v["b_rate"]),
            str(v["b_found_a_missed"]),
            str(v["a_found_b_missed"]),
            num(v["sign_test_p"], 4),
        ]
        for t, v in result["by_question_type"].items()
    ]
    return (
        f"# `{a['run_id']}` (A) against `{b['run_id']}` (B)\n\n"
        + table(
            ["", "A", "B"],
            [
                ["configuration", a["config"], b["config"]],
                ["chunking metadata", a["chunking_metadata"], b["chunking_metadata"]],
                ["Pinakes", a["pinakes"], b["pinakes"]],
            ],
        )
        + '\n\nPaired on the same answerable questions. "B found, A missed" counts questions; the '
        "sign test is exact, two-sided, on those discordant pairs.\n\n"
        + table(
            ["measure", "A", "B", "B found, A missed", "A found, B missed", "sign test p"], rows
        )
        + "\n\nBy question type, for *every gold article in the first 8 passages*:\n\n"
        + table(
            ["type", "n", "A", "B", "B found, A missed", "A found, B missed", "sign test p"],
            type_rows,
        )
        + f"\n\nTop-1 score AUROC, answerable vs null: A {num(result['null_top1_auroc']['a'])}, "
        f"B {num(result['null_top1_auroc']['b'])}.\n"
    )
