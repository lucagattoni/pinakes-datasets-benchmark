"""Two runs, question by question.

An aggregate hides which questions moved, and a change that lifts some and sinks others can leave
the headline almost unchanged. So the comparison is paired on the base query id: for each metric it
counts the questions B found that A missed, the reverse, and the ties, and gives the exact sign
test on the discordant pairs. The sign test is the statistic Pinakes itself holds changes to
(`sign_test` in its eval tooling, p < 0.05).

The comparison refuses to pair runs that do not measure the same thing: a different dataset
revision, or a different set of questions, makes a paired difference meaningless. It does *not*
refuse two different Pinakes builds or configurations, because that is what it is for; it states
them at the top.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from pnkbench import items as items_module
from pnkbench import runner, scoring
from pnkbench.multihop_rag import Dataset

Row = Mapping[str, Any]


class CompareError(Exception):
    pass


def _load(run_dir: Path) -> tuple[dict[str, Any], dict[str, Row]]:
    header = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if header.get("status") != "complete":
        raise CompareError(f"{run_dir.name} is not a complete run")
    rows = {r["id"]: r for r in runner.read_rows(run_dir / "rows.jsonl")}
    return header, rows


def _metrics() -> dict[str, Callable[[Row, scoring.QueryInfo], bool]]:
    def docs(row: Row) -> list[int]:
        return scoring.docs_in_rank_order(row["p"])

    return {
        "all_evidence_in_first_8_passages": lambda r, q: scoring.found_all_passages(r, q.gold, 8),
        "all_evidence_in_first_20_passages": lambda r, q: scoring.found_all_passages(r, q.gold, 20),
        "all_evidence_in_first_5_docs": lambda r, q: scoring.found_all(docs(r), q.gold, 5),
        "all_evidence_in_first_10_docs": lambda r, q: scoring.found_all(docs(r), q.gold, 10),
        "any_evidence_in_first_5_passages": lambda r, q: scoring.found_any_passages(r, q.gold, 5),
    }


def _paired(
    ids: Sequence[str],
    rows_a: Mapping[str, Row],
    rows_b: Mapping[str, Row],
    info: Mapping[str, scoring.QueryInfo],
    predicate: Callable[[Row, scoring.QueryInfo], bool],
) -> dict[str, Any]:
    only_a = only_b = both = neither = 0
    for base_id in ids:
        a = predicate(rows_a[base_id], info[base_id])
        b = predicate(rows_b[base_id], info[base_id])
        only_a += a and not b
        only_b += b and not a
        both += a and b
        neither += (not a) and (not b)
    n = len(ids)
    return {
        "n": n,
        "a_rate": (only_a + both) / n if n else None,
        "b_rate": (only_b + both) / n if n else None,
        "b_found_a_missed": only_b,
        "a_found_b_missed": only_a,
        "both_found": both,
        "neither_found": neither,
        "sign_test_p": scoring.sign_test_p(only_b, only_a),
    }


def chunk_spans(index_db: Path) -> dict[str, frozenset[tuple[int, int]]]:
    """Each active document's chunk spans, `(char_start, char_end)`, read from a built index."""
    connection = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT d.path, c.char_start, c.char_end FROM chunks c "
            "JOIN documents d ON d.id = c.doc_id WHERE d.state = 'active'"
        ).fetchall()
    finally:
        connection.close()
    spans: dict[str, set[tuple[int, int]]] = {}
    for path, start, end in rows:
        spans.setdefault(path, set()).add((start, end))
    return {path: frozenset(found) for path, found in spans.items()}


def articles_chunked_differently(
    index_a: Path, index_b: Path, path_index: Mapping[str, int]
) -> frozenset[int]:
    """Articles whose chunk boundaries differ between two built indexes.

    A configuration that changes `max_tokens` re-splits some articles; a paired comparison of the
    two runs is then confounded for the questions that depend on those articles. This names them,
    from the indexes themselves, so the comparison can also be read without them.
    """
    spans_a, spans_b = chunk_spans(index_a), chunk_spans(index_b)
    if set(spans_a) != set(spans_b):
        raise CompareError("the two indexes hold different documents")
    return frozenset(path_index[path] for path in spans_a if spans_a[path] != spans_b[path])


def compare_runs(
    dir_a: Path,
    dir_b: Path,
    dataset: Dataset,
    *,
    set_aside: Collection[int] = (),
) -> dict[str, Any]:
    header_a, rows_a = _load(dir_a)
    header_b, rows_b = _load(dir_b)
    if header_a["dataset"]["sha256"] != header_b["dataset"]["sha256"]:
        raise CompareError("the runs used different dataset files")
    limits = (
        header_a["execution"]["passages_per_search"],
        header_b["execution"]["passages_per_search"],
    )
    if limits[0] != limits[1]:
        raise CompareError(
            f"the runs searched for different numbers of passages ({limits[0]} and {limits[1]}), "
            "so every measure counted in passages would compare unlike things"
        )
    info = scoring.query_info(dataset)
    base_a = {i for i, r in rows_a.items() if r["variant"] == items_module.BASE}
    base_b = {i for i, r in rows_b.items() if r["variant"] == items_module.BASE}
    common = sorted(base_a & base_b)
    if not common:
        raise CompareError("the runs share no question, so there is nothing to pair")
    answerable = [i for i in common if info[i].answerable]
    null = [i for i in common if not info[i].answerable]

    result: dict[str, Any] = {
        "a": _describe(header_a),
        "b": _describe(header_b),
        "questions": {"a": len(base_a), "b": len(base_b), "compared": len(common)},
        "identical_searches": {
            "of": len(common),
            "same_passages_and_scores": sum(1 for i in common if rows_a[i]["p"] == rows_b[i]["p"]),
        },
        "answerable": {
            name: _paired(answerable, rows_a, rows_b, info, fn) for name, fn in _metrics().items()
        },
        "by_question_type": {},
        "null_top1_auroc": {
            side: scoring.auroc(
                [s for i in answerable if (s := scoring.top1_score(rows[i])) is not None],
                [s for i in null if (s := scoring.top1_score(rows[i])) is not None],
            )
            for side, rows in (("a", rows_a), ("b", rows_b))
        },
    }
    if set_aside:
        aside = frozenset(set_aside)
        kept = [i for i in answerable if not info[i].gold & aside]
        result["without_set_aside_articles"] = {
            "articles": len(aside),
            "questions_kept": len(kept),
            "questions_set_aside": len(answerable) - len(kept),
            "answerable": {
                name: _paired(kept, rows_a, rows_b, info, fn) for name, fn in _metrics().items()
            },
        }
    primary = _metrics()["all_evidence_in_first_8_passages"]
    for qtype in sorted({info[i].qtype for i in answerable}):
        ids = [i for i in answerable if info[i].qtype == qtype]
        result["by_question_type"][qtype] = _paired(ids, rows_a, rows_b, info, primary)
    return result


def _describe(header: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": header["run_id"],
        "config": header["config"]["name"],
        "passages_per_search": header["execution"]["passages_per_search"],
        "chunking_metadata": header["config"]["chunking_metadata"],
        "chunking_max_tokens": header["config"].get("chunking_max_tokens"),
        "pinakes": f"{header['pinakes']['version']} ({header['pinakes']['source']}, "
        f"code {header['pinakes']['code_sha256'][:12]})",
    }
