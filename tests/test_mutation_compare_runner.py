"""Tests written against the mutants of `compare.py` and `runner.py` that an independent review
found still surviving the suite.

The comparison tests build two runs over the ten toy queries in which every query has a known
search pattern, so each of the five paired measures can be read off by hand. The patterns and the
resulting counts are laid out in the comments above `_RUN_A` / `_RUN_B`.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from toy import make_dataset

from pnkbench import compare, pinakes_api, runner, scoring
from pnkbench.admission import FilterSpec
from pnkbench.items import RunItem


def _ps(*indices: int, top: float = 9.0) -> list[list[Any]]:
    return [[index, top - 0.01 * rank] for rank, index in enumerate(indices)]


def _row(item_id: str, passages: list[list[Any]]) -> dict[str, Any]:
    return {
        "id": item_id,
        "base_id": item_id,
        "variant": "base",
        "conf": "unknown",
        "considered": 20,
        "p": passages,
        "ms": 1,
    }


def _write_run(
    directory: Path, name: str, rows: list[dict[str, Any]], *, max_tokens: int | None = None
) -> Path:
    run_dir = directory / name
    run_dir.mkdir()
    header = {
        "run_id": name,
        "status": "complete",
        "dataset": {"sha256": {"corpus.json": "s"}},
        "execution": {"passages_per_search": 20},
        "config": {"name": f"config-{name}", "chunking_metadata": "off"},
        "pinakes": {"version": "0.32.8", "source": "pypi", "code_sha256": "0123456789abcdef" * 4},
    }
    if max_tokens is not None:
        header["config"]["chunking_max_tokens"] = max_tokens
    (run_dir / "run.json").write_text(json.dumps(header))
    (run_dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run_dir


# Article 9 is nobody's gold in the toy dataset, so it is the filler. These patterns are for a
# gold set of two articles (g1, g2); the other sizes only use FIRST and MISS.
#
#   pattern   passages                                     m1  m2  m3  m4  m5
#   FIRST     g1 g2                                         1   1   1   1   1
#   MISS      9                                             0   0   0   0   0
#   ONE       g1 9                                          0   0   0   0   1
#   G6        9 9 9 9 9 g1 g2   (gold: passages 6, 7)       1   1   1   1   0
#   P10       ten 9s, g1 g2     (gold: passages 11, 12)     0   1   1   1   0
#   D3        3 other articles, g1 g2  (gold: articles 4, 5) 1   1   1   1   1
#   D6        6 other articles, g1 g2  (gold: articles 7, 8) 1   1   0   1   0
#   D8        8 other articles, g1 g2  (gold: articles 9, 10) 0  1   0   1   0
#
# m1 all gold in the first 8 passages, m2 in the first 20 passages, m3 in the first 5 articles,
# m4 in the first 10 articles, m5 any gold in the first 5 passages.
_FILLERS = [9, 6, 7, 10, 5, 2, 1, 4, 8, 3, 0, 11]


def _passages(pattern: str, gold: frozenset[int], top: float) -> list[list[Any]]:
    g = sorted(gold)

    def others(count: int) -> list[int]:
        return [f for f in _FILLERS if f not in gold][:count]

    shapes = {
        "FIRST": g,
        "MISS": [9],
        "ONE": [g[0], 9],
        "G6": [9] * 5 + g,
        "P10": [9] * 10 + g,
        "D3": others(3) + g,
        "D4": others(4) + g,
        "D6": others(6) + g,
        "D7": others(7) + g,
        "D8": others(8) + g,
        "D9": others(9) + g,
        "P18": [9] * 18 + g,
        "P19": [9] * 19 + g,
    }
    return _ps(*shapes[pattern], top=top)


#  query     A (pattern, top-1)   B (pattern, top-1)     m1  m2  m3  m4  m5   (a, b)
_RUN_A = {
    "mhr-0000": ("FIRST", 5.0),  # q0 comparison, gold {0,3}
    "mhr-0001": ("FIRST", 6.0),  # q1 inference,  gold {1,2,3}
    "mhr-0002": ("G6", 4.0),  # q2 temporal,   gold {1,4}
    "mhr-0003": ("P10", 3.0),  # q3 comparison, gold {5,8}
    "mhr-0004": ("MISS", 1.0),  # q4 inference,  gold {0,4,8,11}
    "mhr-0006": ("D6", 2.0),  # q6 comparison, gold {0,4}
    "mhr-0008": ("ONE", 7.0),  # q8 temporal,   gold {10,11}
    "mhr-0009": ("MISS", 8.0),  # q9 comparison, gold {0,11}
}
_RUN_B = {
    "mhr-0000": ("G6", 9.0),
    "mhr-0001": ("FIRST", 6.0),
    "mhr-0002": ("P10", 9.0),
    "mhr-0003": ("D6", 9.0),
    "mhr-0004": ("MISS", 1.0),
    "mhr-0006": ("D3", 9.0),
    "mhr-0008": ("FIRST", 9.0),
    "mhr-0009": ("D8", 9.0),
}
# (a, b) per measure, query by query, in the order q0 q1 q2 q3 q4 q6 q8 q9:
#   m1 (11) (11) (10) (01) (00) (11) (01) (00)   both 3, only A 1, only B 2, neither 2
#   m2 (11) (11) (11) (11) (00) (11) (01) (01)   both 5, only A 0, only B 2, neither 1
#   m3 (11) (11) (11) (10) (00) (01) (01) (00)   both 3, only A 1, only B 2, neither 2
#   m4 (11) (11) (11) (11) (00) (11) (01) (01)   both 5, only A 0, only B 2, neither 1
#   m5 (10) (11) (00) (00) (00) (01) (11) (00)   both 2, only A 1, only B 1, neither 4
_NULL_TOPS = {"mhr-0005": 3.5, "mhr-0007": 6.5}


def _rows_of(run: dict[str, tuple[str, float]]) -> list[dict[str, Any]]:
    info = scoring.query_info(make_dataset())
    rows = [
        _row(query_id, _passages(pattern, info[query_id].gold, top))
        for query_id, (pattern, top) in run.items()
    ]
    rows += [_row(query_id, _ps(9, top=top)) for query_id, top in _NULL_TOPS.items()]
    return sorted(rows, key=lambda r: r["id"])


@pytest.fixture
def compared(tmp_path: Path) -> dict[str, Any]:
    a = _write_run(tmp_path, "a", _rows_of(_RUN_A))
    b = _write_run(tmp_path, "b", _rows_of(_RUN_B))
    return compare.compare_runs(a, b, make_dataset())


def test_the_two_rates_of_each_measure_are_each_runs_own_share_of_questions_found(
    compared: dict[str, Any],
) -> None:
    got = {name: (m["a_rate"], m["b_rate"]) for name, m in compared["answerable"].items()}
    assert got == {
        "all_evidence_in_first_8_passages": (4 / 8, 5 / 8),
        "all_evidence_in_first_20_passages": (5 / 8, 7 / 8),
        "all_evidence_in_first_5_docs": (4 / 8, 5 / 8),
        "all_evidence_in_first_10_docs": (5 / 8, 7 / 8),
        "any_evidence_in_first_5_passages": (3 / 8, 3 / 8),
    }


def test_the_four_pair_counts_of_each_measure(compared: dict[str, Any]) -> None:
    def counts(m: dict[str, Any]) -> tuple[int, int, int, int, int]:
        return (
            m["n"],
            m["b_found_a_missed"],
            m["a_found_b_missed"],
            m["both_found"],
            m["neither_found"],
        )

    assert {name: counts(m) for name, m in compared["answerable"].items()} == {
        # name:                               n, only B, only A, both, neither
        "all_evidence_in_first_8_passages": (8, 2, 1, 3, 2),
        "all_evidence_in_first_20_passages": (8, 2, 0, 5, 1),
        "all_evidence_in_first_5_docs": (8, 2, 1, 3, 2),
        "all_evidence_in_first_10_docs": (8, 2, 0, 5, 1),
        "any_evidence_in_first_5_passages": (8, 1, 1, 2, 4),
    }


def test_the_sign_test_is_on_the_questions_only_one_run_found(compared: dict[str, Any]) -> None:
    p = {name: m["sign_test_p"] for name, m in compared["answerable"].items()}
    # (only B, only A) = (2, 1): n = 3, tail (1 + 3) / 8, doubled and capped at 1
    assert p["all_evidence_in_first_8_passages"] == 1.0
    assert p["all_evidence_in_first_5_docs"] == 1.0
    # (2, 0): n = 2, tail 1 / 4, doubled
    assert p["all_evidence_in_first_20_passages"] == pytest.approx(0.5)
    assert p["all_evidence_in_first_10_docs"] == pytest.approx(0.5)
    # (1, 1): n = 2, tail (1 + 2) / 4, doubled and capped at 1
    assert p["any_evidence_in_first_5_passages"] == 1.0


def test_the_question_type_breakdown_uses_the_eight_passage_measure(
    compared: dict[str, Any],
) -> None:
    by_type = compared["by_question_type"]
    assert list(by_type) == ["comparison_query", "inference_query", "temporal_query"]
    # m1 per query: comparison q0 (11) q3 (01) q6 (11) q9 (00); inference q1 (11) q4 (00);
    # temporal q2 (10) q8 (01)
    comparison = by_type["comparison_query"]
    assert (comparison["n"], comparison["a_rate"], comparison["b_rate"]) == (4, 2 / 4, 3 / 4)
    assert (comparison["b_found_a_missed"], comparison["a_found_b_missed"]) == (1, 0)
    assert (comparison["both_found"], comparison["neither_found"]) == (2, 1)
    inference = by_type["inference_query"]
    assert (inference["n"], inference["a_rate"], inference["b_rate"]) == (2, 0.5, 0.5)
    assert inference["sign_test_p"] is None  # no discordant pair
    temporal = by_type["temporal_query"]
    assert (temporal["n"], temporal["a_rate"], temporal["b_rate"]) == (2, 0.5, 0.5)
    assert (temporal["b_found_a_missed"], temporal["a_found_b_missed"]) == (1, 1)
    assert (temporal["both_found"], temporal["neither_found"]) == (0, 0)


def test_each_runs_auroc_is_answerable_against_null_on_its_own_top_scores(
    compared: dict[str, Any],
) -> None:
    # nulls score 3.5 and 6.5 in both runs. Run A's answerable top-1 scores are
    # 5, 6, 4, 3, 1, 2, 7, 8: wins 1 + 1 + 1 + 0 + 0 + 0 + 2 + 2 = 7 of 16.
    # Run B's are 9 (six times), 6 and 1: wins 12 + 1 + 0 = 13 of 16.
    assert compared["null_top1_auroc"] == {"a": 7 / 16, "b": 13 / 16}


def test_the_counts_of_questions_and_identical_searches(compared: dict[str, Any]) -> None:
    assert compared["questions"] == {"a": 10, "b": 10, "compared": 10}
    # the same passages and scores: q1 and q4 (same pattern, same top score) and both nulls
    assert compared["identical_searches"] == {"of": 10, "same_passages_and_scores": 4}


def test_each_run_is_described_with_a_twelve_character_code_hash(tmp_path: Path) -> None:
    a = _write_run(tmp_path, "a", _rows_of(_RUN_A))
    b = _write_run(tmp_path, "b", _rows_of(_RUN_B), max_tokens=400)
    result = compare.compare_runs(a, b, make_dataset())
    # a header with no chunking_max_tokens (an older run, or the default) describes it as None
    assert result["a"] == {
        "run_id": "a",
        "config": "config-a",
        "passages_per_search": 20,
        "chunking_metadata": "off",
        "chunking_max_tokens": None,
        "pinakes": "0.32.8 (pypi, code 0123456789ab)",
    }
    assert result["b"]["run_id"] == "b" and result["b"]["config"] == "config-b"
    assert result["b"]["chunking_max_tokens"] == 400


_METRICS = [
    "all_evidence_in_first_8_passages",
    "all_evidence_in_first_20_passages",
    "all_evidence_in_first_5_docs",
    "all_evidence_in_first_10_docs",
    "any_evidence_in_first_5_passages",
]
# What each pattern scores on the five measures above, for a gold set of two articles:
_EDGES = {
    "D3": (1, 1, 1, 1, 1),  # gold are passages 4 and 5 (articles 4 and 5): inside every window
    "D4": (1, 1, 0, 1, 1),  # passages 5 and 6: articles 5 and 6 miss the five-article window
    "G6": (1, 1, 1, 1, 0),  # passages 6 and 7: first gold just outside the five-passage window
    "D6": (1, 1, 0, 1, 0),  # passages 7 and 8: last gold exactly at the eight-passage edge
    "D7": (0, 1, 0, 1, 0),  # passages 8 and 9: one past the eight-passage window
    "D8": (0, 1, 0, 1, 0),  # passages 9 and 10: last gold exactly at the ten-article edge
    "D9": (0, 1, 0, 0, 0),  # passages 10 and 11: one past the ten-article window
    "P18": (0, 1, 1, 1, 0),  # passages 19 and 20: last gold exactly at the twenty-passage edge
    "P19": (0, 0, 1, 1, 0),  # passages 20 and 21: one past it
}


@pytest.mark.parametrize(("pattern", "hits"), list(_EDGES.items()))
def test_each_paired_measure_stops_exactly_where_its_name_says(
    tmp_path: Path, pattern: str, hits: tuple[int, ...]
) -> None:
    dataset = make_dataset()
    info = scoring.query_info(dataset)

    def rows(first: str) -> list[dict[str, Any]]:
        out = []
        for q in dataset.queries:
            if q.id == "mhr-0000":
                passages = _passages(first, info[q.id].gold, 5.0)
            else:
                passages = _ps(9, top=1.0)  # a miss, whatever the query
            out.append(_row(q.id, passages))
        return out

    # run A shows `pattern` for one question and misses the rest; run B misses everything
    result = compare.compare_runs(
        _write_run(tmp_path, "a", rows(pattern)), _write_run(tmp_path, "b", rows("MISS")), dataset
    )
    for name, hit in zip(_METRICS, hits, strict=True):
        measure = result["answerable"][name]
        assert (measure["a_found_b_missed"], measure["b_found_a_missed"]) == (hit, 0), name
        assert measure["both_found"] == 0 and measure["a_rate"] == hit / 8, name


# --- runner -------------------------------------------------------------------------------


def test_a_filter_survives_the_trip_through_the_payload_with_every_field_in_place() -> None:
    spec = FilterSpec(
        tags=("sports", "source-the-verge"),
        path_prefix="docs/sports/",
        modified_after=1.5,
        modified_before=2.5,
    )
    payload = runner.to_payload(RunItem("i:f-date", "i", "f-date", "a question", spec, 3))
    assert payload["after"] == 1.5 and payload["before"] == 2.5
    assert runner._spec(payload) == spec


def test_spec_reads_after_as_the_lower_bound_and_before_as_the_upper_one() -> None:
    raw = {"tags": ["a", "b"], "path_prefix": "docs/x/", "after": 10.0, "before": 20.0}
    spec = runner._spec(raw)
    assert spec.tags == ("a", "b")
    assert spec.path_prefix == "docs/x/"
    assert (spec.modified_after, spec.modified_before) == (10.0, 20.0)
    assert (
        runner._spec({"tags": [], "path_prefix": None, "after": None, "before": None})
        == FilterSpec()
    )


def test_run_one_searches_with_the_whole_filter_and_records_what_came_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def fake_search(kb: Any, query: str, spec: FilterSpec, *, limit: int) -> pinakes_api.Outcome:
        seen.update(kb=kb, query=query, spec=spec, limit=limit)
        return pinakes_api.Outcome(
            passages=(("docs/a.txt", 1.5), ("docs/b.txt", None)), confidence="high", considered=17
        )

    ticks = iter([100.0, 101.5])
    monkeypatch.setattr(pinakes_api, "search_kb", fake_search)
    monkeypatch.setattr(runner, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    handle = object()
    payload = {
        "id": "x:f-date",
        "base_id": "x",
        "variant": "f-date",
        "query": "the question",
        "tags": ["sports"],
        "path_prefix": "docs/sports/",
        "after": 5.0,
        "before": 9.0,
    }
    row = runner.run_one(handle, {"docs/a.txt": 4, "docs/b.txt": 7}, payload, 12)
    assert seen["kb"] is handle and seen["query"] == "the question" and seen["limit"] == 12
    assert seen["spec"] == FilterSpec(("sports",), "docs/sports/", 5.0, 9.0)
    # 17 candidates were considered though only two passages came back; 1.5 s is 1500 ms
    assert row == {
        "id": "x:f-date",
        "base_id": "x",
        "variant": "f-date",
        "conf": "high",
        "considered": 17,
        "p": [[4, 1.5], [7, None]],
        "ms": 1500,
    }
    assert isinstance(row["ms"], int)


def test_read_rows_returns_a_complete_last_row_that_lacks_only_its_newline(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(_row("a", [])) + "\n" + json.dumps(_row("b", [])))
    assert [r["id"] for r in runner.read_rows(path)] == ["a", "b"]


def _run_item(n: int) -> RunItem:
    return RunItem(f"r{n:02d}", f"r{n:02d}", "base", "a question", FilterSpec(), 12)


def test_finalize_names_only_the_first_three_missing_and_the_first_three_unknown_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(_row(f"u{n}", [])) + "\n" for n in (5, 4, 3, 2, 1)))
    expected = [_run_item(n) for n in range(1, 6)]
    with pytest.raises(RuntimeError) as caught:
        runner.finalize(path, expected)
    assert str(caught.value) == (
        f"{path} does not match the item list: 5 missing (e.g. ['r01', 'r02', 'r03']), "
        "5 unknown (e.g. ['u1', 'u2', 'u3'])"
    )


class _Pools:
    """Stands in for a multiprocessing context: each batch comes back at once as a tiny row per
    item, and what the pool was asked to do is recorded."""

    def __init__(self) -> None:
        self.created: list[tuple[int, tuple[Any, ...]]] = []
        self.batches: list[list[str]] = []
        self.timeouts: list[float] = []

    def Pool(self, workers: int, initializer: Any, initargs: tuple[Any, ...]) -> _Pools:
        self.created.append((workers, initargs))
        return self

    def __enter__(self) -> _Pools:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def imap_unordered(self, fn: Any, batches: list[list[dict[str, Any]]]) -> _Pools:
        self.batches = [[payload["id"] for payload in batch] for batch in batches]
        self._answers = iter(
            [[{"id": p["id"], "p": [[1, 2.5]]} for p in batch] for batch in batches]
        )
        return self

    def next(self, timeout: float) -> list[dict[str, Any]]:
        self.timeouts.append(timeout)
        return next(self._answers)


def test_run_items_works_in_batches_of_eight_and_a_resume_runs_only_what_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import multiprocessing

    pools = _Pools()
    monkeypatch.setattr(multiprocessing, "get_context", lambda name: pools)
    items_ = [_run_item(n) for n in range(21)]
    out = tmp_path / "not" / "yet" / "rows.jsonl"  # the folders are made, parents included
    messages: list[str] = []

    def run() -> dict[str, Any]:
        return runner.run_items(
            tmp_path,
            items_,
            {"docs/a.txt": 1},
            out,
            workers=3,
            threads=None,
            limit=20,
            log=messages.append,
        )

    first = run()
    assert (first["ran"], first["skipped"]) == (21, 0)
    assert [len(b) for b in pools.batches] == [8, 8, 5]
    assert pools.created == [(3, (str(tmp_path), {"docs/a.txt": 1}, 20, None))]
    assert pools.timeouts == [runner.STALL_SECONDS] * 3
    assert messages[0] == "0 rows already present; 21 to run with 3 worker(s)"
    lines = out.read_text().split("\n")
    assert lines[0] == '{"id":"r00","p":[[1,2.5]]}' and lines[-1] == ""
    assert len(lines) == 22

    # nothing missing: no pool is even created
    second = run()
    assert (second["ran"], second["skipped"]) == (0, 21)
    assert len(pools.created) == 1

    # the last nine rows are lost: only they run again, in batches of eight and one
    out.write_text("\n".join(lines[:12]) + "\n")
    third = run()
    assert (third["ran"], third["skipped"]) == (9, 12)
    assert pools.batches == [[f"r{n:02d}" for n in range(12, 20)], ["r20"]]
    assert len(out.read_text().splitlines()) == 12 + 9


def test_run_items_reports_progress_every_200_rows_and_at_the_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import multiprocessing

    pools = _Pools()
    monkeypatch.setattr(multiprocessing, "get_context", lambda name: pools)
    clock = iter([10.0])  # the start; every later reading is 100 s on

    def monotonic() -> float:
        return next(clock, 110.0)

    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=monotonic))
    messages: list[str] = []
    result = runner.run_items(
        tmp_path,
        [_run_item(n) for n in range(450)],
        {},
        tmp_path / "rows.jsonl",
        workers=3,
        threads=None,
        limit=20,
        log=messages.append,
    )
    # batches of eight finish at 8, 16, ... 448 and then 450; a line is due when the count passes
    # a multiple of 200 (200 and 400 here) and at the very end. The rate is rows over 100 s, and
    # the time left is the rows still to do over that rate.
    assert messages == [
        "0 rows already present; 450 to run with 3 worker(s)",
        "200/450 rows, 2.00/s, ~2 min left",  # 250 rows at 2/s is 125 s
        "400/450 rows, 4.00/s, ~0 min left",  # 50 rows at 4/s is 12.5 s
        "450/450 rows, 4.50/s, ~0 min left",
    ]
    assert result == {"ran": 450, "skipped": 0, "seconds": 100.0}
    assert runner.STALL_SECONDS == 300  # "five minutes is generous"
