"""Round two of the mutation review: tests for code added or changed after the first review.

Fast, no models. Each section names the behaviour it pins. Expected values come from hand
arithmetic or from the module docstrings; nothing is copied from a run of the code under test.
The toy dataset facts used throughout (gold sets, fit/eval halves) are those of `tests/toy.py`:
gold mhr-0000 {0,3}, 0001 {1,2,3}, 0002 {1,4}, 0003 {5,8}, 0004 {0,4,8,11}, 0006 {0,4},
0008 {10,11}, 0009 {0,11}; mhr-0005 and mhr-0007 are null.
"""

from __future__ import annotations

import json
import multiprocessing
import shutil
import sqlite3
import sys
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from toy import make_dataset

from pnkbench import (
    calibration,
    compare,
    items,
    kb,
    layout,
    paths,
    pinakes_api,
    pipeline,
    report,
    runner,
    scoring,
)
from pnkbench import config as config_module
from pnkbench.admission import FilterSpec
from pnkbench.config import Config, ModelPin, Models
from pnkbench.items import RunItem
from pnkbench.multihop_rag import Dataset

REAL_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _ps(*indices: int, top: float = 9.0) -> list[list[Any]]:
    return [[index, top - 0.01 * rank] for rank, index in enumerate(indices)]


def _row(item_id: str, passages: list[list[Any]], variant: str = "base") -> dict[str, Any]:
    return {
        "id": item_id,
        "base_id": item_id.split(":")[0],
        "variant": variant,
        "conf": "unknown",
        "considered": 20,
        "p": passages,
        "ms": 1,
    }


# --- items.add_variant: every gold article is checked, not only the first ----------------------


def test_a_variant_that_excludes_a_later_gold_article_is_refused_by_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = layout.doc_relpath
    # mhr-0000's gold are articles 0 and 3; only the second one ends up outside docs/sports/
    monkeypatch.setattr(
        layout, "doc_relpath", lambda a: "docs/elsewhere/x.txt" if a.index == 3 else real(a)
    )
    with pytest.raises(
        ValueError, match=r"mhr-0000: the f-cat-path filter excludes gold article 3"
    ):
        items.derive_items(make_dataset())


def test_a_variant_that_excludes_a_later_gold_article_is_refused_by_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = layout.tags_for
    monkeypatch.setattr(layout, "tags_for", lambda a: ("unrelated",) if a.index == 3 else real(a))
    with pytest.raises(ValueError, match=r"mhr-0000: the f-cat-tag filter excludes gold article 3"):
        items.derive_items(make_dataset())


# --- compare.chunk_spans and articles_chunked_differently ---------------------------------------


def _index(
    directory: Path,
    name: str,
    spans: dict[str, list[tuple[int, int]]],
    *,
    deleted: tuple[str, ...] = (),
) -> Path:
    """A SQLite file with the two tables the comparison reads from a built Pinakes index."""
    path = directory / name
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE documents (id TEXT, path TEXT, state TEXT)")
    db.execute("CREATE TABLE chunks (doc_id TEXT, char_start INT, char_end INT)")
    for number, (doc, found) in enumerate(spans.items()):
        state = "deleted" if doc in deleted else "active"
        db.execute("INSERT INTO documents VALUES (?, ?, ?)", (f"id{number}", doc, state))
        db.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?)", [(f"id{number}", s, e) for s, e in found]
        )
    db.commit()
    db.close()
    return path


_PATH_INDEX = {"docs/a.txt": 7, "docs/b.txt": 9}


def test_chunk_spans_are_each_documents_start_and_end_pairs(tmp_path: Path) -> None:
    db = _index(tmp_path, "a.db", {"docs/a.txt": [(0, 10), (10, 20)], "docs/b.txt": [(3, 7)]})
    assert compare.chunk_spans(db) == {
        "docs/a.txt": frozenset({(0, 10), (10, 20)}),
        "docs/b.txt": frozenset({(3, 7)}),
    }


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ([(0, 10)], [(0, 8)]),  # only the end differs
        ([(0, 10)], [(2, 10)]),  # only the start differs
        ([(0, 10), (10, 20)], [(0, 20), (10, 10)]),  # same starts, same ends, paired otherwise
    ],
)
def test_an_article_is_chunked_differently_when_any_boundary_pair_differs(
    tmp_path: Path, before: list[tuple[int, int]], after: list[tuple[int, int]]
) -> None:
    one = _index(tmp_path, "one.db", {"docs/a.txt": before, "docs/b.txt": [(0, 5)]})
    two = _index(tmp_path, "two.db", {"docs/a.txt": after, "docs/b.txt": [(0, 5)]})
    assert compare.articles_chunked_differently(one, two, _PATH_INDEX) == frozenset({7})
    assert compare.articles_chunked_differently(two, one, _PATH_INDEX) == frozenset({7})
    assert compare.articles_chunked_differently(one, one, _PATH_INDEX) == frozenset()


def test_two_indexes_with_as_many_documents_but_different_ones_are_refused(tmp_path: Path) -> None:
    one = _index(tmp_path, "one.db", {"docs/a.txt": [(0, 5)], "docs/b.txt": [(0, 5)]})
    two = _index(tmp_path, "two.db", {"docs/a.txt": [(0, 5)], "docs/c.txt": [(0, 5)]})
    with pytest.raises(compare.CompareError, match="different documents"):
        compare.articles_chunked_differently(one, two, {**_PATH_INDEX, "docs/c.txt": 11})


def test_an_index_is_opened_read_only_so_a_missing_one_is_an_error_not_a_new_file(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "nope.db"
    with pytest.raises(sqlite3.OperationalError):
        compare.chunk_spans(missing)
    assert not missing.exists()


def test_reading_an_index_leaves_it_byte_for_byte_as_it_was(tmp_path: Path) -> None:
    db = _index(tmp_path, "a.db", {"docs/a.txt": [(0, 10)]})
    before = db.read_bytes()
    compare.chunk_spans(db)
    assert db.read_bytes() == before


# --- compare_runs: the reading without the set-aside articles ----------------------------------


def _write_run(
    directory: Path,
    name: str,
    rows: list[dict[str, Any]],
    *,
    config_name: str | None = None,
    max_tokens: int | str | None = "absent",
) -> Path:
    run_dir = directory / name
    run_dir.mkdir()
    config: dict[str, Any] = {"name": config_name or f"config-{name}", "chunking_metadata": "off"}
    if max_tokens != "absent":
        config["chunking_max_tokens"] = max_tokens
    header = {
        "run_id": name,
        "status": "complete",
        "dataset": {"sha256": {"corpus.json": "s"}},
        "execution": {"passages_per_search": 20},
        "config": config,
        "pinakes": {"version": "0.32.8", "source": "pypi", "code_sha256": "c" * 64},
    }
    (run_dir / "run.json").write_text(json.dumps(header))
    (run_dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run_dir


def _toy_rows(plan: dict[str, str]) -> list[dict[str, Any]]:
    """One base row per toy query: 'first' puts the gold articles first, 'miss' returns one
    article that is nobody's gold. Null questions always miss."""
    info = scoring.query_info(make_dataset())
    rows = []
    for query_id, known in info.items():
        if known.answerable and plan[query_id] == "first":
            rows.append(_row(query_id, _ps(*sorted(known.gold), top=5.0)))
        else:
            rows.append(_row(query_id, _ps(9, top=1.0)))
    return rows


_PLAN_A = {
    "mhr-0000": "miss",
    "mhr-0001": "miss",
    "mhr-0002": "miss",
    "mhr-0003": "miss",
    "mhr-0004": "miss",
    "mhr-0006": "miss",
    "mhr-0008": "first",
    "mhr-0009": "first",
}
_PLAN_B = {
    "mhr-0000": "first",
    "mhr-0001": "first",
    "mhr-0002": "first",
    "mhr-0003": "first",
    "mhr-0004": "first",
    "mhr-0006": "first",
    "mhr-0008": "first",
    "mhr-0009": "miss",
}


@pytest.fixture
def two_runs(tmp_path: Path) -> tuple[Path, Path]:
    return (
        _write_run(tmp_path, "a", _toy_rows(_PLAN_A)),
        _write_run(tmp_path, "b", _toy_rows(_PLAN_B)),
    )


_FIVE = [
    "all_evidence_in_first_8_passages",
    "all_evidence_in_first_20_passages",
    "all_evidence_in_first_5_docs",
    "all_evidence_in_first_10_docs",
    "any_evidence_in_first_5_passages",
]


def test_the_reading_without_set_aside_articles_keeps_only_questions_clear_of_all_of_them(
    two_runs: tuple[Path, Path],
) -> None:
    # Set aside articles 3, 4, 5 and 8. A question is dropped when its gold touches any of them,
    # whether it overlaps in part (mhr-0000 {0,3}, 0001 {1,2,3}, 0002 {1,4}, 0004, 0006 {0,4}) or
    # lies wholly inside (mhr-0003 {5,8}). Kept: mhr-0008 {10,11} and mhr-0009 {0,11}.
    # Of the two kept questions A finds both and B finds mhr-0008 only.
    result = compare.compare_runs(*two_runs, make_dataset(), set_aside=[3, 4, 5, 8])
    aside = result["without_set_aside_articles"]
    assert (aside["articles"], aside["questions_kept"], aside["questions_set_aside"]) == (4, 2, 6)
    assert list(aside["answerable"]) == _FIVE
    for name, measure in aside["answerable"].items():
        assert measure["n"] == 2, name
        assert (measure["a_rate"], measure["b_rate"]) == (1.0, 0.5), name
        assert (measure["a_found_b_missed"], measure["b_found_a_missed"]) == (1, 0), name
        assert (measure["both_found"], measure["neither_found"]) == (1, 0), name
        assert measure["sign_test_p"] == 1.0, name  # one discordant pair: p = 2 * 1/2, capped


def test_the_full_comparison_is_untouched_by_setting_articles_aside(
    two_runs: tuple[Path, Path],
) -> None:
    plain = compare.compare_runs(*two_runs, make_dataset())
    aside = compare.compare_runs(*two_runs, make_dataset(), set_aside=[3, 4, 5, 8])
    assert "without_set_aside_articles" not in plain
    assert {k: v for k, v in aside.items() if k != "without_set_aside_articles"} == plain
    # all eight answerable questions: A finds two (mhr-0008, 0009), B finds seven (all but 0009)
    m = plain["answerable"]["all_evidence_in_first_8_passages"]
    assert (m["n"], m["a_rate"], m["b_rate"]) == (8, 2 / 8, 7 / 8)


def test_the_set_aside_articles_are_counted_as_a_set_whatever_the_collection(
    two_runs: tuple[Path, Path],
) -> None:
    # 3 appears twice in the list: four distinct articles
    from_list = compare.compare_runs(*two_runs, make_dataset(), set_aside=[3, 4, 5, 8, 3])
    from_frozen = compare.compare_runs(*two_runs, make_dataset(), set_aside=frozenset({8, 5, 4, 3}))
    assert from_list["without_set_aside_articles"]["articles"] == 4
    assert from_list == from_frozen


def test_a_single_article_set_aside_removes_only_the_questions_that_name_it(
    two_runs: tuple[Path, Path],
) -> None:
    aside = compare.compare_runs(*two_runs, make_dataset(), set_aside={11})[
        "without_set_aside_articles"
    ]
    # article 11 is in the gold of mhr-0004, 0008 and 0009; the other five answerable remain
    assert (aside["articles"], aside["questions_kept"], aside["questions_set_aside"]) == (1, 5, 3)


def test_an_empty_set_aside_adds_no_section(two_runs: tuple[Path, Path]) -> None:
    for nothing in ((), [], frozenset()):
        result = compare.compare_runs(*two_runs, make_dataset(), set_aside=nothing)
        assert "without_set_aside_articles" not in result


# --- compare_runs: how few shared questions are enough -----------------------------------------


def test_exactly_one_shared_question_is_compared_and_none_is_refused(tmp_path: Path) -> None:
    found = [_row("mhr-0003", _ps(5, 8, top=5.0))]  # gold {5, 8}
    missed = [_row("mhr-0003", _ps(9, top=1.0))]
    a = _write_run(tmp_path, "a", found)
    b = _write_run(tmp_path, "b", missed)
    result = compare.compare_runs(a, b, make_dataset())
    assert result["questions"] == {"a": 1, "b": 1, "compared": 1}
    measure = result["answerable"]["all_evidence_in_first_8_passages"]
    assert (measure["n"], measure["a_rate"], measure["b_rate"]) == (1, 1.0, 0.0)
    assert (measure["a_found_b_missed"], measure["b_found_a_missed"]) == (1, 0)

    elsewhere = _write_run(tmp_path, "c", [_row("mhr-0004", _ps(9, top=1.0))])
    with pytest.raises(compare.CompareError, match="share no question"):
        compare.compare_runs(a, elsewhere, make_dataset())


def test_one_shared_null_question_compares_without_error(tmp_path: Path) -> None:
    a = _write_run(tmp_path, "a", [_row("mhr-0005", _ps(9, top=1.0))])
    b = _write_run(tmp_path, "b", [_row("mhr-0005", _ps(9, top=2.0))])
    result = compare.compare_runs(a, b, make_dataset())
    assert result["questions"]["compared"] == 1
    assert result["answerable"]["all_evidence_in_first_8_passages"]["n"] == 0
    assert result["null_top1_auroc"] == {"a": None, "b": None}


# --- runner: reading and resuming the rows file -------------------------------------------------


def test_a_damaged_last_line_is_tolerated_only_when_nothing_follows_it(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    good = json.dumps(_row("a", []))
    path.write_text(good + '\n{"id": "b", "p": [[1,')  # killed mid-write: no newline after it
    assert [r["id"] for r in runner.read_rows(path)] == ["a"]
    path.write_text(good + '\n{"id": "b", "p": [[1,\n')  # damaged and complete with its newline
    with pytest.raises(RuntimeError, match=r"rows\.jsonl:2: a row is damaged"):
        runner.read_rows(path)
    path.write_text(good + '\n{"id": "b", "p": [[1,\n\n')  # followed by a blank line as well
    with pytest.raises(RuntimeError, match=r"rows\.jsonl:2: a row is damaged"):
        runner.read_rows(path)


def test_a_row_is_one_line_however_exotic_the_characters_inside_it(tmp_path: Path) -> None:
    # U+2028 and U+0085 are line breaks to str.splitlines() but not to JSON or to this file format
    first = '{"id": "a", "note": "x\u2028y", "p": []}'
    second = '{"id": "b", "note": "u\u0085v", "p": []}'
    path = tmp_path / "rows.jsonl"
    path.write_text(first + "\n" + second + "\n", encoding="utf-8")
    assert [(r["id"], r["note"]) for r in runner.read_rows(path)] == [
        ("a", "x\u2028y"),
        ("b", "u\u0085v"),
    ]


def test_rows_written_with_windows_line_endings_are_still_read(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"id": "a", "p": []}\r\n{"id": "b", "p": []}\r\n')
    assert [r["id"] for r in runner.read_rows(path)] == ["a", "b"]


class _Pools:
    """A multiprocessing context whose pool answers each batch at once with a tiny row per item."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def Pool(self, workers: int, initializer: Any, initargs: tuple[Any, ...]) -> _Pools:
        return self

    def __enter__(self) -> _Pools:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def imap_unordered(self, fn: Any, batches: list[list[dict[str, Any]]]) -> _Pools:
        self.batches = [[p["id"] for p in batch] for batch in batches]
        self._answers = iter([[{"id": p["id"], "p": [[1, 2.5]]} for p in b] for b in batches])
        return self

    def next(self, timeout: float) -> list[dict[str, Any]]:
        return next(self._answers)


def _item(n: int) -> RunItem:
    return RunItem(f"r{n:02d}", f"r{n:02d}", "base", "a question", FilterSpec(), 12)


@pytest.mark.parametrize(
    ("existing", "skipped"),
    [
        # killed in the middle of the second row's write
        ('{"id":"r00","p":[[1,2.5]]}\n{"id":"r01","p":[[1,', 1),
        # the second row is whole but its newline never reached the disk
        ('{"id":"r00","p":[[1,2.5]]}\n{"id":"r01","p":[[1,2.5]]}', 2),
    ],
)
def test_a_resume_after_a_kill_loses_no_row_and_glues_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str, skipped: int
) -> None:
    monkeypatch.setattr(multiprocessing, "get_context", lambda name: _Pools())
    out = tmp_path / "rows.jsonl"
    out.write_text(existing)
    result = runner.run_items(
        tmp_path,
        [_item(n) for n in range(3)],
        {},
        out,
        workers=1,
        threads=None,
        limit=20,
        log=lambda message: None,
    )
    assert (result["skipped"], result["ran"]) == (skipped, 3 - skipped)
    # every line of the file is a whole row, and each item appears once, in the order it was done
    lines = out.read_text().split("\n")
    assert lines[-1] == ""
    assert [json.loads(line)["id"] for line in lines[:-1]] == ["r00", "r01", "r02"]


# --- calibration: one disagreement is enough to stop; the parity sample -------------------------

_MANIFEST = """[kb]
name = "probe"

[chunking]
strategy = "structural"

[retrieval]
rerank = "local"
"""


def _calibrate_toy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, parity: dict[str, Any]
) -> dict[str, Any]:
    """`calibrate` on the toy dataset with Pinakes' tool and the repeated searches replaced.

    The only fit-half null question is mhr-0005, with stored top-1 score -4: its median and its
    0.9 percentile are both -4, which is what the stand-in for `pinakes.calibrate` prints."""
    dataset = make_dataset()
    derived = items.derive_items(dataset)
    rows = []
    for item in derived:
        if item.variant == items.BASE:
            null = item.id in ("mhr-0005", "mhr-0007")
            rows.append(_row(item.id, _ps(0, top=-4.0 if null else 2.0)))
    kb_root = tmp_path / "kb"
    kb_root.mkdir()
    (kb_root / "pinakes.toml").write_text(_MANIFEST, encoding="utf-8")
    monkeypatch.setattr(
        calibration,
        "run_pinakes_calibrate",
        lambda root, questions: calibration.Fit("m@r", -4.0, -4.0, "the tool's output"),
    )
    monkeypatch.setattr(calibration, "parity_check", lambda *a, **k: parity)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    return calibration.calibrate(kb_root, run_dir, derived, dataset, rows, {}, limit=20)


def test_a_single_label_disagreement_with_pinakes_stops_the_calibration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parity = {"label_mismatches": 1, "checked": 10, "mismatch_ids": ["mhr-0001"]}
    with pytest.raises(calibration.CalibrationError, match=r"differed from ours on 1 of 10"):
        _calibrate_toy(monkeypatch, tmp_path, parity)


def test_no_disagreement_lets_the_calibration_finish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parity = {"label_mismatches": 0, "checked": 10, "mismatch_ids": []}
    result = _calibrate_toy(monkeypatch, tmp_path, parity)
    assert result["parity_with_pinakes_labels"] == parity
    assert result["cross_check_against_stored_scores"]["agrees"] is True


class _Handle:
    def close(self) -> None:
        pass


@pytest.mark.parametrize("count", [1200, 1800])
def test_the_parity_sample_steps_by_the_item_count_over_600(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    # 1200 // 600 is 2 and 1800 // 600 is 3, whereas dividing by 601 would give 1 and 2
    ids = [f"mhr-{n:04d}" for n in range(count)]
    visited: list[str] = []
    monkeypatch.setattr(pinakes_api, "open_kb", lambda root: _Handle())

    def run_one(handle: Any, index: Any, payload: dict[str, Any], limit: int) -> dict[str, Any]:
        visited.append(payload["id"])
        return {"p": [[0, -5.0]], "conf": "low"}

    monkeypatch.setattr(runner, "run_one", run_one)
    base = [RunItem(i, i, items.BASE, "q", FilterSpec(), 1) for i in ids]
    recorded = {i: {"p": [[0, -5.0]]} for i in ids}
    fit = calibration.Fit("m@r", -3.0, 1.0, "")
    result = calibration.parity_check(Path("."), base, {}, recorded, fit, limit=20)
    step = count // 600
    assert result["checked"] == 600
    assert visited == [ids[n] for n in range(0, count, step)][:600]


# --- config: the max_tokens bounds --------------------------------------------------------------


@pytest.mark.parametrize("value", [1, 400, 459, 510])
def test_a_max_tokens_from_one_to_510_is_accepted(value: int) -> None:
    assert config_module.parse("c", {"chunking_max_tokens": value}).chunking_max_tokens == value


@pytest.mark.parametrize("value", [0, -1, 511, 600, True, False, "459", 459.0, [459]])
def test_a_max_tokens_outside_one_to_510_or_not_an_integer_is_refused(value: object) -> None:
    with pytest.raises(config_module.ConfigError, match="chunking_max_tokens"):
        config_module.parse("c", {"chunking_max_tokens": value})


def test_no_max_tokens_means_pinakes_own_default() -> None:
    assert config_module.parse("c", {}).chunking_max_tokens is None


# --- report: the comparison, with every number different ------------------------------------------


def _paired(
    n: int, a_rate: float, b_rate: float, b_missed: int, a_missed: int, p: float
) -> dict[str, Any]:
    return {
        "n": n,
        "a_rate": a_rate,
        "b_rate": b_rate,
        "b_found_a_missed": b_missed,
        "a_found_b_missed": a_missed,
        "both_found": 0,
        "neither_found": 0,
        "sign_test_p": p,
    }


def _comparison() -> dict[str, Any]:
    return {
        "a": {
            "run_id": "run-alpha",
            "config": "cfg-a",
            "chunking_metadata": "off",
            "chunking_max_tokens": None,
            "pinakes": "pinakes-a",
        },
        "b": {
            "run_id": "run-beta",
            "config": "cfg-b",
            "chunking_metadata": "prefix",
            "chunking_max_tokens": 459,
            "pinakes": "pinakes-b",
        },
        "questions": {"a": 101, "b": 203, "compared": 87},
        "identical_searches": {"of": 87, "same_passages_and_scores": 41},
        "answerable": {
            "all_evidence_in_first_8_passages": _paired(60, 0.25, 0.5, 7, 3, 0.2),
        },
        "by_question_type": {"comparison_query": _paired(33, 0.75, 0.125, 9, 4, 0.1)},
        "null_top1_auroc": {"a": 0.61, "b": 0.82},
        "without_set_aside_articles": {
            "articles": 13,
            "questions_kept": 29,
            "questions_set_aside": 17,
            "answerable": {"any_evidence_in_first_5_passages": _paired(29, 0.2, 0.6, 5, 2, 0.5)},
        },
    }


def test_the_comparison_states_the_question_counts_in_the_right_places() -> None:
    text = report.render_compare(_comparison())
    assert "Questions: A has 101, B has 203; compared on the 87 they share." in text
    # the identical searches are "N of M": 41 of the 87 compared
    assert "with the same scores: 41 of 87." in text


def test_the_comparison_table_rows_carry_each_number_under_its_own_heading() -> None:
    text = report.render_compare(_comparison())
    assert "| configuration | cfg-a | cfg-b |" in text
    assert "| chunking metadata | off | prefix |" in text
    assert "| chunking max_tokens | default | 459 |" in text
    assert "| all evidence in first 8 passages | 25.0% | 50.0% | 7 | 3 | 0.2000 |" in text
    assert "| comparison_query | 33 | 75.0% | 12.5% | 9 | 4 | 0.1000 |" in text
    assert "A 0.610, B 0.820" in text


def test_the_set_aside_section_reports_kept_and_set_aside_the_right_way_round() -> None:
    text = report.render_compare(_comparison())
    section = text.split("## Read without the articles the two configurations chunk differently")[1]
    assert "splits 13 articles differently" in section
    assert (
        "The 17 answerable questions whose gold includes one of them are set aside, leaving 29;"
        in section
    )


def test_the_set_aside_table_names_its_rows_in_words_and_orders_a_before_b() -> None:
    text = report.render_compare(_comparison())
    section = text.split("## Read without the articles the two configurations chunk differently")[1]
    assert "| any evidence in first 5 passages | 20.0% | 60.0% | 5 | 2 | 0.5000 |" in section
    assert "any_evidence_in_first_5_passages" not in text
    assert "all_evidence_in_first_8_passages" not in text


def _header() -> dict[str, Any]:
    return {
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


def test_the_scored_by_row_sits_between_the_benchmark_repo_and_the_environment() -> None:
    scored = {
        "benchmark": {"version": "0.1.0", "commit": "d" * 40, "dirty": False},
        "scored_utc": "2026-10-30T20:40:00+00:00",
    }
    text = report._header_section(_header(), scored)
    # skip the heading, the blank line, the table's head and its separator
    names = [line[2:].split(" |")[0] for line in text.split("\n")[4:]]
    assert names[-3:] == ["Benchmark repo", "Scored by", "Environment"]
    assert names.count("Scored by") == 1


# --- kb.build_kb: a configuration can lower max_tokens ------------------------------------------

_BUILD_MANIFEST = """[kb]
name = "probe"

[embedding]
model = "e-model"

[chunking]
strategy = "structural"

[retrieval]
rerank = "local"

[rerank]
model = "r-model"
"""
_PINS = Models(embedding=ModelPin("e-model", "5" * 40), rerank=ModelPin("r-model", "2" * 40))


def _fake_pinakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def run_pnk(args: list[str], *, log: Path | None = None) -> str:
        if args[0] == "init":
            Path(args[1]).mkdir(parents=True, exist_ok=True)
            (Path(args[1]) / "pinakes.toml").write_text(_BUILD_MANIFEST, encoding="utf-8")
        return ""

    monkeypatch.setenv(paths.ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(pinakes_api, "require_interfaces", lambda: None)
    monkeypatch.setattr(pinakes_api, "run_pnk", run_pnk)
    monkeypatch.setattr(pinakes_api, "mint_sidecar", lambda *a, **k: None)
    monkeypatch.setattr(kb, "load_models", lambda: _PINS)
    monkeypatch.setattr(kb, "verify_kb", lambda dataset, root: {"documents": 12, "chunks": 7})


@pytest.mark.parametrize(
    ("max_tokens", "metadata", "expected"),
    [
        (459, "prefix", {"strategy": "structural", "max_tokens": 459, "metadata": "prefix"}),
        (459, "off", {"strategy": "structural", "max_tokens": 459}),
        (None, "prefix", {"strategy": "structural", "metadata": "prefix"}),
        (None, "off", {"strategy": "structural"}),
    ],
)
def test_build_writes_max_tokens_into_the_manifest_only_when_the_configuration_asks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    max_tokens: int | None,
    metadata: str,
    expected: dict[str, Any],
) -> None:
    _fake_pinakes(monkeypatch, tmp_path)
    config = Config("probe", "", "st", metadata, chunking_max_tokens=max_tokens)
    kb.build_kb(make_dataset(), config, log=lambda message: None)
    manifest = tomllib.loads((paths.kb_dir("probe") / "pinakes.toml").read_text(encoding="utf-8"))
    assert manifest["chunking"] == expected
    assert manifest["embedding"]["revision"] == "5" * 40  # the pins are still written


# --- a repository laid out like the real one, for the steps that read run directories -----------


def _write_toy_dataset_like_the_real_one(root: Path) -> None:
    """The toy dataset as raw files plus a manifest, so the production loader reads it."""
    import hashlib

    from pnkbench import multihop_rag as mh

    dataset = make_dataset()
    corpus = [
        {
            "title": a.title,
            "author": a.author,
            "source": a.source,
            "category": a.category,
            "url": a.url,
            "published_at": a.published_at.isoformat(),
            "body": a.body,
        }
        for a in dataset.articles
    ]
    by_title = dataset.article_by_title()
    queries = [
        {
            "query": q.text,
            "answer": q.answer,
            "question_type": q.qtype,
            "evidence_list": [
                {
                    "title": e.title,
                    "url": e.url,
                    "fact": e.fact,
                    "author": None,
                    "source": by_title[e.title].source,
                    "category": by_title[e.title].category,
                    "published_at": by_title[e.title].published_at.isoformat(),
                }
                for e in q.evidence
            ],
        }
        for q in dataset.queries
    ]
    folder = root / "datasets" / mh.DATASET
    (folder / "raw").mkdir(parents=True)
    files = []
    for name, role, payload in (
        ("corpus.json", "corpus", corpus),
        ("MultiHopRAG.json", "queries", queries),
    ):
        data = json.dumps(payload).encode()
        (folder / "raw" / name).write_bytes(data)
        files.append(
            {
                "path": name,
                "role": role,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (folder / "manifest.json").write_text(
        json.dumps(
            {"source_repo": "owner/toy", "revision": "a" * 40, "licence": "n/a", "files": files}
        )
    )


def _toy_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    shutil.copytree(REAL_CONFIGS, root / "configs")
    monkeypatch.setenv(paths.ROOT_ENV, str(root))
    _write_toy_dataset_like_the_real_one(root)
    (root / "runs").mkdir()
    return root


# --- pipeline.compare_step: which indexes are read, in which order, and when ---------------------


def _finished_run(
    name: str, config_name: str, max_tokens: int | str | None, rows: list[dict[str, Any]]
) -> Path:
    run_dir = paths.runs_dir() / name
    run_dir.mkdir()
    config: dict[str, Any] = {"name": config_name, "chunking_metadata": "off"}
    if max_tokens != "absent":
        config["chunking_max_tokens"] = max_tokens
    header = {
        "run_id": name,
        "status": "complete",
        "dataset": {"sha256": {"corpus.json": "s"}},
        "execution": {"passages_per_search": 20},
        "config": config,
        "pinakes": {"version": "0.32.8", "source": "pypi", "code_sha256": "c" * 64},
    }
    (run_dir / "run.json").write_text(json.dumps(header))
    (run_dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run_dir


@pytest.mark.parametrize(
    ("tokens_a", "tokens_b", "reads_the_indexes"),
    [
        ("absent", 459, True),
        (None, 459, True),
        (459, None, True),
        (459, 400, True),
        (None, None, False),
        ("absent", None, False),
        (None, "absent", False),
        (459, 459, False),
    ],
)
def test_the_indexes_are_compared_only_when_the_two_runs_chunk_with_different_max_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tokens_a: int | str | None,
    tokens_b: int | str | None,
    reads_the_indexes: bool,
) -> None:
    _toy_repo(tmp_path, monkeypatch)
    calls: list[tuple[Path, Path, dict[str, int]]] = []

    def differently(index_a: Path, index_b: Path, path_index: dict[str, int]) -> frozenset[int]:
        calls.append((index_a, index_b, path_index))
        return frozenset({3, 4, 5, 8})

    monkeypatch.setattr(compare, "articles_chunked_differently", differently)
    monkeypatch.setattr(
        pinakes_api, "index_path", lambda kb_root: Path("/indexes") / kb_root.name / "index.db"
    )
    rows = _toy_rows(_PLAN_A)
    run_a = _finished_run("run-a", "baseline", tokens_a, rows)
    run_b = _finished_run("run-b", "metadata-prefix", tokens_b, _toy_rows(_PLAN_B))

    target = pipeline.compare_step(run_a, run_b)

    result = json.loads(target.with_suffix(".json").read_text())
    if reads_the_indexes:
        dataset = make_dataset()
        # A's index first, then B's, each from its own run's configuration
        assert calls == [
            (
                Path("/indexes/baseline/index.db"),
                Path("/indexes/metadata-prefix/index.db"),
                {layout.doc_relpath(a): a.index for a in dataset.articles},
            )
        ]
        assert result["without_set_aside_articles"]["articles"] == 4
        assert "Read without the articles the two configurations chunk differently" in (
            target.read_text()
        )
    else:
        assert calls == []
        assert "without_set_aside_articles" not in result
        assert "Read without the articles" not in target.read_text()
    assert target == paths.runs_dir() / "compare--run-a--run-b.md"


# --- pipeline.run: what a resume refuses and what it carries over --------------------------------

_PROVENANCE = {"source": "pypi", "version": "0.32.8", "commit": None, "code_sha256": "b" * 64}


class _Stopped(Exception):
    pass


def _resume_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A toy repository with an empty KB directory for each configuration and Pinakes replaced."""
    root = _toy_repo(tmp_path, monkeypatch)
    for name in ("baseline", "metadata-prefix"):
        paths.kb_dir(name).mkdir(parents=True)
        (paths.kb_dir(name) / "pinakes.toml").write_text("")
    monkeypatch.setattr(pinakes_api, "require_interfaces", lambda: None)
    monkeypatch.setattr(pinakes_api, "provenance", lambda: dict(_PROVENANCE))
    return root


def _earlier_run(name: str, **header: Any) -> Path:
    run_dir = paths.runs_dir() / name
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps(header))
    return run_dir


def test_a_resume_refuses_a_run_started_by_another_build_of_pinakes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _resume_world(tmp_path, monkeypatch)
    run_dir = _earlier_run(
        "earlier", pinakes={"code_sha256": "a" * 64}, config={"name": "baseline"}
    )
    with pytest.raises(RuntimeError, match=r"cannot resume: the installed Pinakes differs"):
        pipeline.run("baseline", workers=1, resume=run_dir, log=lambda m: None)


def test_a_resume_refuses_a_run_made_with_another_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _resume_world(tmp_path, monkeypatch)
    run_dir = _earlier_run(
        "earlier", pinakes={"code_sha256": _PROVENANCE["code_sha256"]}, config={"name": "baseline"}
    )
    with pytest.raises(RuntimeError, match=r"cannot resume: this run used another configuration"):
        pipeline.run("metadata-prefix", workers=1, resume=run_dir, log=lambda m: None)


def test_a_resume_that_passes_both_checks_goes_on_to_open_the_knowledge_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _resume_world(tmp_path, monkeypatch)
    run_dir = _earlier_run(
        "earlier", pinakes={"code_sha256": _PROVENANCE["code_sha256"]}, config={"name": "baseline"}
    )

    def stop(root: Path) -> None:
        raise _Stopped

    monkeypatch.setattr(pinakes_api, "open_kb", stop)
    with pytest.raises(_Stopped):
        pipeline.run("baseline", workers=1, resume=run_dir, log=lambda m: None)


def _fake_stack(monkeypatch: pytest.MonkeyPatch, settings: dict[str, Any]) -> None:
    """Everything `pipeline.run` takes from Pinakes, the models, the pool and the clock."""
    handle = SimpleNamespace(
        manifest=SimpleNamespace(retrieval=SimpleNamespace(fusion_top_k=20)),
        backend=None,
        reranker=None,
        close=lambda: None,
    )
    torch = SimpleNamespace(
        __version__="2.0", backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(pinakes_api, "open_kb", lambda root: handle)
    monkeypatch.setattr(pinakes_api, "settings_record", lambda h: settings)
    monkeypatch.setattr(
        runner, "run_items", lambda *a, **k: {"ran": 0, "skipped": 0, "seconds": 0.0}
    )
    monkeypatch.setattr(runner, "finalize", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "reproducibility", lambda *a, **k: {"checked": 0})
    monkeypatch.setattr(pipeline, "git_state", lambda: {"commit": None, "dirty": None})
    # the clock: every reading is a minute later, and every monotonic reading 1.5 s after the last
    minutes = iter(range(10_000))
    monkeypatch.setattr(
        pipeline,
        "utc_now",
        lambda: datetime(2026, 11, 1, tzinfo=UTC) + timedelta(minutes=next(minutes)),
    )
    seconds = iter(x * 1.5 for x in range(10_000))
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(monotonic=lambda: next(seconds)))


def _header_of(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "run.json").read_text())


def test_every_resume_adds_itself_to_the_history_and_the_wall_time_accumulates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _resume_world(tmp_path, monkeypatch)
    _fake_stack(monkeypatch, {"retrieval": {"fusion_top_k": 20}})
    quiet = {"workers": 1, "log": lambda m: None}

    run_dir = pipeline.run("baseline", **quiet)
    first = _header_of(run_dir)
    assert first["execution"]["wall_seconds"] == 1.5
    assert "resumed_utc" not in first["execution"]
    assert "earlier_wall_seconds" not in first["execution"]

    assert pipeline.run("baseline", resume=run_dir, **quiet) == run_dir
    second = _header_of(run_dir)
    assert second["started_utc"] == first["started_utc"]  # the original start is kept
    assert len(second["execution"]["resumed_utc"]) == 1
    assert second["execution"]["earlier_wall_seconds"] == 1.5  # the first session's 1.5 s
    assert second["execution"]["wall_seconds"] == 1.5  # this session's own

    assert pipeline.run("baseline", resume=run_dir, **quiet) == run_dir
    third = _header_of(run_dir)
    assert third["started_utc"] == first["started_utc"]
    assert len(third["execution"]["resumed_utc"]) == 2
    assert third["execution"]["resumed_utc"][0] == second["execution"]["resumed_utc"][0]
    assert third["execution"]["resumed_utc"][1] != third["execution"]["resumed_utc"][0]
    assert third["execution"]["earlier_wall_seconds"] == 3.0  # 1.5 + 1.5
    assert third["execution"]["wall_seconds"] == 1.5


@pytest.mark.parametrize(
    ("what", "phrase"),
    [
        ("settings", r"cannot resume: settings \(models, chunking, retrieval\) differ"),
        ("dataset", r"cannot resume: dataset files differ"),
        ("sample", r"cannot resume: sample differ"),
    ],
)
def test_a_resume_names_what_changed_since_the_run_began(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, what: str, phrase: str
) -> None:
    _resume_world(tmp_path, monkeypatch)
    settings = {"retrieval": {"fusion_top_k": 20}}
    _fake_stack(monkeypatch, settings)
    run_dir = pipeline.run("baseline", workers=1, log=lambda m: None)
    header = _header_of(run_dir)
    sample: int | None = None
    if what == "settings":
        monkeypatch.setattr(pinakes_api, "settings_record", lambda h: {"retrieval": {"x": 1}})
    elif what == "dataset":
        header["dataset"]["sha256"]["corpus.json"] = "0" * 64
        (run_dir / "run.json").write_text(json.dumps(header))
    else:
        sample = 3
    with pytest.raises(RuntimeError, match=phrase):
        pipeline.run("baseline", workers=1, sample=sample, resume=run_dir, log=lambda m: None)
    # the refusal leaves the earlier run's record as it was
    assert _header_of(run_dir) == header


def test_a_resume_that_changes_nothing_is_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _resume_world(tmp_path, monkeypatch)
    _fake_stack(monkeypatch, {"retrieval": {"fusion_top_k": 20}})
    run_dir = pipeline.run("baseline", workers=1, log=lambda m: None)
    assert pipeline.run("baseline", workers=1, resume=run_dir, log=lambda m: None) == run_dir


def test_the_toy_dataset_is_what_the_resume_tests_think_it_is() -> None:
    # guards the hand-derived numbers above: ten queries, eight answerable, twelve articles
    dataset: Dataset = make_dataset()
    assert (len(dataset.queries), len(dataset.articles)) == (10, 12)
    assert sum(1 for q in dataset.queries if q.answerable) == 8
