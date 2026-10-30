from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from toy import make_dataset

from pnkbench import calibration, compare, items, pinakes_api, runner, scoring


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


# --- runner -------------------------------------------------------------------------------


def test_read_rows_skips_a_line_cut_off_by_a_kill(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(_row("a", [])) + "\n" + json.dumps(_row("b", []))[:15])
    assert [r["id"] for r in runner.read_rows(path)] == ["a"]
    assert runner.read_rows(tmp_path / "missing.jsonl") == []


def test_finalize_orders_rows_and_checks_the_item_list(tmp_path: Path) -> None:
    dataset = make_dataset()
    derived = items.derive_items(dataset)
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "".join(json.dumps(_row(i.id, [], i.variant)) + "\n" for i in reversed(derived))
    )
    rows = runner.finalize(path, derived)
    assert [r["id"] for r in rows] == [i.id for i in derived]
    assert [r["id"] for r in runner.read_rows(path)] == [i.id for i in derived]


def test_finalize_refuses_missing_duplicate_and_unknown_rows(tmp_path: Path) -> None:
    derived = items.derive_items(make_dataset())
    path = tmp_path / "rows.jsonl"
    full = [json.dumps(_row(i.id, [], i.variant)) for i in derived]

    path.write_text("\n".join(full[:-1]) + "\n")
    with pytest.raises(RuntimeError, match="1 missing"):
        runner.finalize(path, derived)

    path.write_text("\n".join([*full, full[0]]) + "\n")
    with pytest.raises(RuntimeError, match="appears twice"):
        runner.finalize(path, derived)

    path.write_text("\n".join([*full, json.dumps(_row("ghost", []))]) + "\n")
    with pytest.raises(RuntimeError, match="1 unknown"):
        runner.finalize(path, derived)


def test_run_one_maps_paths_to_article_indices_and_rejects_strangers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = pinakes_api.Outcome(
        passages=(("docs/a.txt", 1.5), ("docs/b.txt", None)), confidence="unknown", considered=2
    )
    monkeypatch.setattr(pinakes_api, "search_kb", lambda *a, **k: outcome)
    payload = {
        "id": "x",
        "base_id": "x",
        "variant": "base",
        "query": "q",
        "tags": [],
        "path_prefix": None,
        "after": None,
        "before": None,
    }
    row = runner.run_one(SimpleNamespace(), {"docs/a.txt": 4, "docs/b.txt": 7}, payload, 20)
    assert row["p"] == [[4, 1.5], [7, None]] and row["considered"] == 2
    with pytest.raises(RuntimeError, match="which is no article"):
        runner.run_one(SimpleNamespace(), {"docs/a.txt": 4}, payload, 20)


def test_payload_carries_the_whole_filter() -> None:
    item = items.RunItem("i", "i", "f-date", "q", items.FilterSpec(("t",), "docs/x/", 1.0, 2.0), 3)
    payload = runner.to_payload(item)
    assert (payload["tags"], payload["path_prefix"], payload["after"], payload["before"]) == (
        ["t"],
        "docs/x/",
        1.0,
        2.0,
    )


# --- calibration --------------------------------------------------------------------------


def test_thresholds_follow_pinakes_formula() -> None:
    assert calibration.expected_thresholds([-9, -8, -7, -6, -5]) == (-7, -5)
    low, high = calibration.expected_thresholds([1.0])
    assert (low, high) == (1.0, 1.0)


def test_percentile_matches_pinakes_own_function() -> None:
    pinakes_calibrate = pytest.importorskip("pinakes.calibrate")
    import random

    rng = random.Random(3)
    for n in (1, 2, 3, 10, 57, 150):
        values = [rng.uniform(-10, 5) for _ in range(n)]
        for fraction in (0.0, 0.5, 0.9, 1.0):
            assert calibration.pinakes_percentile(
                values, fraction
            ) == pinakes_calibrate._percentile(values, fraction)


def test_parse_block_reads_what_pinakes_prints_and_refuses_anything_else() -> None:
    printed = 'fitted on 3 answerable and 3 unanswerable\n\n[retrieval.confidence]\nfitted_for = "m@abc"\nlow_below  = -4.1234\nhigh_above = 0.5000\n'
    fit = calibration.parse_block(printed)
    assert (fit.fitted_for, fit.low_below, fit.high_above) == ("m@abc", -4.1234, 0.5)
    with pytest.raises(calibration.CalibrationError, match="printed no"):
        calibration.parse_block("nothing useful")


def test_fit_items_are_the_fit_half_nulls_and_a_capped_answerable_sample() -> None:
    dataset = make_dataset()
    derived = items.derive_items(dataset)
    chosen = calibration.fit_items(derived, dataset)
    info = scoring.query_info(dataset)
    assert all(i.variant == items.BASE and info[i.base_id].split == items.FIT for i in chosen)
    fit_null = [
        q.id for q in dataset.queries if not q.answerable and items.split_of(q.id) == items.FIT
    ]
    assert sorted(i.id for i in chosen if not info[i.base_id].answerable) == sorted(fit_null)
    assert (
        len([i for i in chosen if info[i.base_id].answerable]) <= items.CALIBRATION_ANSWERABLE_CAP
    )
    assert not {i.id for i in chosen} & {
        q.id for q in dataset.queries if items.split_of(q.id) == items.EVAL
    }


# --- compare ------------------------------------------------------------------------------


def _write_run(
    directory: Path,
    name: str,
    rows: list[dict[str, Any]],
    *,
    sha: str = "s",
    status: str = "complete",
    passages: int = 20,
) -> Path:
    run_dir = directory / name
    run_dir.mkdir()
    header = {
        "run_id": name,
        "status": status,
        "dataset": {"sha256": {"corpus.json": sha}},
        "execution": {"passages_per_search": passages},
        "config": {"name": name, "chunking_metadata": "off"},
        "pinakes": {"version": "0.32.8", "source": "pypi", "code_sha256": "c" * 64},
    }
    (run_dir / "run.json").write_text(json.dumps(header))
    (run_dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run_dir


def _base_rows(found: dict[str, bool]) -> list[dict[str, Any]]:
    dataset = make_dataset()
    info = scoring.query_info(dataset)
    rows = []
    for q in dataset.queries:
        if q.answerable:
            passages = (
                [[g, 1.0] for g in sorted(info[q.id].gold)]
                if found.get(q.id, True)
                else [[11, 0.1]]
            )
        else:
            passages = [[0, -5.0]]
        rows.append(_row(q.id, passages))
    return rows


def test_compare_counts_questions_each_side_alone_found(tmp_path: Path) -> None:
    a = _write_run(tmp_path, "a", _base_rows({"mhr-0000": False}))
    b = _write_run(tmp_path, "b", _base_rows({"mhr-0001": False, "mhr-0002": False}))
    result = compare.compare_runs(a, b, make_dataset())
    primary = result["answerable"]["all_evidence_in_first_8_passages"]
    assert (primary["b_found_a_missed"], primary["a_found_b_missed"]) == (1, 2)
    assert primary["n"] == 8
    assert primary["both_found"] == 5 and primary["neither_found"] == 0
    assert result["by_question_type"]["comparison_query"]["n"] > 0


def test_compare_refuses_runs_that_do_not_measure_the_same_thing(tmp_path: Path) -> None:
    rows = _base_rows({})
    a = _write_run(tmp_path, "a", rows)
    with pytest.raises(compare.CompareError, match="different dataset"):
        compare.compare_runs(a, _write_run(tmp_path, "b", rows, sha="other"), make_dataset())
    with pytest.raises(compare.CompareError, match="not a complete run"):
        compare.compare_runs(a, _write_run(tmp_path, "c", rows, status="running"), make_dataset())
    stranger = [dict(r, id="zzz", base_id="zzz") for r in rows]
    with pytest.raises(compare.CompareError, match="share no question"):
        compare.compare_runs(a, _write_run(tmp_path, "d", stranger), make_dataset())


def test_compare_pairs_a_sample_run_with_a_full_run_on_their_common_questions(
    tmp_path: Path,
) -> None:
    rows = _base_rows({})
    full = _write_run(tmp_path, "full", rows)
    sample = _write_run(tmp_path, "sample", rows[:6])
    result = compare.compare_runs(full, sample, make_dataset())
    assert result["questions"] == {"a": 10, "b": 6, "compared": 6}
    assert result["identical_searches"] == {"of": 6, "same_passages_and_scores": 6}


def test_identical_searches_counts_only_exact_matches(tmp_path: Path) -> None:
    rows = _base_rows({})
    shifted = [
        dict(r, p=[[p[0], p[1] + 0.001] for p in r["p"]]) if r["id"] == "mhr-0000" else r
        for r in rows
    ]
    result = compare.compare_runs(
        _write_run(tmp_path, "a", rows), _write_run(tmp_path, "b", shifted), make_dataset()
    )
    assert result["identical_searches"] == {"of": 10, "same_passages_and_scores": 9}


def test_the_comparison_report_states_the_overlap_and_the_identical_searches(
    tmp_path: Path,
) -> None:
    from pnkbench import report

    rows = _base_rows({})
    result = compare.compare_runs(
        _write_run(tmp_path, "a", rows), _write_run(tmp_path, "b", rows[:6]), make_dataset()
    )
    text = report.render_compare(result)
    assert "compared on the 6 they share" in text and "same scores: 6 of 6" in text


# --- parity with Pinakes' own code --------------------------------------------------------


def test_confidence_label_equals_pinakes_confidence_of_everywhere_including_the_edges() -> None:
    search = pytest.importorskip("pinakes.search")
    fingerprint = "BAAI/bge-reranker-base@abc"
    reranker = SimpleNamespace(info=lambda: SimpleNamespace(fingerprint=lambda: fingerprint))
    manifest = SimpleNamespace(
        retrieval=SimpleNamespace(
            rerank="local",
            confidence=SimpleNamespace(fitted_for=fingerprint, low_below=-3.0, high_above=1.0),
        )
    )
    for score in (-9.0, -3.0001, -3.0, -2.9999, 0.0, 0.9999, 1.0, 1.0001, 7.0):
        label, _ = search.confidence_of(manifest, reranker, [SimpleNamespace(rerank_score=score)])
        assert scoring.confidence_label(score, -3.0, 1.0) == label, score


def test_false_rates_equal_pinakes_score_rows_on_the_same_outcomes() -> None:
    ev = pytest.importorskip("pinakes.eval")
    dataset = make_dataset()
    info = scoring.query_info(dataset)
    low, high = -3.0, 1.0
    rows, outcome_rows = [], []
    scores = [-9.0, -4.0, -3.0, -1.0, 1.0, 2.0]
    for n, q in enumerate(dataset.queries):
        score = scores[n % len(scores)]
        gold = sorted(info[q.id].gold)
        passages = [[gold[0], score]] if q.answerable and n % 3 else [[11, score]]
        if not q.answerable:
            passages = [[0, score]]
        rows.append(_row(q.id, passages))
        label = scoring.confidence_label(score, low, high)
        hit_rank = (
            1 if q.answerable and scoring.found_any_passages(rows[-1], info[q.id].gold, 5) else None
        )
        outcome_rows.append(
            ev.OutcomeRow(
                id=q.id,
                kind="multi-hop" if q.answerable else "no-answer",
                hit=hit_rank is not None,
                hit_rank=hit_rank,
                confidence=label,
            )
        )
    theirs = ev.score_rows(outcome_rows).as_dict()
    ours = scoring.abstention_report(rows, info, {"low_below": low, "high_above": high})

    def pooled(key: str) -> tuple[int, int]:
        sides = [ours["labels"][s][key] for s in ("eval", "fit")]
        return sum(s["successes"] for s in sides), sum(s["n"] for s in sides)

    successes, n = pooled("false_abstain_pinakes_definition")
    assert successes / n == pytest.approx(theirs["false_abstain"], abs=1e-4)
    successes, n = pooled("false_confidence_null_labelled_high")
    assert successes / n == pytest.approx(theirs["false_confidence"], abs=1e-4)


def test_a_pool_that_never_finishes_a_batch_is_reported_not_waited_for(
    tmp_path: Path, monkeypatch
) -> None:
    """A worker that cannot start is respawned forever by multiprocessing; the deadline turns that into an error."""
    import multiprocessing

    monkeypatch.setattr(runner, "STALL_SECONDS", 0.2)

    class _Hung:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def imap_unordered(self, fn, batches):
            class _It:
                def next(self, timeout):
                    raise multiprocessing.TimeoutError

            return _It()

    class _Context:
        def Pool(self, *a, **k):
            return _Hung()

    monkeypatch.setattr(multiprocessing, "get_context", lambda name: _Context())
    derived = items.derive_items(make_dataset())
    with pytest.raises(RuntimeError, match="no batch finished"):
        runner.run_items(
            tmp_path,
            derived,
            {},
            tmp_path / "rows.jsonl",
            workers=1,
            threads=None,
            limit=20,
            log=lambda m: None,
        )


def test_git_state_ignores_the_runs_directory_but_notices_other_changes(
    tmp_path: Path, monkeypatch
) -> None:
    import subprocess

    from pnkbench import paths, pipeline

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.test")
    git("config", "user.name", "t")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("add", "code.py")
    git("commit", "-q", "-m", "init")
    monkeypatch.setenv(paths.ROOT_ENV, str(tmp_path))
    assert pipeline.git_state()["dirty"] is False
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    (tmp_path / "runs" / "r1" / "rows.jsonl").write_text("{}\n")
    assert pipeline.git_state()["dirty"] is False  # this tool's own output
    (tmp_path / "code.py").write_text("x = 2\n")
    assert pipeline.git_state()["dirty"] is True


def test_compare_refuses_runs_that_searched_for_different_numbers_of_passages(
    tmp_path: Path,
) -> None:
    """Measures counted in passages would compare unlike things; this once passed silently."""
    rows = _base_rows({})
    eight = _write_run(tmp_path, "eight", rows, passages=8)
    twenty = _write_run(tmp_path, "twenty", rows, passages=20)
    with pytest.raises(compare.CompareError, match=r"different numbers of passages \(8 and 20\)"):
        compare.compare_runs(eight, twenty, make_dataset())
    assert compare.compare_runs(twenty, twenty, make_dataset())["a"]["passages_per_search"] == 20


def test_read_rows_tolerates_only_a_cut_off_last_line(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    good = [json.dumps(_row(name, [])) for name in ("a", "b", "c")]
    path.write_text("\n".join([good[0], "{not json", good[2]]) + "\n")
    with pytest.raises(RuntimeError, match=r"rows\.jsonl:2: a row is damaged"):
        runner.read_rows(path)
    path.write_text(
        "\n".join([good[0], good[1], good[2][:10]])
    )  # the kill landed in the last write
    assert [r["id"] for r in runner.read_rows(path)] == ["a", "b"]
    path.write_text(good[0] + "\n\n" + good[1] + "\n")  # blank lines are not damage
    assert [r["id"] for r in runner.read_rows(path)] == ["a", "b"]


def test_a_resume_after_a_cut_off_write_loses_no_row(tmp_path: Path) -> None:
    """Appending to an unterminated line glued the next row onto it and dropped both."""
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(_row("a", [])) + "\n" + json.dumps(_row("b", []))[:20])
    runner.repair_tail(path)
    with path.open("a") as sink:
        sink.write(json.dumps(_row("d", [])) + "\n")
    assert [r["id"] for r in runner.read_rows(path)] == ["a", "d"]  # b was cut off, so b runs again


def test_a_complete_row_missing_only_its_newline_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(_row("a", [])) + "\n" + json.dumps(_row("b", [])))
    runner.repair_tail(path)
    assert path.read_text().endswith("}\n")
    with path.open("a") as sink:
        sink.write(json.dumps(_row("c", [])) + "\n")
    assert [r["id"] for r in runner.read_rows(path)] == ["a", "b", "c"]


def test_repair_tail_leaves_a_clean_or_absent_file_alone(tmp_path: Path) -> None:
    runner.repair_tail(tmp_path / "missing.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    runner.repair_tail(empty)
    assert empty.read_text() == ""
    clean = tmp_path / "clean.jsonl"
    clean.write_text(json.dumps(_row("a", [])) + "\n")
    before = clean.read_bytes()
    runner.repair_tail(clean)
    assert clean.read_bytes() == before
