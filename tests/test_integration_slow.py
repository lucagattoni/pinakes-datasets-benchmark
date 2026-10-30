"""Real Pinakes, real models, a 12-article corpus. Run with `uv run pytest -m slow`.

This is the one place the filter machinery is exercised against Pinakes itself rather than against
rows written by hand: build the KB through the production code path, search every derived item,
and score the result with the production scorer.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from toy import make_dataset

pytestmark = pytest.mark.slow

pytest.importorskip("pinakes")

from pnkbench import config, items, kb, layout, paths, pinakes_api, pipeline, runner, scoring  # noqa: E402

REAL_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("repo")
    shutil.copytree(REAL_CONFIGS, root / "configs")
    mp = pytest.MonkeyPatch()
    mp.setenv(paths.ROOT_ENV, str(root))
    dataset = make_dataset()
    report = kb.build_kb(dataset, config.load("baseline"), log=lambda m: None)
    handle = pinakes_api.open_kb(paths.kb_dir("baseline"))
    yield dataset, report, handle
    handle.close()
    mp.undo()


def test_the_built_index_is_exactly_the_dataset(built) -> None:
    dataset, report, _ = built
    assert report["documents"] == len(dataset.articles) == 12
    assert report["chunks"] >= 36  # three paragraphs each


def test_a_distinctive_word_finds_its_article_first(built) -> None:
    dataset, _, handle = built
    index = pipeline.path_index(dataset)
    for article in dataset.articles:
        payload = {
            "id": "x",
            "base_id": "x",
            "variant": "base",
            "query": f"zebra{article.index}",
            "tags": [],
            "path_prefix": None,
            "after": None,
            "before": None,
        }
        row = runner.run_one(handle, index, payload, 20)
        assert row["p"][0][0] == article.index, f"zebra{article.index} -> {row['p'][0]}"


def test_pinakes_applies_every_derived_filter_correctly(built) -> None:
    dataset, _, handle = built
    index = pipeline.path_index(dataset)
    derived = items.derive_items(dataset)
    rows = [runner.run_one(handle, index, runner.to_payload(i), 20) for i in derived]
    report = scoring.filter_report(
        {r["id"]: r for r in rows},
        derived,
        dataset,
        scoring.query_info(dataset),
        relpath_of=layout.doc_relpath,
        tags_of=layout.tags_for,
        fusion_top_k=20,
    )
    assert set(report) >= {items.F_CAT_TAG, items.F_CAT_PATH, items.F_SRC_TAG, items.F_DATE}
    for variant in items.FILTER_VARIANTS:
        assert report[variant]["soundness_violations"] == 0, report[variant]
    assert report["equivalence_tag_vs_path_prefix"]["mismatches"] == 0
    assert report["equivalence_tag_vs_path_prefix"]["pairs"] >= 3


def test_a_filter_that_matches_nothing_returns_nothing(built) -> None:
    _, _, handle = built
    outcome = pinakes_api.search_kb(
        handle, "zebra3", items.FilterSpec(tags=("no-such-tag",)), limit=20
    )
    assert outcome.passages == ()


def test_the_date_window_uses_the_publication_time_set_on_each_file(built) -> None:
    dataset, _, handle = built
    window = items.FilterSpec(
        modified_after=dataset.articles[2].published_at.timestamp(),
        modified_before=dataset.articles[4].published_at.timestamp(),
    )
    outcome = pinakes_api.search_kb(handle, "article", window, limit=20)
    index = pipeline.path_index(dataset)
    assert {index[path] for path, _ in outcome.passages} <= {2, 3, 4}
    assert {index[path] for path, _ in outcome.passages} == {2, 3, 4}


def test_calibration_runs_end_to_end_and_pinakes_agrees_with_our_labels(built, tmp_path) -> None:
    """The whole stage: pinakes.calibrate on the fit half, re-derivation, the manifest write, and
    Pinakes' own `confidence` labels against ours on searches repeated with the thresholds in."""
    import tomllib

    from pnkbench import calibration

    dataset, _, handle = built
    index = pipeline.path_index(dataset)
    derived = items.derive_items(dataset)
    rows = [runner.run_one(handle, index, runner.to_payload(i), 20) for i in derived]
    kb_root = paths.kb_dir("baseline")
    result = calibration.calibrate(kb_root, tmp_path, derived, dataset, rows, index, limit=20)

    assert result["cross_check_against_stored_scores"]["agrees"] is True
    assert (
        result["cross_check_against_stored_scores"]["null_scores"] == 1
    )  # one null query is in the fit half
    parity = result["parity_with_pinakes_labels"]
    assert parity["checked"] == 10 and parity["label_mismatches"] == 0
    assert parity["max_abs_top1_difference_vs_recorded"] < 1e-4
    written = tomllib.loads((kb_root / "pinakes.toml").read_text(encoding="utf-8"))["retrieval"]
    assert written["confidence"]["fitted_for"] == result["thresholds"]["fitted_for"]
    assert result["thresholds"]["fitted_for"].startswith("BAAI/bge-reranker-base@")
    assert (tmp_path / "calibration-questions.yaml").exists()


def _write_toy_dataset_like_the_real_one(root: Path) -> None:
    """The toy dataset as raw files plus a manifest, so the production loader reads it."""
    import hashlib
    import json

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


def test_the_whole_pipeline_runs_on_a_dataset_laid_out_like_the_real_one(
    tmp_path, monkeypatch
) -> None:
    """build -> run -> calibrate -> score -> harness -> report, through the same functions `pnkbench all` calls."""
    import json

    root = tmp_path / "repo"
    root.mkdir()
    shutil.copytree(REAL_CONFIGS, root / "configs")
    monkeypatch.setenv(paths.ROOT_ENV, str(root))
    _write_toy_dataset_like_the_real_one(root)

    run_dir = pipeline.run_all("baseline", workers=1, log=lambda m: None)

    header = json.loads((run_dir / "run.json").read_text())
    assert header["status"] == "complete" and header["items"]["total"] == len(
        items.derive_items(make_dataset())
    )
    assert header["pinakes"]["source"] == "pypi" and len(header["pinakes"]["code_sha256"]) == 64
    assert header["settings"]["embedding"]["revision"] == "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    assert header["reproducibility"]["ranking_mismatches"] == 0
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["scored_by"]["benchmark"]["version"] and summary["scored_by"]["scored_utc"]
    assert summary["counts"]["base"] == 10 and summary["counts"]["null"] == 2
    assert summary["abstention"]["labels"] is not None
    for variant in items.FILTER_VARIANTS:
        assert summary["filters"][variant]["soundness_violations"] == 0
    harness_result = json.loads((run_dir / "harness.json").read_text())
    assert harness_result["metrics"]["questions"] == harness_result["questions"]
    report = (run_dir / "report.md").read_text()
    for heading in (
        "## What produced this",
        "## Answerable questions",
        "## Abstention",
        "## Filters",
        "## Latency",
    ):
        assert heading in report, heading
    for name in (
        "run.json",
        "rows.jsonl",
        "summary.json",
        "calibration.json",
        "harness.json",
        "report.md",
    ):
        assert (run_dir / name).stat().st_size > 0, name


def _toy_repo(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    shutil.copytree(REAL_CONFIGS, root / "configs")
    monkeypatch.setenv(paths.ROOT_ENV, str(root))
    _write_toy_dataset_like_the_real_one(root)
    return root


def test_a_resume_keeps_the_run_history_and_refuses_what_changed(tmp_path, monkeypatch) -> None:
    import json

    _toy_repo(tmp_path, monkeypatch)
    pipeline.build("baseline", log=lambda m: None)
    run_dir = pipeline.run("baseline", workers=1, log=lambda m: None)
    first = json.loads((run_dir / "run.json").read_text())

    assert pipeline.run("baseline", workers=1, resume=run_dir, log=lambda m: None) == run_dir
    second = json.loads((run_dir / "run.json").read_text())
    assert second["started_utc"] == first["started_utc"]  # the original start is kept
    assert len(second["execution"]["resumed_utc"]) == 1
    assert second["execution"]["earlier_wall_seconds"] == first["execution"]["wall_seconds"]
    assert (second["execution"]["ran"], second["execution"]["skipped"]) == (
        0,
        first["items"]["total"],
    )

    with pytest.raises(RuntimeError, match=r"cannot resume: sample differ"):
        pipeline.run("baseline", workers=1, sample=3, resume=run_dir, log=lambda m: None)

    manifest_file = paths.kb_dir("baseline") / "pinakes.toml"
    manifest_file.write_text(kb.set_key(manifest_file.read_text(), "retrieval", "final_k", "9"))
    with pytest.raises(RuntimeError, match=r"cannot resume: settings"):
        pipeline.run("baseline", workers=1, resume=run_dir, log=lambda m: None)


def test_calibration_stops_when_pinakes_and_we_disagree_about_a_label(
    built, tmp_path, monkeypatch
) -> None:
    from pnkbench import calibration

    dataset, _, handle = built
    index = pipeline.path_index(dataset)
    derived = items.derive_items(dataset)
    rows = [runner.run_one(handle, index, runner.to_payload(i), 20) for i in derived]
    disagreement = {
        "checked": 10,
        "label_mismatches": 1,
        "mismatch_ids": ["mhr-0001"],
        "max_abs_top1_difference_vs_recorded": 0.0,
    }
    monkeypatch.setattr(calibration, "parity_check", lambda *a, **k: disagreement)
    with pytest.raises(calibration.CalibrationError, match=r"differed from ours on 1 of 10"):
        calibration.calibrate(
            paths.kb_dir("baseline"), tmp_path, derived, dataset, rows, index, limit=20
        )
