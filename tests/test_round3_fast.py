"""Round three of the mutation review: code that no earlier round had mutated.

Fast, no models. The toy dataset facts used here are those of `tests/toy.py`: gold mhr-0000 {0,3},
0001 {1,2,3}, 0002 {1,4}, 0003 {5,8}, 0004 {0,4,8,11}, 0006 {0,4}, 0008 {10,11}, 0009 {0,11};
mhr-0005 and mhr-0007 are null. The fit/eval halves and the calibration order of its ten ids were
computed with the system `shasum`, not with the code under test:

    printf %s 'pnkbench-split-v1:order:mhr-0002' | shasum -a 256     (and likewise for the rest)

giving, from the smallest digest up: 0002, 0004, 0000, 0007, 0008, 0003, 0006, 0009, 0001, 0005;
the eval half is mhr-0000, 0001, 0004, 0007, 0008, 0009 and the fit half the other four.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ruamel.yaml import YAML
from toy import make_dataset

from pnkbench import (
    compare,
    harness,
    items,
    kb,
    layout,
    paths,
    pinakes_api,
    pipeline,
    scoring,
)
from pnkbench import config as config_module
from pnkbench.admission import admits

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


# --- pipeline.git_state ---------------------------------------------------------------------------


def test_outside_a_git_repository_there_is_no_commit_and_no_dirtiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(paths.ROOT_ENV, str(tmp_path))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))  # never find a parent repo
    assert pipeline.git_state() == {"commit": None, "dirty": None}


def test_inside_a_repository_the_state_is_the_head_commit_and_whether_the_tree_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@example.test")
    git("config", "user.name", "t")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("add", "code.py")
    git("commit", "-q", "-m", "init")
    monkeypatch.setenv(paths.ROOT_ENV, str(tmp_path))
    clean = pipeline.git_state()
    assert clean["dirty"] is False
    assert len(clean["commit"]) == 40 and set(clean["commit"]) <= set("0123456789abcdef")
    (tmp_path / "code.py").write_text("x = 2\n")
    assert pipeline.git_state()["dirty"] is True


# --- pipeline.select_items -------------------------------------------------------------------------

_FIRST_BY_HASH = ["mhr-0002", "mhr-0004", "mhr-0000", "mhr-0007"]  # the order, from shasum
_ITEMS_OF = {
    "mhr-0000": ["mhr-0000", "mhr-0000:f-cat-tag", "mhr-0000:f-cat-path", "mhr-0000:f-date"],
    "mhr-0002": ["mhr-0002", "mhr-0002:f-cat-tag", "mhr-0002:f-cat-path", "mhr-0002:f-date"],
    "mhr-0004": ["mhr-0004"],  # its gold spans every category, source and day: no variant
    "mhr-0007": ["mhr-0007"],  # a null question has no variant
}


def _ids(selected: list[items.RunItem]) -> list[str]:
    return [i.id for i in selected]


def test_a_sample_takes_the_first_n_base_queries_in_calibration_order_with_their_variants() -> None:
    dataset = make_dataset()
    three = pipeline.select_items(dataset, sample=3)
    # the three smallest digests are 0002, 0004 and 0000; the items keep the dataset's own order
    assert _ids(three) == [*_ITEMS_OF["mhr-0000"], *_ITEMS_OF["mhr-0002"], *_ITEMS_OF["mhr-0004"]]
    one = pipeline.select_items(dataset, sample=1)
    assert _ids(one) == _ITEMS_OF["mhr-0002"]
    four = pipeline.select_items(dataset, sample=4)
    assert {i.base_id for i in four} == set(_FIRST_BY_HASH)
    assert _ids(four)[-1] == "mhr-0007"


def test_a_sample_is_not_the_first_n_ids() -> None:
    chosen = {i.base_id for i in pipeline.select_items(make_dataset(), sample=3)}
    assert chosen == {"mhr-0002", "mhr-0004", "mhr-0000"}
    assert chosen != {"mhr-0000", "mhr-0001", "mhr-0002"}


def test_no_sample_or_a_sample_as_large_as_the_dataset_selects_everything() -> None:
    dataset = make_dataset()
    everything = _ids(items.derive_items(dataset))
    assert len(everything) == 23
    assert _ids(pipeline.select_items(dataset, sample=None)) == everything
    assert _ids(pipeline.select_items(dataset, sample=10)) == everything
    assert _ids(pipeline.select_items(dataset, sample=500)) == everything


def test_a_sample_of_zero_is_nothing_not_everything() -> None:
    assert pipeline.select_items(make_dataset(), sample=0) == []


# --- pipeline.pinakes_label and new_run_id ---------------------------------------------------------


@pytest.mark.parametrize(
    ("provenance", "label"),
    [
        ({"source": "pypi", "version": "0.32.8", "commit": None}, "pypi-0.32.8"),
        (
            {
                "source": "git",
                "version": "0.32.8",
                "commit": "b1bad6803346f1d3884ad9dbc64570fd0ec57aa4",
            },
            "git-b1bad68",
        ),
        ({"source": "git", "version": "0.32.8", "commit": None}, "git"),
        ({"source": "editable", "version": "0.32.8", "commit": None}, "editable"),
        ({"source": "url", "version": "0.32.8", "commit": None}, "url"),
    ],
)
def test_the_pinakes_label_is_the_version_for_pypi_seven_commit_digits_for_git_else_the_source(
    provenance: dict[str, Any], label: str
) -> None:
    assert pipeline.pinakes_label(provenance) == label


@pytest.mark.parametrize(
    ("moment", "stamp"),
    [
        (datetime(2026, 10, 30, 21, 5, tzinfo=UTC), "20261030_2105"),  # a 24-hour clock
        (datetime(2026, 10, 30, 9, 5, tzinfo=UTC), "20261030_0905"),
        (datetime(2026, 10, 30, 0, 7, tzinfo=UTC), "20261030_0007"),
        (datetime(2026, 1, 2, 13, 45, tzinfo=UTC), "20260102_1345"),
    ],
)
def test_a_run_id_is_utc_time_then_configuration_then_pinakes(
    monkeypatch: pytest.MonkeyPatch, moment: datetime, stamp: str
) -> None:
    monkeypatch.setattr(pipeline, "utc_now", lambda: moment)
    provenance = {"source": "pypi", "version": "0.32.8", "commit": None}
    assert pipeline.new_run_id("baseline", provenance) == f"{stamp}-baseline-pypi-0.32.8"


# --- a toy repository on disk, for the steps that read a run directory ------------------------------


def _write_toy_dataset_like_the_real_one(root: Path) -> None:
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
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    _write_toy_dataset_like_the_real_one(root)
    (root / "runs").mkdir()
    return root


def _toy_rows() -> list[dict[str, Any]]:
    """A row for every derived toy item: the gold first for a base item that has any, the gold that
    the filter admits for a variant, and one unrelated article for a null question."""
    dataset = make_dataset()
    info = scoring.query_info(dataset)
    rows = []
    for item in items.derive_items(dataset):
        gold = sorted(info[item.base_id].gold)
        if item.variant == items.BASE:
            passages = _ps(*gold, top=2.0) if gold else _ps(9, top=-5.0)
            rows.append(_row(item.id, passages))
            continue
        admitted = {
            a.index
            for a in dataset.articles
            if admits(item.filters, a, layout.tags_for(a), layout.doc_relpath(a))
        }
        rows.append(_row(item.id, _ps(*[g for g in gold if g in admitted]), item.variant))
    return rows


def _finished_run(
    name: str, *, sample: int | None = None, rows: list[dict[str, Any]] | None = None
) -> Path:
    run_dir = paths.runs_dir() / name
    run_dir.mkdir()
    header = {
        "run_id": name,
        "status": "complete",
        "config": {"name": "baseline", "chunking_metadata": "off"},
        "settings": {"retrieval": {"fusion_top_k": 20}},
        "execution": {"passages_per_search": 20, "sample_base_queries": sample},
    }
    (run_dir / "run.json").write_text(json.dumps(header))
    body = rows if rows is not None else _toy_rows()
    (run_dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in body))
    return run_dir


# --- pipeline._context --------------------------------------------------------------------------


def test_the_context_of_a_sampled_run_holds_only_the_sampled_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _toy_repo(tmp_path, monkeypatch)
    sampled = _finished_run("sampled", sample=3)
    header, dataset, selected, index = pipeline._context(sampled)
    assert header["execution"]["sample_base_queries"] == 3
    assert _ids(selected) == [
        *_ITEMS_OF["mhr-0000"],
        *_ITEMS_OF["mhr-0002"],
        *_ITEMS_OF["mhr-0004"],
    ]
    assert len(dataset.articles) == 12
    assert index == {layout.doc_relpath(a): a.index for a in make_dataset().articles}


def test_the_context_of_a_whole_run_holds_every_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _toy_repo(tmp_path, monkeypatch)
    _, _, selected, _ = pipeline._context(_finished_run("whole", sample=None))
    assert len(selected) == 23


def test_the_context_of_a_run_that_did_not_finish_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _toy_repo(tmp_path, monkeypatch)
    run_dir = _finished_run("running")
    header = json.loads((run_dir / "run.json").read_text())
    header["status"] = "running"
    (run_dir / "run.json").write_text(json.dumps(header))
    with pytest.raises(RuntimeError, match=r"running is not a complete run \(status 'running'\)"):
        pipeline._context(run_dir)


# --- pipeline.score_step ---------------------------------------------------------------------------


def test_scoring_reads_the_thresholds_from_calibration_json_when_it_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _toy_repo(tmp_path, monkeypatch)
    run_dir = _finished_run("calibrated")
    (run_dir / "calibration.json").write_text(
        json.dumps({"thresholds": {"low_below": -3.0, "high_above": 1.0}})
    )
    summary = pipeline.score_step(run_dir)
    assert summary["abstention"]["thresholds"] == {"low_below": -3.0, "high_above": 1.0}
    # the eight answerable questions score 2.0 (above 1.0: high) and the two null ones -5.0
    # (below -3.0: low), whichever half they are in
    labels = summary["abstention"]["labels"]
    assert (
        sum(labels["eval"]["answerable"].values()) + sum(labels["fit"]["answerable"].values()) == 8
    )
    assert labels["eval"]["answerable"]["high"] + labels["fit"]["answerable"]["high"] == 8
    assert labels["eval"]["null"]["low"] + labels["fit"]["null"]["low"] == 2
    assert json.loads((run_dir / "summary.json").read_text()) == summary


def test_scoring_without_a_calibration_has_no_thresholds_and_no_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _toy_repo(tmp_path, monkeypatch)
    run_dir = _finished_run("uncalibrated")
    summary = pipeline.score_step(run_dir)
    assert summary["abstention"]["labels"] is None
    assert "thresholds" not in summary["abstention"]
    assert summary["counts"]["base"] == 10 and summary["counts"]["rows"] == 23


def test_scoring_records_which_commit_scored_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _toy_repo(tmp_path, monkeypatch)
    summary = pipeline.score_step(_finished_run("scored"))
    # the toy repository is not a git repository, so no commit and no dirtiness are recorded
    assert summary["scored_by"]["benchmark"]["commit"] is None
    assert summary["scored_by"]["benchmark"]["dirty"] is None
    assert summary["scored_by"]["scored_utc"].endswith("+00:00")


# --- kb.verify_kb: the order of a document's tags -------------------------------------------------


def _index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tags: list[str]) -> None:
    """An index of the twelve toy articles in which article 0 carries `tags`."""
    dataset = make_dataset()
    database = tmp_path / "index.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE documents (path, title, mtime, metadata, state, source_type)")
    connection.execute("CREATE TABLE chunks (id)")
    connection.execute("CREATE TABLE failures (id)")
    for article in dataset.articles:
        found = tags if article.index == 0 else list(layout.tags_for(article))
        connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, 'active', 'text')",
            (
                layout.doc_relpath(article),
                article.title,
                article.published_at.timestamp(),
                json.dumps({"tags": found}),
            ),
        )
    connection.commit()
    connection.close()
    monkeypatch.setattr(pinakes_api, "index_path", lambda root: database)


def test_a_document_whose_tags_come_in_another_order_is_not_the_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # layout.tags_for gives the category and then the source: ("sports", "source-the-verge")
    _index(tmp_path, monkeypatch, ["source-the-verge", "sports"])
    with pytest.raises(kb.BuildError, match=r"tags \('source-the-verge', 'sports'\)"):
        kb.verify_kb(make_dataset(), tmp_path)


def test_a_document_whose_tags_are_in_the_dataset_order_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _index(tmp_path, monkeypatch, ["sports", "source-the-verge"])
    assert kb.verify_kb(make_dataset(), tmp_path) == {"documents": 12, "chunks": 0}


# --- harness.questions_yaml ----------------------------------------------------------------------------


def _questions(selected: list[items.RunItem], dataset: Any = None) -> list[dict[str, Any]]:
    text = harness.questions_yaml(dataset or make_dataset(), selected)
    assert text.startswith("# Generated by pnkbench")
    return YAML(typ="safe").load(text)["questions"]


def _relpaths(*indices: int) -> list[str]:
    articles = make_dataset().articles
    return [layout.doc_relpath(articles[i]) for i in indices]


def test_only_the_base_and_the_tag_variants_are_exported_in_the_datasets_order() -> None:
    entries = _questions(items.derive_items(make_dataset()))
    assert [e["id"] for e in entries] == [
        "mhr-0000",
        "mhr-0000:f-cat-tag",
        "mhr-0001",
        "mhr-0002",
        "mhr-0002:f-cat-tag",
        "mhr-0003",
        "mhr-0003:f-cat-tag",
        "mhr-0004",
        "mhr-0005",
        "mhr-0006",
        "mhr-0006:f-src-tag",
        "mhr-0007",
        "mhr-0008",
        "mhr-0009",
    ]  # the path-prefix and date variants have no spelling in Pinakes' questions.yaml


def test_each_item_is_exported_with_the_kind_its_shape_calls_for() -> None:
    kinds = {e["id"]: e["kind"] for e in _questions(items.derive_items(make_dataset()))}
    assert kinds["mhr-0005"] == kinds["mhr-0007"] == "no-answer"  # the null questions
    assert kinds["mhr-0000"] == kinds["mhr-0004"] == "multi-hop"  # base items with an answer
    assert kinds["mhr-0000:f-cat-tag"] == kinds["mhr-0006:f-src-tag"] == "filter"


def test_an_answerable_item_expects_its_gold_articles_paths_and_a_null_one_expects_none() -> None:
    by_id = {e["id"]: e for e in _questions(items.derive_items(make_dataset()))}
    assert by_id["mhr-0000"]["expect"] == _relpaths(0, 3)
    assert by_id["mhr-0001"]["expect"] == _relpaths(1, 2, 3)
    assert by_id["mhr-0003"]["expect"] == _relpaths(5, 8)  # article 5 is listed twice: once
    assert by_id["mhr-0004"]["expect"] == _relpaths(0, 4, 8, 11)
    assert by_id["mhr-0000:f-cat-tag"]["expect"] == _relpaths(0, 3)
    assert "expect" not in by_id["mhr-0005"] and "expect" not in by_id["mhr-0007"]


def test_the_expected_paths_follow_the_order_the_evidence_lists_the_articles() -> None:
    dataset = make_dataset()
    first = dataset.queries[0]  # evidence is articles 0 then 3
    flipped = dataclasses.replace(first, evidence=tuple(reversed(first.evidence)))
    reordered = dataclasses.replace(dataset, queries=(flipped, *dataset.queries[1:]))
    base = [i for i in items.derive_items(reordered) if i.id == "mhr-0000"]
    assert _questions(base, reordered)[0]["expect"] == _relpaths(3, 0)


def test_only_a_tag_variant_carries_filters_and_they_are_its_tags() -> None:
    by_id = {e["id"]: e for e in _questions(items.derive_items(make_dataset()))}
    assert by_id["mhr-0000:f-cat-tag"]["filters"] == {"tags": ["sports"]}
    assert by_id["mhr-0003:f-cat-tag"]["filters"] == {"tags": ["business"]}
    assert by_id["mhr-0006:f-src-tag"]["filters"] == {"tags": ["source-the-verge"]}
    for plain in ("mhr-0000", "mhr-0004", "mhr-0005"):
        assert "filters" not in by_id[plain], plain


def test_each_entry_has_exactly_the_keys_of_its_kind() -> None:
    keys = {e["id"]: set(e) for e in _questions(items.derive_items(make_dataset()))}
    assert keys["mhr-0005"] == {"id", "question", "kind"}
    assert keys["mhr-0000"] == {"id", "question", "kind", "expect"}
    assert keys["mhr-0000:f-cat-tag"] == {"id", "question", "kind", "expect", "filters"}


def test_an_entrys_question_is_the_items_text() -> None:
    by_id = {e["id"]: e for e in _questions(items.derive_items(make_dataset()))}
    assert by_id["mhr-0000"]["question"] == "q0 sports pair"
    assert by_id["mhr-0000:f-cat-tag"]["question"] == "q0 sports pair"
    assert by_id["mhr-0005"]["question"] == "q5 null"


def test_no_items_export_an_empty_list() -> None:
    assert _questions([]) == []


def test_pinakes_eval_is_asked_for_five_passages() -> None:
    assert harness.HARNESS_K == 5


def test_run_harness_scores_the_eval_half_with_five_passages_and_reports_what_it_exported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen["command"], seen["kwargs"] = command, kwargs
        return subprocess.CompletedProcess(
            command, 0, stdout='{"recall": 0.5}\n\nwrote the baseline', stderr=""
        )

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    dataset = make_dataset()
    result = harness.run_harness(tmp_path / "kb", tmp_path, dataset, items.derive_items(dataset))
    out = tmp_path / "harness"
    assert seen["command"] == [
        sys.executable,
        "-m",
        "pinakes.eval",
        str(tmp_path / "kb"),
        "--questions",
        str(out / "questions.yaml"),
        "-k",
        "5",
        "--write-baseline",
        "--baseline",
        str(out / "baseline.json"),
        "--outcomes",
        str(out / "outcomes.json"),
    ]
    assert seen["kwargs"] == {"capture_output": True, "text": True, "check": False}
    # the eval half is mhr-0000, 0001, 0004, 0007, 0008, 0009; of their variants only the
    # category-tag one of mhr-0000 is exportable (mhr-0006, the source-tag one, is in the fit half)
    exported = [
        e["id"] for e in YAML(typ="safe").load((out / "questions.yaml").read_text())["questions"]
    ]
    assert exported == [
        "mhr-0000",
        "mhr-0000:f-cat-tag",
        "mhr-0001",
        "mhr-0004",
        "mhr-0007",
        "mhr-0008",
        "mhr-0009",
    ]
    assert result["k"] == 5 and result["questions"] == 7
    assert result["by_variant"] == {"base": 6, "f-cat-tag": 1, "f-src-tag": 0}
    assert result["metrics"] == {"recall": 0.5}


def test_run_harness_reports_a_failing_evaluator_with_the_tail_of_its_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="x" * 1200 + "THE-END")

    monkeypatch.setattr(harness.subprocess, "run", fake_run)
    dataset = make_dataset()
    with pytest.raises(RuntimeError) as caught:
        harness.run_harness(tmp_path / "kb", tmp_path, dataset, items.derive_items(dataset))
    message = str(caught.value)
    assert message.startswith("pinakes.eval exited 2:\n") and message.endswith("THE-END")
    assert len(message.split("\n", 1)[1]) == 1000


# --- pinakes_api.code_fingerprint ----------------------------------------------------------------------


def _package(directory: Path, files: dict[str, bytes]) -> Path:
    for name, data in files.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return directory


@pytest.fixture
def fingerprint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """`fingerprint(files)` is the code fingerprint of a fresh package holding those files."""
    count = 0

    def of(files: dict[str, bytes]) -> str:
        nonlocal count
        count += 1
        root = _package(tmp_path / f"pkg{count}", files)
        monkeypatch.setattr(pinakes_api, "_package_dir", lambda: root)
        return pinakes_api.code_fingerprint()

    return of


def test_the_fingerprint_is_the_sha256_of_each_files_path_and_bytes_in_sorted_order(
    fingerprint: Any,
) -> None:
    got = fingerprint({"b/c.py": b"BBB", "a.py": b"AAA"})
    expected = hashlib.sha256(b"a.py\0AAA\0b/c.py\0BBB\0").hexdigest()
    assert got == expected


def test_the_fingerprint_changes_with_a_files_bytes(fingerprint: Any) -> None:
    assert fingerprint({"a.py": b"one"}) != fingerprint({"a.py": b"two"})


def test_the_fingerprint_changes_with_a_files_relative_path(fingerprint: Any) -> None:
    assert fingerprint({"a.py": b"same"}) != fingerprint({"b.py": b"same"})
    assert fingerprint({"x/a.py": b"same"}) != fingerprint({"y/a.py": b"same"})


def test_the_fingerprint_keeps_a_path_apart_from_the_bytes_that_follow_it(
    fingerprint: Any,
) -> None:
    # "a.py" then "b.py" and "a.pyb.py" then nothing read the same without a separator
    assert fingerprint({"a.py": b"b.py"}) != fingerprint({"a.pyb.py": b""})


def test_the_fingerprint_ignores_files_that_are_not_python_source(fingerprint: Any) -> None:
    plain = fingerprint({"a.py": b"AAA"})
    extra = fingerprint(
        {
            "a.py": b"AAA",
            "README.md": b"docs",
            "data.json": b"{}",
            "__pycache__/a.cpython-313.pyc": b"\x00\x01",
            "notes.py.txt": b"text",
        }
    )
    assert extra == plain


def test_the_fingerprint_counts_python_files_in_every_subdirectory(fingerprint: Any) -> None:
    assert fingerprint({"a.py": b"A"}) != fingerprint({"a.py": b"A", "deep/er/b.py": b"B"})


def test_the_fingerprint_does_not_depend_on_the_order_the_filesystem_lists_files(
    fingerprint: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {"a.py": b"A", "b.py": b"B", "sub/c.py": b"C"}
    real = Path.rglob

    def listed(reverse: bool) -> Any:
        return lambda self, pattern: iter(sorted(real(self, pattern), reverse=reverse))

    monkeypatch.setattr(Path, "rglob", listed(reverse=False))
    forwards = fingerprint(files)
    monkeypatch.setattr(Path, "rglob", listed(reverse=True))
    assert fingerprint(files) == forwards


def test_the_package_directory_is_where_the_import_system_says_pinakes_lives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    where = SimpleNamespace(submodule_search_locations=[str(tmp_path), "/elsewhere"])
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: where)
    assert pinakes_api._package_dir() == tmp_path


@pytest.mark.parametrize("spec", [None, SimpleNamespace(submodule_search_locations=[])])
def test_a_missing_pinakes_is_reported_as_not_installed(
    monkeypatch: pytest.MonkeyPatch, spec: Any
) -> None:
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: spec)
    with pytest.raises(pinakes_api.InterfaceError, match="pinakes is not installed"):
        pinakes_api._package_dir()


# --- pinakes_api.provenance ----------------------------------------------------------------------------


class _Distribution:
    version = "9.8.7"

    def __init__(self, direct_url: str | None) -> None:
        self.direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        assert name == "direct_url.json"
        return self.direct_url


def _provenance(monkeypatch: pytest.MonkeyPatch, direct_url: dict[str, Any] | str | None) -> Any:
    text = json.dumps(direct_url) if isinstance(direct_url, dict) else direct_url
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: _Distribution(text))
    monkeypatch.setattr(pinakes_api, "code_fingerprint", lambda: "f" * 64)
    return pinakes_api.provenance()


def test_a_pinakes_with_no_direct_url_came_from_the_index(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _provenance(monkeypatch, None) == {
        "version": "9.8.7",
        "source": "pypi",
        "commit": None,
        "url": None,
        "code_sha256": "f" * 64,
    }
    assert _provenance(monkeypatch, "")["source"] == "pypi"


def test_a_pinakes_installed_from_git_records_the_commit_it_resolved_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = {
        "url": "https://github.com/example/pinakes",
        "vcs_info": {"vcs": "git", "commit_id": "abc123", "requested_revision": "main"},
    }
    assert _provenance(monkeypatch, direct) == {
        "version": "9.8.7",
        "source": "git",
        "commit": "abc123",
        "url": "https://github.com/example/pinakes",
        "code_sha256": "f" * 64,
    }


def test_an_editable_install_is_recorded_as_editable_with_no_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct = {"url": "file:///work/pinakes", "dir_info": {"editable": True}}
    assert _provenance(monkeypatch, direct) == {
        "version": "9.8.7",
        "source": "editable",
        "commit": None,
        "url": "file:///work/pinakes",
        "code_sha256": "f" * 64,
    }


@pytest.mark.parametrize(
    "direct",
    [
        {"url": "https://example.test/pinakes-9.8.7.whl", "archive_info": {}},
        {"url": "file:///work/pinakes", "dir_info": {}},
        {"url": "file:///work/pinakes", "dir_info": {"editable": False}},
    ],
)
def test_any_other_direct_url_is_recorded_as_a_url(
    monkeypatch: pytest.MonkeyPatch, direct: dict[str, Any]
) -> None:
    got = _provenance(monkeypatch, direct)
    assert (got["source"], got["commit"], got["url"]) == ("url", None, direct["url"])


# --- pinakes_api.require_interfaces ----------------------------------------------------------------


def test_the_installed_pinakes_satisfies_every_interface_the_benchmark_calls() -> None:
    pytest.importorskip("pinakes")
    assert pinakes_api.require_interfaces() is None


def test_a_name_pinakes_no_longer_has_is_named_in_the_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pinakes")
    required = pinakes_api._REQUIRED_NAMES["pinakes.search"]
    monkeypatch.setitem(pinakes_api._REQUIRED_NAMES, "pinakes.search", (*required, "vanished"))
    with pytest.raises(pinakes_api.InterfaceError) as caught:
        pinakes_api.require_interfaces()
    message = str(caught.value)
    assert message.startswith("the installed Pinakes does not match what the benchmark calls:")
    assert "\n  - pinakes.search has no vanished" in message
    assert message.endswith("the one place that depends on Pinakes.")


def test_a_module_pinakes_no_longer_has_is_named_in_the_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pinakes")
    monkeypatch.setitem(pinakes_api._REQUIRED_NAMES, "pinakes.not_there", ("anything",))
    with pytest.raises(pinakes_api.InterfaceError, match=r"cannot import pinakes\.not_there"):
        pinakes_api.require_interfaces()


def test_a_search_that_lost_a_parameter_is_named_but_one_that_gained_one_is_fine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_module = pytest.importorskip("pinakes.search")

    def lost(connection, manifest, query, backend, reranker, filters):  # no `limit`
        raise AssertionError("never called")

    monkeypatch.setattr(search_module, "search", lost)
    with pytest.raises(pinakes_api.InterfaceError) as caught:
        pinakes_api.require_interfaces()
    assert "pinakes.search.search lost parameters ['limit']" in str(caught.value)

    def gained(connection, manifest, query, backend, reranker, filters, limit, brand_new=None):
        raise AssertionError("never called")

    monkeypatch.setattr(search_module, "search", gained)
    assert pinakes_api.require_interfaces() is None


def test_filters_and_sidecars_that_lost_fields_are_named(monkeypatch: pytest.MonkeyPatch) -> None:
    search_module = pytest.importorskip("pinakes.search")
    sidecar_module = pytest.importorskip("pinakes.sidecar")
    fields = [f.name for f in dataclasses.fields(search_module.Filters) if f.name != "tags"]
    monkeypatch.setattr(
        search_module,
        "Filters",
        dataclasses.make_dataclass("Filters", [(n, Any, None) for n in fields]),
    )
    with pytest.raises(pinakes_api.InterfaceError) as caught:
        pinakes_api.require_interfaces()
    assert "pinakes.search.Filters lost fields ['tags']" in str(caught.value)
    monkeypatch.undo()

    kept = [f.name for f in dataclasses.fields(sidecar_module.Sidecar) if f.name != "provenance"]
    monkeypatch.setattr(
        sidecar_module,
        "Sidecar",
        dataclasses.make_dataclass("Sidecar", [(n, Any, None) for n in kept]),
    )
    with pytest.raises(pinakes_api.InterfaceError) as caught:
        pinakes_api.require_interfaces()
    assert "pinakes.sidecar.Sidecar lost fields ['provenance']" in str(caught.value)


# --- config.load_models and config.parse -----------------------------------------------------------------


def _models_toml(directory: Path, embedding: str, rerank: str = "b" * 40) -> Path:
    (directory / "models.toml").write_text(
        f'[embedding]\nmodel = "e"\nrevision = "{embedding}"\n\n'
        f'[rerank]\nmodel = "r"\nrevision = "{rerank}"\n',
        encoding="utf-8",
    )
    return directory


def test_a_full_forty_hex_revision_is_accepted_for_both_models(tmp_path: Path) -> None:
    models = config_module.load_models(_models_toml(tmp_path, "0123456789abcdef" * 2 + "01234567"))
    assert models.embedding.revision == "0123456789abcdef0123456789abcdef01234567"
    assert (models.embedding.model, models.rerank.model, models.rerank.revision) == (
        "e",
        "r",
        "b" * 40,
    )


@pytest.mark.parametrize(
    "revision", ["a" * 41, "a" * 39, "a" * 80, "", "z" * 40, "A" * 40, "g" + "a" * 39]
)
def test_a_revision_that_is_not_exactly_forty_lowercase_hex_digits_is_refused(
    tmp_path: Path, revision: str
) -> None:
    with pytest.raises(
        config_module.ConfigError, match=r"\[embedding\] revision must be a full 40-hex"
    ):
        config_module.load_models(_models_toml(tmp_path, revision))
    with pytest.raises(
        config_module.ConfigError, match=r"\[rerank\] revision must be a full 40-hex"
    ):
        config_module.load_models(_models_toml(tmp_path, "a" * 40, rerank=revision))


@pytest.mark.parametrize("backend", ["st", "light"])
def test_the_two_backends_pinakes_offers_are_accepted(backend: str) -> None:
    assert config_module.parse("c", {"backend": backend}).backend == backend
    assert config_module.parse("c", {}).backend == "st"


@pytest.mark.parametrize("backend", ["fastembed", "ST", "", "openai"])
def test_any_other_backend_is_refused_with_the_two_that_are_allowed(backend: str) -> None:
    with pytest.raises(
        config_module.ConfigError, match=r"backend must be one of \('st', 'light'\)"
    ):
        config_module.parse("c", {"backend": backend})


# --- compare against a real Pinakes schema, built with Pinakes' own pieces ---------------------------------


class _OneTokenPerWord:
    """A token counter that needs no model: a text is as many tokens as it has words."""

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_the_comparison_finds_the_one_article_a_lower_max_tokens_splits(tmp_path: Path) -> None:
    """A guard against schema drift: both indexes are made by Pinakes' own DDL and chunker."""
    chunk = pytest.importorskip("pinakes.chunk")
    store = pytest.importorskip("pinakes.store")
    short = "A short article.\n\nTwo small paragraphs."
    long = " ".join(f"w{n}" for n in range(450))  # one paragraph of 450 tokens
    bodies = {"docs/a.txt": short, "docs/long.txt": long, "docs/c.txt": short}

    def build(name: str, max_tokens: int) -> Path:
        connection = store.create(tmp_path / name)
        for number, (path, body) in enumerate(bodies.items()):
            connection.execute(
                "INSERT INTO documents (id, path, content_hash, mtime, source_type) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"id{number}", path, "h", 0.0, "text"),
            )
            chunks = chunk.chunk_document(
                body, counter=_OneTokenPerWord(), max_tokens=max_tokens, overlap=64, kind="text"
            )
            store.replace_chunks(connection, f"id{number}", [c.as_row() for c in chunks])
        connection.commit()
        connection.close()
        return tmp_path / name

    roomy, tight = build("roomy.db", 510), build("tight.db", 400)
    at_510, at_400 = compare.chunk_spans(roomy), compare.chunk_spans(tight)
    assert set(at_510) == set(at_400) == set(bodies)
    # 450 tokens fit in 510: one chunk covering the text. At 400 with an overlap of 64 the first
    # chunk takes 400 tokens and the second starts 64 tokens back: ceil((450 - 64) / (400 - 64)) = 2
    assert at_510["docs/long.txt"] == frozenset({(0, len(long))})
    assert len(at_400["docs/long.txt"]) == 2
    assert min(start for start, _ in at_400["docs/long.txt"]) == 0
    assert max(end for _, end in at_400["docs/long.txt"]) == len(long)
    assert at_510["docs/a.txt"] == at_400["docs/a.txt"] == at_510["docs/c.txt"]

    index = {"docs/a.txt": 0, "docs/long.txt": 1, "docs/c.txt": 2}
    assert compare.articles_chunked_differently(roomy, tight, index) == frozenset({1})
    assert compare.articles_chunked_differently(roomy, roomy, index) == frozenset()
