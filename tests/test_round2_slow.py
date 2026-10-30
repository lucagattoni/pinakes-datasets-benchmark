"""Round two of the mutation review, for the code that needs the real models.

Real Pinakes, the real bge models, a 12-article toy corpus laid out like the real dataset. Run with
`pytest -m slow tests/test_round2_slow.py` (about a minute: one build per configuration and one
search run). What is pinned here is what the fast file can only imitate with stand-ins: that a
resume really refuses a changed dataset, another Pinakes build or another configuration, that its
bookkeeping survives several resumes of a real run, and that the comparison's SQL reads a real index.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from toy import make_dataset

pytestmark = pytest.mark.slow

pytest.importorskip("pinakes")

from pnkbench import compare, layout, paths, pinakes_api, pipeline  # noqa: E402
from pnkbench import multihop_rag as mh  # noqa: E402

REAL_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _write_toy_dataset_like_the_real_one(root: Path) -> None:
    """The toy dataset as raw files plus a manifest, so the production loader reads it."""
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


def _quiet(message: str) -> None:
    pass


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """A toy repository with both shipped configurations built and the baseline run once."""
    root = tmp_path_factory.mktemp("repo")
    shutil.copytree(REAL_CONFIGS, root / "configs")
    patch = pytest.MonkeyPatch()
    patch.setenv(paths.ROOT_ENV, str(root))
    _write_toy_dataset_like_the_real_one(root)
    pipeline.build("baseline", log=_quiet)
    pipeline.build("metadata-prefix", log=_quiet)
    run_dir = pipeline.run("baseline", workers=1, log=_quiet)
    yield {"root": root, "run_dir": run_dir}
    patch.undo()


def _header(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "run.json").read_text())


def test_a_resume_refuses_when_the_dataset_files_differ(world: dict[str, Any]) -> None:
    folder = world["root"] / "datasets" / mh.DATASET
    corpus_file, manifest_file = folder / "raw" / "corpus.json", folder / "manifest.json"
    original_corpus, original_manifest = corpus_file.read_bytes(), manifest_file.read_bytes()
    try:
        # change one article's body, and the manifest's size and checksum with it, so that the
        # loader accepts the files: only the record in the earlier run's header can tell
        corpus = json.loads(original_corpus)
        corpus[5]["body"] += "\n\nA paragraph that was not in the dataset the run searched."
        data = json.dumps(corpus).encode()
        corpus_file.write_bytes(data)
        manifest = json.loads(original_manifest)
        entry = next(f for f in manifest["files"] if f["role"] == "corpus")
        entry["bytes"], entry["sha256"] = len(data), hashlib.sha256(data).hexdigest()
        manifest_file.write_text(json.dumps(manifest))
        assert len(mh.load().articles) == 12  # the changed dataset is a valid one

        before = _header(world["run_dir"])
        with pytest.raises(
            RuntimeError, match=r"cannot resume: dataset files differ from the run being continued"
        ):
            pipeline.run("baseline", workers=1, resume=world["run_dir"], log=_quiet)
        assert _header(world["run_dir"]) == before  # a refused resume writes nothing
    finally:
        corpus_file.write_bytes(original_corpus)
        manifest_file.write_bytes(original_manifest)


def test_a_resume_refuses_a_run_started_by_another_build_of_pinakes(
    world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = pinakes_api.provenance()
    assert _header(world["run_dir"])["pinakes"]["code_sha256"] == real["code_sha256"]
    monkeypatch.setattr(
        pinakes_api, "provenance", lambda: {**real, "code_sha256": real["code_sha256"][::-1]}
    )
    with pytest.raises(RuntimeError, match=r"cannot resume: the installed Pinakes differs"):
        pipeline.run("baseline", workers=1, resume=world["run_dir"], log=_quiet)


def test_a_resume_refuses_a_run_made_with_another_configuration(world: dict[str, Any]) -> None:
    # the metadata-prefix knowledge base exists (the fixture built it), so only the name can differ
    assert (paths.kb_dir("metadata-prefix") / "pinakes.toml").exists()
    with pytest.raises(RuntimeError, match=r"cannot resume: this run used another configuration"):
        pipeline.run("metadata-prefix", workers=1, resume=world["run_dir"], log=_quiet)


def test_each_resume_of_a_real_run_adds_to_the_history_and_to_the_wall_time(
    world: dict[str, Any],
) -> None:
    run_dir = world["run_dir"]
    before = _header(run_dir)
    assert pipeline.run("baseline", workers=1, resume=run_dir, log=_quiet) == run_dir
    once = _header(run_dir)
    assert pipeline.run("baseline", workers=1, resume=run_dir, log=_quiet) == run_dir
    twice = _header(run_dir)

    resumed_before = before["execution"].get("resumed_utc", [])
    assert len(once["execution"]["resumed_utc"]) == len(resumed_before) + 1
    assert len(twice["execution"]["resumed_utc"]) == len(resumed_before) + 2
    assert (
        twice["execution"]["resumed_utc"][: len(resumed_before) + 1]
        == (once["execution"]["resumed_utc"])
    )  # earlier resumes are carried over, not replaced
    assert once["started_utc"] == twice["started_utc"] == before["started_utc"]

    # the wall time of every earlier session is added up: earlier + the last session's own
    def earlier(header: dict[str, Any]) -> float:
        return header["execution"].get("earlier_wall_seconds", 0)

    assert once["execution"]["earlier_wall_seconds"] == round(
        earlier(before) + before["execution"]["wall_seconds"], 1
    )
    assert twice["execution"]["earlier_wall_seconds"] == round(
        earlier(once) + once["execution"]["wall_seconds"], 1
    )
    # nothing was missing, so a resume searched nothing and skipped everything
    assert (twice["execution"]["ran"], twice["execution"]["skipped"]) == (
        0,
        twice["items"]["total"],
    )


# --- the comparison's SQL against a real index ----------------------------------------------------


def _index_of(config_name: str) -> Path:
    return pinakes_api.index_path(paths.kb_dir(config_name))


def test_chunk_spans_reads_every_toy_article_from_a_real_index(world: dict[str, Any]) -> None:
    dataset = make_dataset()
    spans = compare.chunk_spans(_index_of("baseline"))
    assert set(spans) == {layout.doc_relpath(a) for a in dataset.articles}
    for article in dataset.articles:
        found = sorted(spans[layout.doc_relpath(article)])
        # three paragraphs, so at least three chunks: Pinakes splits plain text by paragraph
        assert len(found) >= 3, article.index
        assert all(0 <= start < end <= len(article.body) for start, end in found), article.index
        assert found[0][0] == 0, article.index  # the first chunk starts the document


def test_two_real_indexes_that_chunk_alike_set_no_article_aside(world: dict[str, Any]) -> None:
    # baseline (max_tokens 510) and metadata-prefix (459) cut three short paragraphs the same way
    index = pipeline.path_index(make_dataset())
    assert (
        compare.articles_chunked_differently(
            _index_of("baseline"), _index_of("metadata-prefix"), index
        )
        == frozenset()
    )


def test_a_real_index_is_read_without_being_changed(world: dict[str, Any]) -> None:
    index = _index_of("baseline")
    before = index.read_bytes()
    compare.chunk_spans(index)
    assert index.read_bytes() == before
    connection = sqlite3.connect(f"file:{index}?mode=ro", uri=True)
    try:
        assert connection.execute("SELECT count(*) FROM documents").fetchone()[0] == 12
    finally:
        connection.close()
