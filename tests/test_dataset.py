from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from pnkbench import multihop_rag as mh


def _article(i: int, **over: Any) -> dict[str, Any]:
    base = {
        "title": f"Title {i}",
        "author": None if i == 0 else "A. Writer",
        "source": "The Verge",
        "published_at": f"2023-10-{i + 1:02d}T09:30:00+00:00",
        "category": "technology",
        "url": f"https://example.test/{i}",
        "body": f"Body of article {i}.\n\nSecond paragraph.",
    }
    base.update(over)
    return base


def _evidence(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": article["title"],
        "url": article["url"],
        "fact": "a fact",
        "author": None,
        "source": article["source"],
        "category": article["category"],
        "published_at": article["published_at"],
    }


def _write(
    tmp_path: Path, corpus: list[dict[str, Any]], queries: list[dict[str, Any]]
) -> mh.DatasetManifest:
    files = []
    for name, role, payload in (
        ("corpus.json", "corpus", corpus),
        ("MultiHopRAG.json", "queries", queries),
    ):
        data = json.dumps(payload).encode()
        (tmp_path / name).write_bytes(data)
        files.append(mh.FileSpec(name, role, len(data), hashlib.sha256(data).hexdigest()))
    return mh.DatasetManifest("owner/repo", "a" * 40, "ODC-By", tuple(files))


def _good() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corpus = [_article(i) for i in range(3)]
    queries = [
        {
            "query": "q?",
            "answer": "yes",
            "question_type": "comparison_query",
            "evidence_list": [_evidence(corpus[0]), _evidence(corpus[1])],
        },
        {
            "query": "n?",
            "answer": "Insufficient information.",
            "question_type": "null_query",
            "evidence_list": [],
        },
    ]
    return corpus, queries


def test_load_reads_a_consistent_dataset(tmp_path: Path) -> None:
    manifest = _write(tmp_path, *_good())
    dataset = mh.load(tmp_path, manifest)
    assert [a.title for a in dataset.articles] == ["Title 0", "Title 1", "Title 2"]
    assert dataset.articles[0].author is None and dataset.articles[1].author == "A. Writer"
    assert [q.id for q in dataset.queries] == ["mhr-0000", "mhr-0001"]
    assert [q.answerable for q in dataset.queries] == [True, False]
    assert dataset.articles[0].published_at.utcoffset().total_seconds() == 0  # type: ignore[union-attr]
    assert set(dataset.sha256) == {"corpus.json", "MultiHopRAG.json"}


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda c, q: c.append(_article(9, title="Title 0")), "titles are not unique"),
        (lambda c, q: c.append(_article(9, url="https://example.test/0")), "URLs are not unique"),
        (lambda c, q: q[0]["evidence_list"][0].update(title="No such"), "names no article"),
        (lambda c, q: q[0]["evidence_list"][0].update(url="https://x.test/"), "URL differs"),
        (lambda c, q: q[1]["evidence_list"].append(_evidence(c[0])), "null query carries evidence"),
        (lambda c, q: q[0]["evidence_list"].pop(), "carries 1 evidence"),
        (lambda c, q: q[0].update(question_type="mystery"), "unknown question_type"),
        (lambda c, q: c[0].update(published_at="2023-10-01T09:30:00"), "carries no timezone"),
    ],
)
def test_load_rejects_each_broken_invariant(tmp_path: Path, mutate, fragment: str) -> None:
    corpus, queries = _good()
    mutate(corpus, queries)
    manifest = _write(tmp_path, corpus, queries)
    with pytest.raises(mh.DatasetError, match=fragment):
        mh.load(tmp_path, manifest)


def test_verify_reports_size_and_digest_of_a_changed_file(tmp_path: Path) -> None:
    manifest = _write(tmp_path, *_good())
    (tmp_path / "corpus.json").write_bytes(b"[]")
    with pytest.raises(mh.DatasetError, match="does not match the manifest"):
        mh.verify(tmp_path / "corpus.json", manifest.spec("corpus"))
    with pytest.raises(mh.DatasetError, match="is missing"):
        mh.verify(tmp_path / "absent.json", manifest.spec("corpus"))


def test_load_refuses_a_file_edited_after_download(tmp_path: Path) -> None:
    manifest = _write(tmp_path, *_good())
    (tmp_path / "MultiHopRAG.json").write_text("[]")
    with pytest.raises(mh.DatasetError, match="does not match"):
        mh.load(tmp_path, manifest)


class _Opener:
    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads, self.calls = payloads, []

    def __call__(self, request):
        self.calls.append(request.full_url)
        return io.BytesIO(self.payloads[request.full_url])


def test_fetch_downloads_verifies_and_then_does_nothing(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    manifest = _write(source, *_good())
    payloads = {manifest.url(s): (source / s.path).read_bytes() for s in manifest.files}
    dest = tmp_path / "dest"
    opener = _Opener(payloads)
    out = mh.fetch(dest, manifest, opener=opener)
    assert sorted(p.name for p in out) == ["MultiHopRAG.json", "corpus.json"]
    assert len(opener.calls) == 2
    assert (
        mh.fetch(dest, manifest, opener=opener) and len(opener.calls) == 2
    )  # nothing re-downloaded
    assert all(".part" not in p.name for p in dest.iterdir())


def test_fetch_rejects_a_tampered_download_and_leaves_nothing_behind(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    manifest = _write(source, *_good())
    payloads = {manifest.url(s): (source / s.path).read_bytes() for s in manifest.files}
    corpus_url = manifest.url(manifest.spec("corpus"))
    payloads[corpus_url] = payloads[corpus_url][:-1] + b"X"
    dest = tmp_path / "dest"
    with pytest.raises(mh.DatasetError, match="does not match"):
        mh.fetch(dest, manifest, opener=_Opener(payloads))
    assert list(dest.iterdir()) == []


def test_fetch_never_overwrites_a_modified_existing_file(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    manifest = _write(source, *_good())
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "corpus.json").write_text("edited by hand")
    with pytest.raises(mh.DatasetError, match="does not match"):
        mh.fetch(dest, manifest, opener=_Opener({}))
    assert (dest / "corpus.json").read_text() == "edited by hand"


def test_the_committed_manifest_names_a_full_revision_and_both_checksums() -> None:
    manifest = mh.read_manifest()
    assert len(manifest.revision) == 40 and set(manifest.revision) <= set("0123456789abcdef")
    assert manifest.url(manifest.spec("corpus")).endswith(
        f"/resolve/{manifest.revision}/corpus.json"
    )
    assert all(len(f.sha256) == 64 for f in manifest.files)


def test_real_data_still_has_the_published_shape() -> None:
    """Skipped unless `pnkbench fetch` has run. The counts are the ones the paper's data had when
    the manifest was written (first measured 20261010; this repo re-derived them)."""
    from pnkbench import paths

    if not (paths.raw_data_dir(mh.DATASET) / "corpus.json").exists():
        pytest.skip("run `pnkbench fetch` first")
    dataset = mh.load()
    assert (len(dataset.articles), len(dataset.queries)) == (609, 2556)
    kinds = {t: sum(1 for q in dataset.queries if q.qtype == t) for t in mh.QUESTION_TYPES}
    assert kinds == {
        "comparison_query": 856,
        "inference_query": 816,
        "temporal_query": 583,
        "null_query": 301,
    }
    sizes = {
        n: sum(1 for q in dataset.queries if q.answerable and len(dataset.gold(q)) == n)
        for n in (2, 3, 4)
    }
    assert sizes == {2: 1169, 3: 774, 4: 312}


def test_load_names_a_malformed_timestamp(tmp_path: Path) -> None:
    corpus, queries = _good()
    corpus[1]["published_at"] = "yesterday"
    manifest = _write(tmp_path, corpus, queries)
    with pytest.raises(mh.DatasetError, match="not an ISO timestamp"):
        mh.load(tmp_path, manifest)


def test_real_data_yields_the_item_and_split_counts_the_dataset_readme_states() -> None:
    from collections import Counter

    from pnkbench import items, paths

    if not (paths.raw_data_dir(mh.DATASET) / "corpus.json").exists():
        pytest.skip("run `pnkbench fetch` first")
    dataset = mh.load()
    derived = items.derive_items(dataset)
    assert Counter(i.variant for i in derived) == {
        "base": 2556,
        "f-date": 2255,
        "f-cat-tag": 1544,
        "f-cat-path": 1544,
        "f-src-tag": 598,
    }
    sides = Counter((items.split_of(q.id), q.answerable) for q in dataset.queries)
    assert sides == {
        ("fit", True): 1115,
        ("fit", False): 147,
        ("eval", True): 1140,
        ("eval", False): 154,
    }
