"""Tests written against the mutants of `multihop_rag.py` that an independent review found still
surviving the suite: numbering, author and timestamp normalisation, and the clean-up of a download
that stops half way.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from pnkbench import multihop_rag as mh


def _article(i: int, **over: Any) -> dict[str, Any]:
    base = {
        "title": f"Title {i}",
        "author": "A. Writer",
        "source": "The Verge",
        "published_at": f"2023-10-{i + 1:02d}T09:30:00+00:00",
        "category": "technology",
        "url": f"https://example.test/{i}",
        "body": f"Body of article {i}.",
    }
    base.update(over)
    return base


def _evidence(article: dict[str, Any]) -> dict[str, Any]:
    return {"title": article["title"], "url": article["url"], "fact": "a fact"}


def _query(corpus: list[dict[str, Any]], first: int, second: int) -> dict[str, Any]:
    return {
        "query": f"q {first} {second}?",
        "answer": "yes",
        "question_type": "comparison_query",
        "evidence_list": [_evidence(corpus[first]), _evidence(corpus[second])],
    }


def _write(
    directory: Path, corpus: list[dict[str, Any]], queries: list[dict[str, Any]]
) -> mh.DatasetManifest:
    files = []
    for name, role, payload in (
        ("corpus.json", "corpus", corpus),
        ("MultiHopRAG.json", "queries", queries),
    ):
        data = json.dumps(payload).encode()
        (directory / name).write_bytes(data)
        files.append(mh.FileSpec(name, role, len(data), hashlib.sha256(data).hexdigest()))
    return mh.DatasetManifest("owner/repo", "a" * 40, "ODC-By", tuple(files))


def _load(
    tmp_path: Path, corpus: list[dict[str, Any]], queries: list[dict[str, Any]]
) -> mh.Dataset:
    return mh.load(tmp_path, _write(tmp_path, corpus, queries))


# --- numbering ----------------------------------------------------------------------------


def test_articles_and_queries_are_numbered_from_zero_in_file_order(tmp_path: Path) -> None:
    corpus = [_article(i) for i in range(4)]
    queries = [_query(corpus, n % 3, n % 3 + 1) for n in range(12)]
    dataset = _load(tmp_path, corpus, queries)
    assert [a.index for a in dataset.articles] == [0, 1, 2, 3]
    assert [a.title for a in dataset.articles] == ["Title 0", "Title 1", "Title 2", "Title 3"]
    assert [q.index for q in dataset.queries] == list(range(12))
    # four digits, zero padded, counted from zero: the twelfth query is mhr-0011
    assert dataset.queries[0].id == "mhr-0000" and dataset.queries[11].id == "mhr-0011"
    assert [a.index for a in dataset.gold(dataset.queries[2])] == [2, 3]


# --- authors ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  A. Writer  ", "A. Writer"),
        ("Jane Doe", "Jane Doe"),
        ("", None),
        ("   ", None),
        (None, None),
        (42, None),
    ],
)
def test_an_author_is_stripped_and_blank_or_odd_values_become_none(
    tmp_path: Path, raw: object, expected: str | None
) -> None:
    corpus = [_article(0, author=raw), _article(1)]
    dataset = _load(tmp_path, corpus, [_query(corpus, 0, 1)])
    assert dataset.articles[0].author == expected


def test_a_missing_author_key_is_none(tmp_path: Path) -> None:
    corpus = [_article(0), _article(1)]
    del corpus[0]["author"]
    dataset = _load(tmp_path, corpus, [_query(corpus, 0, 1)])
    assert dataset.articles[0].author is None and dataset.articles[1].author == "A. Writer"


# --- timestamps ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "utc"),
    [
        ("2023-10-05T01:30:00+05:00", datetime(2023, 10, 4, 20, 30, tzinfo=UTC)),
        ("2023-10-04T22:00:00-05:00", datetime(2023, 10, 5, 3, 0, tzinfo=UTC)),
        ("2023-10-01T09:30:00Z", datetime(2023, 10, 1, 9, 30, tzinfo=UTC)),
        ("2023-10-01T09:30:00+00:00", datetime(2023, 10, 1, 9, 30, tzinfo=UTC)),
    ],
)
def test_a_timestamp_with_an_offset_is_stored_as_the_same_instant_in_utc(
    tmp_path: Path, raw: str, utc: datetime
) -> None:
    corpus = [_article(0, published_at=raw), _article(1)]
    stored = _load(tmp_path, corpus, [_query(corpus, 0, 1)]).articles[0].published_at
    # an aware datetime compares equal across zones, so the fields are checked as well:
    # the calendar day and hour in UTC are what the date filters and the UTC-day windows read
    assert stored == utc
    assert stored.utcoffset() == timedelta(0)
    assert (stored.year, stored.month, stored.day, stored.hour, stored.minute) == (
        utc.year,
        utc.month,
        utc.day,
        utc.hour,
        utc.minute,
    )


def test_a_timestamp_that_is_not_a_string_is_refused(tmp_path: Path) -> None:
    corpus = [_article(0, published_at=20231001), _article(1)]
    with pytest.raises(mh.DatasetError, match=r"article 0: published_at must be a string"):
        _load(tmp_path, corpus, [_query(corpus, 0, 1)])


# --- a download that stops half way -------------------------------------------------------


class _Stops:
    """A response that raises as soon as it is read, as a Ctrl-C during a download would."""

    def __enter__(self) -> _Stops:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self, size: int) -> bytes:
        raise KeyboardInterrupt


def test_an_interrupted_download_leaves_no_partial_file_behind(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    manifest = _write(source, *_good_pair())
    dest = tmp_path / "not" / "yet" / "there"  # the folder is made, parents included
    with pytest.raises(KeyboardInterrupt):
        mh.fetch(dest, manifest, opener=lambda request: _Stops())  # type: ignore[arg-type,return-value]
    assert list(dest.iterdir()) == []


def _good_pair() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corpus = [_article(i) for i in range(3)]
    return corpus, [_query(corpus, 0, 1)]


class _Spy:
    """Serves a payload and records which `.part` files exist, and where, at the first read."""

    def __init__(self, payload: bytes, dest: Path, seen: list[dict[str, list[str]]]) -> None:
        self.stream, self.dest, self.seen, self.recorded = io.BytesIO(payload), dest, seen, False

    def __enter__(self) -> _Spy:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self, size: int) -> bytes:
        if not self.recorded:
            self.recorded = True
            self.seen.append(
                {
                    "beside_target": sorted(
                        p.name for p in self.dest.iterdir() if p.name.endswith(".part")
                    ),
                    "above_target": sorted(
                        p.name for p in self.dest.parent.iterdir() if p.name.endswith(".part")
                    ),
                }
            )
        return self.stream.read(size)


def test_the_partial_file_is_written_beside_its_target_and_nowhere_else(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    manifest = _write(source, *_good_pair())
    payloads = {manifest.url(s): (source / s.path).read_bytes() for s in manifest.files}
    dest = tmp_path / "dest"
    seen: list[dict[str, list[str]]] = []
    mh.fetch(
        dest,
        manifest,
        opener=lambda request: _Spy(payloads[request.full_url], dest, seen),  # type: ignore[arg-type,return-value]
    )
    assert len(seen) == 2
    names = ["corpus.json", "MultiHopRAG.json"]
    for name, snapshot in zip(names, seen, strict=True):
        assert len(snapshot["beside_target"]) == 1
        assert snapshot["beside_target"][0].startswith(f".{name}.")
        assert snapshot["above_target"] == []
    assert sorted(p.name for p in dest.iterdir()) == sorted(names)
