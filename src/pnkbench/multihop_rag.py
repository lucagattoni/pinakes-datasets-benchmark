"""MultiHop-RAG: fetch it, verify it byte for byte, and load it with its invariants checked.

Source: Tang and Yang, 2024 (https://arxiv.org/abs/2401.15391). A news knowledge base and queries
whose evidence is spread over two to four articles; `null` queries are the ones the articles cannot
answer, and the right behaviour there is to decline.

**Nothing is trusted that was not checked.** The manifest (`datasets/multihop-rag/manifest.json`) pins
a Hub revision and the size and SHA-256 of both files. `fetch` compares what arrives with it and
refuses a mismatch; `load` re-checks the files on disk before parsing, so a file edited after
download cannot slip through. It then asserts the structural facts every later stage leans on:
article titles are unique, every evidence item names exactly one article by title and URL alike, a
`null` query has no evidence and every other query has at least two items.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, cast

from pnkbench import paths

DATASET = "multihop-rag"
NULL = "null_query"
QUESTION_TYPES = ("comparison_query", "inference_query", "temporal_query", NULL)
_CHUNK = 1 << 20

Opener = Callable[[urllib.request.Request], IO[bytes]]


class DatasetError(Exception):
    """The dataset on disk is not the dataset the manifest describes."""


@dataclass(frozen=True, slots=True)
class FileSpec:
    path: str
    role: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    source_repo: str
    revision: str
    licence: str
    files: tuple[FileSpec, ...]

    def spec(self, role: str) -> FileSpec:
        for spec in self.files:
            if spec.role == role:
                return spec
        raise DatasetError(f"the manifest names no file with role {role!r}")

    def url(self, spec: FileSpec) -> str:
        return f"https://huggingface.co/datasets/{self.source_repo}/resolve/{self.revision}/{spec.path}"


@dataclass(frozen=True, slots=True)
class Article:
    index: int
    title: str
    author: str | None
    source: str
    category: str
    url: str
    published_at: datetime
    body: str


@dataclass(frozen=True, slots=True)
class Evidence:
    title: str
    url: str
    fact: str


@dataclass(frozen=True, slots=True)
class Query:
    id: str
    index: int
    text: str
    answer: str
    qtype: str
    evidence: tuple[Evidence, ...]

    @property
    def answerable(self) -> bool:
        return self.qtype != NULL


@dataclass(frozen=True, slots=True)
class Dataset:
    articles: tuple[Article, ...]
    queries: tuple[Query, ...]
    sha256: Mapping[str, str]

    def article_by_title(self) -> dict[str, Article]:
        return {article.title: article for article in self.articles}

    def gold(self, query: Query) -> tuple[Article, ...]:
        """The distinct articles a query's evidence names, in the order the evidence lists them.

        168 queries list one article twice, so this is the set a retriever must find, not the
        evidence count.
        """
        by_title = self.article_by_title()
        seen: dict[str, Article] = {}
        for item in query.evidence:
            seen.setdefault(item.title, by_title[item.title])
        return tuple(seen.values())


def read_manifest(path: Path | None = None) -> DatasetManifest:
    target = path or paths.dataset_dir(DATASET) / "manifest.json"
    raw = cast(dict[str, Any], json.loads(target.read_text(encoding="utf-8")))
    return DatasetManifest(
        source_repo=str(raw["source_repo"]),
        revision=str(raw["revision"]),
        licence=str(raw["licence"]),
        files=tuple(
            FileSpec(
                path=str(item["path"]),
                role=str(item["role"]),
                bytes=int(item["bytes"]),
                sha256=str(item["sha256"]),
            )
            for item in raw["files"]
        ),
    )


def _digest(path: Path) -> tuple[int, str]:
    sha = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(_CHUNK):
            sha.update(block)
            size += len(block)
    return size, sha.hexdigest()


def verify(path: Path, spec: FileSpec) -> None:
    """Raise unless `path` is exactly the file the manifest describes."""
    if not path.is_file():
        raise DatasetError(f"{path} is missing. Run `pnkbench fetch`.")
    size, digest = _digest(path)
    if size != spec.bytes or digest != spec.sha256:
        raise DatasetError(
            f"{path} does not match the manifest.\n"
            f"  expected {spec.bytes} bytes, sha256 {spec.sha256}\n"
            f"  found    {size} bytes, sha256 {digest}\n"
            "Delete the file and fetch again; if the source really changed, the manifest is "
            "what has to be reviewed, not this check."
        )


def _urlopen(request: urllib.request.Request) -> IO[bytes]:
    return cast(IO[bytes], urllib.request.urlopen(request, timeout=60))


def fetch(
    dest: Path | None = None,
    manifest: DatasetManifest | None = None,
    *,
    opener: Opener = _urlopen,
) -> list[Path]:
    """Download whatever is missing, verify everything, and return the verified paths.

    A file already on disk is verified rather than replaced: a mismatch is an error, never a
    silent re-download over something somebody may have edited.
    """
    manifest = manifest or read_manifest()
    dest = dest or paths.raw_data_dir(DATASET)
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for spec in manifest.files:
        target = dest / spec.path
        if target.exists():
            verify(target, spec)
            out.append(target)
            continue
        request = urllib.request.Request(manifest.url(spec), headers={"User-Agent": "pnkbench/0.1"})
        handle, temp_name = tempfile.mkstemp(dir=dest, prefix=f".{spec.path}.", suffix=".part")
        temp = Path(temp_name)
        try:
            with os.fdopen(handle, "wb") as sink, opener(request) as response:
                while block := response.read(_CHUNK):
                    sink.write(block)
            verify(temp, spec)
            temp.replace(target)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        out.append(target)
    return out


def _author(raw: object) -> str | None:
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


def _timestamp(raw: object, where: str) -> datetime:
    if not isinstance(raw, str):
        raise DatasetError(f"{where}: published_at must be a string, not {raw!r}")
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise DatasetError(f"{where}: published_at {raw!r} is not an ISO timestamp") from exc
    if parsed.tzinfo is None:
        raise DatasetError(f"{where}: published_at {raw!r} carries no timezone")
    return parsed.astimezone(UTC)


def load(dest: Path | None = None, manifest: DatasetManifest | None = None) -> Dataset:
    manifest = manifest or read_manifest()
    dest = dest or paths.raw_data_dir(DATASET)
    digests: dict[str, str] = {}
    for spec in manifest.files:
        verify(dest / spec.path, spec)
        digests[spec.path] = spec.sha256

    corpus_raw = cast(
        list[dict[str, Any]],
        json.loads((dest / manifest.spec("corpus").path).read_text(encoding="utf-8")),
    )
    queries_raw = cast(
        list[dict[str, Any]],
        json.loads((dest / manifest.spec("queries").path).read_text(encoding="utf-8")),
    )

    articles: list[Article] = []
    for index, item in enumerate(corpus_raw):
        articles.append(
            Article(
                index=index,
                title=str(item["title"]),
                author=_author(item.get("author")),
                source=str(item["source"]),
                category=str(item["category"]),
                url=str(item["url"]),
                published_at=_timestamp(item["published_at"], f"article {index}"),
                body=str(item["body"]),
            )
        )
    by_title = {article.title: article for article in articles}
    if len(by_title) != len(articles):
        raise DatasetError("article titles are not unique; evidence is matched by title")
    if len({article.url for article in articles}) != len(articles):
        raise DatasetError("article URLs are not unique")

    queries: list[Query] = []
    for index, item in enumerate(queries_raw):
        where = f"query {index}"
        qtype = str(item["question_type"])
        if qtype not in QUESTION_TYPES:
            raise DatasetError(f"{where}: unknown question_type {qtype!r}")
        evidence = tuple(
            Evidence(title=str(e["title"]), url=str(e["url"]), fact=str(e["fact"]))
            for e in item["evidence_list"]
        )
        for e in evidence:
            article = by_title.get(e.title)
            if article is None:
                raise DatasetError(f"{where}: evidence names no article titled {e.title!r}")
            if article.url != e.url:
                raise DatasetError(f"{where}: evidence URL differs from its article's: {e.url}")
        if qtype == NULL and evidence:
            raise DatasetError(f"{where}: a null query carries evidence")
        if qtype != NULL and len(evidence) < 2:
            raise DatasetError(f"{where}: an answerable query carries {len(evidence)} evidence")
        queries.append(
            Query(
                id=f"mhr-{index:04d}",
                index=index,
                text=str(item["query"]),
                answer=str(item["answer"]),
                qtype=qtype,
                evidence=evidence,
            )
        )
    return Dataset(articles=tuple(articles), queries=tuple(queries), sha256=digests)
