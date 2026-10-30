"""Build a Pinakes knowledge base from the dataset, and prove the build is what was intended.

One `.txt` file per article under `docs/<category>/<source>/`, the article body byte for byte, with
`mtime` set to the publication time (so Pinakes' date filters mean something) and a sidecar minted
**before** the first sync carrying the title, the tags and the URL. The build is always from
scratch: a KB is generated, lives under `build/`, and is safe to delete.

`verify_kb` runs after the sync and is the reason a number from this benchmark can be trusted to
describe the dataset: it checks the index holds exactly the articles, at the expected paths, with
the expected titles, tags and times, and no failure rows.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import time
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pnkbench import layout, paths, pinakes_api
from pnkbench.config import Config, Models, load_models
from pnkbench.multihop_rag import Dataset

_TABLE = re.compile(r"^\s*\[")


class BuildError(Exception):
    pass


def set_key(manifest_text: str, table: str, key: str, literal: str) -> str:
    """Insert (or replace) `key = literal` in `[table]`, leaving every other line as it was.

    `literal` is already TOML (`'"prefix"'`, `'0.5'`). The result is parsed back and the key
    compared, so an edit that did not take is an error here, not a surprise in the first search.
    """
    lines = manifest_text.splitlines(keepends=True)
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == f"[{table}]")
    except StopIteration:
        raise BuildError(f"the stamped manifest has no [{table}] table") from None
    end = start + 1
    while end < len(lines) and not _TABLE.match(lines[end]):
        end += 1
    existing = re.compile(rf"^\s*{re.escape(key)}\s*=")
    body = [line for line in lines[start + 1 : end] if not existing.match(line)]
    last_key = max(
        (i for i, line in enumerate(body) if line.strip() and not line.lstrip().startswith("#")),
        default=-1,
    )
    body.insert(last_key + 1, f"{key} = {literal}\n")
    result = "".join([*lines[: start + 1], *body, *lines[end:]])
    if tomllib.loads(result)[table][key] != tomllib.loads(f"v = {literal}")["v"]:
        raise BuildError(f"the edit of [{table}] {key} did not take")
    return result


def set_chunking_metadata(manifest_text: str, value: str) -> str:
    return set_key(manifest_text, "chunking", "metadata", json.dumps(value))


def pin_models(manifest_text: str, models: Models) -> str:
    """Pin both model revisions, after checking the stamped model ids are the ones pinned.

    A different default model id in a newer Pinakes would make the pinned sha meaningless, so that
    is a stop, not a silent pass.
    """
    stamped = tomllib.loads(manifest_text)
    for table, pin in (("embedding", models.embedding), ("rerank", models.rerank)):
        if stamped[table]["model"] != pin.model:
            raise BuildError(
                f"Pinakes stamped [{table}] model {stamped[table]['model']!r}; configs/models.toml "
                f"pins {pin.model!r}. Review the pin before measuring anything."
            )
        manifest_text = set_key(manifest_text, table, "revision", json.dumps(pin.revision))
    return manifest_text


_CONFIDENCE_TABLE = re.compile(r"^\[retrieval\.confidence\]\s*$")


def set_confidence_thresholds(
    manifest_text: str, *, fitted_for: str, low_below: float, high_above: float
) -> str:
    """Replace any `[retrieval.confidence]` table with the fitted thresholds, appended at the end."""
    lines = manifest_text.splitlines(keepends=True)
    kept: list[str] = []
    skipping = False
    for line in lines:
        if _CONFIDENCE_TABLE.match(line):
            skipping = True
            continue
        if skipping and _TABLE.match(line):
            skipping = False
        if not skipping:
            kept.append(line)
    block = (
        "\n[retrieval.confidence]\n"
        f'fitted_for = "{fitted_for}"\n'
        f"low_below  = {low_below!r}\n"
        f"high_above = {high_above!r}\n"
    )
    result = "".join(kept).rstrip("\n") + "\n" + block
    parsed = tomllib.loads(result)["retrieval"]["confidence"]
    if (parsed["fitted_for"], parsed["low_below"], parsed["high_above"]) != (
        fitted_for,
        low_below,
        high_above,
    ):
        raise BuildError("the threshold edit did not take")
    return result


def _under(parent: Path, child: Path) -> bool:
    return child.resolve().is_relative_to(parent.resolve())


def build_kb(
    dataset: Dataset,
    config: Config,
    *,
    root: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    pinakes_api.require_interfaces()
    root = root or paths.kb_dir(config.name)
    if not _under(paths.build_dir(), root):
        raise BuildError(f"refusing to build outside {paths.build_dir()}: {root}")
    if root.exists():
        log(f"removing the previous build at {root}")
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)

    log(f"pnk init ({config.backend})")
    pinakes_api.run_pnk(
        ["init", str(root), "--backend", config.backend, "--name", f"mhr-{config.name}"]
    )

    relpaths = [layout.doc_relpath(article) for article in dataset.articles]
    if len(set(relpaths)) != len(relpaths):
        raise BuildError("two articles map to one path")
    started = time.monotonic()
    for article, relpath in zip(dataset.articles, relpaths, strict=True):
        document = root / relpath
        document.parent.mkdir(parents=True, exist_ok=True)
        document.write_text(article.body, encoding="utf-8", newline="")
        stamp = article.published_at.timestamp()
        os.utime(document, (stamp, stamp))
        pinakes_api.mint_sidecar(
            document,
            title=article.title,
            tags=layout.tags_for(article),
            source_url=article.url,
        )
    log(f"wrote {len(relpaths)} documents and sidecars in {time.monotonic() - started:.1f}s")

    manifest_file = root / "pinakes.toml"
    text = pin_models(manifest_file.read_text(encoding="utf-8"), load_models())
    if config.chunking_max_tokens is not None:
        text = set_key(text, "chunking", "max_tokens", str(config.chunking_max_tokens))
    if config.chunking_metadata != "off":
        text = set_chunking_metadata(text, config.chunking_metadata)
    manifest_file.write_text(text, encoding="utf-8")

    log("pnk sync (builds the index: embeds every chunk)")
    started = time.monotonic()
    pinakes_api.run_pnk(
        ["sync", "--kb", str(root), "--offline"], log=root.parent / f"{config.name}.sync.log"
    )
    sync_seconds = time.monotonic() - started
    log(f"sync took {sync_seconds:.0f}s")

    report = verify_kb(dataset, root)
    report.update(
        {
            "config": config.name,
            "root": str(root),
            "sync_seconds": round(sync_seconds, 1),
            "manifest_sha256": _sha256_text(manifest_file),
        }
    )
    (root.parent / f"{config.name}.build.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _sha256_text(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_kb(dataset: Dataset, root: Path) -> dict[str, Any]:
    """Raise `BuildError` unless the index holds exactly this dataset, as laid out."""
    index = pinakes_api.index_path(root)
    connection = sqlite3.connect(f"file:{index}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT path, title, mtime, metadata, state, source_type FROM documents"
        ).fetchall()
        chunks = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        failures = connection.execute("SELECT count(*) FROM failures").fetchone()[0]
    finally:
        connection.close()

    problems: list[str] = []
    if failures:
        problems.append(f"{failures} rows in the failures table")
    by_path = {row[0]: row for row in rows}
    expected = {layout.doc_relpath(a): a for a in dataset.articles}
    if set(by_path) != set(expected):
        missing = sorted(set(expected) - set(by_path))[:3]
        extra = sorted(set(by_path) - set(expected))[:3]
        problems.append(f"document set differs (missing {missing}, unexpected {extra})")
    for path, article in expected.items():
        row = by_path.get(path)
        if row is None:
            continue
        _, title, mtime, metadata, state, source_type = row
        tags = tuple(json.loads(metadata).get("tags", ()))
        if state != "active":
            problems.append(f"{path}: state {state}")
        if source_type != "text":
            problems.append(f"{path}: source_type {source_type}")
        if title != article.title:
            problems.append(f"{path}: title {title!r}")
        if tags != layout.tags_for(article):
            problems.append(f"{path}: tags {tags}")
        if abs(mtime - article.published_at.timestamp()) > 1:
            problems.append(f"{path}: mtime {mtime} != {article.published_at.timestamp()}")
        if len(problems) > 10:
            break
    if problems:
        raise BuildError("the built KB is not the dataset:\n  - " + "\n  - ".join(problems[:10]))
    return {"documents": len(rows), "chunks": chunks}
