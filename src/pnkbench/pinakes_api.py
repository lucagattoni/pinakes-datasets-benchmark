"""The only module that imports Pinakes.

Everything the benchmark calls inside Pinakes is listed here, and `require_interfaces` checks the
whole list against the installed build before any work starts. That is what makes the
`release`/`main` switch safe: when `main` moves an internal signature, the run stops with a message
naming it, instead of failing on row 1,400 or, worse, measuring something else.

`provenance` records which Pinakes ran. **The version string is not enough**: a release and the
commit after it both say `0.32.8`, so the record also carries a hash of the installed code.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pnkbench.items import FilterSpec


class InterfaceError(RuntimeError):
    """The installed Pinakes does not offer something this benchmark calls."""


_REQUIRED_NAMES: dict[str, tuple[str, ...]] = {
    "pinakes.search": ("search", "Filters", "confidence_of", "resolve_tier"),
    "pinakes.store": ("connect_ro", "get_meta", "chunking_identity", "chunking_drift"),
    "pinakes.manifest": ("load",),
    "pinakes.embed": ("load_backend", "load_reranker"),
    "pinakes.sidecar": ("create", "skeleton", "sidecar_path", "Sidecar"),
}
_SEARCH_PARAMETERS = {"connection", "manifest", "query", "backend", "reranker", "filters", "limit"}
_FILTER_FIELDS = {"tags", "path_prefix", "source_type", "modified_after", "modified_before"}
_SIDECAR_FIELDS = {"id", "title", "tags", "provenance"}


def require_interfaces() -> None:
    problems: list[str] = []
    for module_name, names in _REQUIRED_NAMES.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            problems.append(f"cannot import {module_name}: {exc}")
            continue
        problems.extend(
            f"{module_name} has no {name}" for name in names if not hasattr(module, name)
        )
    if not problems:
        from pinakes.search import Filters, search
        from pinakes.sidecar import Sidecar

        missing = _SEARCH_PARAMETERS - set(inspect.signature(search).parameters)
        if missing:
            problems.append(f"pinakes.search.search lost parameters {sorted(missing)}")
        missing = _FILTER_FIELDS - {f.name for f in dataclasses.fields(Filters)}
        if missing:
            problems.append(f"pinakes.search.Filters lost fields {sorted(missing)}")
        missing = _SIDECAR_FIELDS - {f.name for f in dataclasses.fields(Sidecar)}
        if missing:
            problems.append(f"pinakes.sidecar.Sidecar lost fields {sorted(missing)}")
    if problems:
        raise InterfaceError(
            "the installed Pinakes does not match what the benchmark calls:\n  - "
            + "\n  - ".join(problems)
            + "\nUpdate src/pnkbench/pinakes_api.py, the one place that depends on Pinakes."
        )


def _package_dir() -> Path:
    spec = importlib.util.find_spec("pinakes")
    if spec is None or not spec.submodule_search_locations:
        raise InterfaceError("pinakes is not installed; sync with `--extra release` or `main`")
    return Path(next(iter(spec.submodule_search_locations)))


def code_fingerprint() -> str:
    """SHA-256 over every installed `.py` file: path and bytes, in a fixed order."""
    root = _package_dir()
    digest = hashlib.sha256()
    for file in sorted(root.rglob("*.py")):
        digest.update(file.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(file.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def provenance() -> dict[str, Any]:
    """Which Pinakes this run used: version, where it came from, and a hash of its code."""
    dist = importlib.metadata.distribution("pinakes")
    source = "pypi"
    commit: str | None = None
    url: str | None = None
    direct = dist.read_text("direct_url.json")
    if direct:
        info: dict[str, Any] = json.loads(direct)
        url = info.get("url")
        if "vcs_info" in info:
            source = "git"
            commit = info["vcs_info"].get("commit_id")
        elif info.get("dir_info", {}).get("editable"):
            source = "editable"
        else:
            source = "url"
    return {
        "version": dist.version,
        "source": source,
        "commit": commit,
        "url": url,
        "code_sha256": code_fingerprint(),
    }


def index_path(root: Path) -> Path:
    """Where the manifest says the index lives, rather than where this module believes it does."""
    from pinakes import manifest as manifest_module

    return Path(manifest_module.load(root).index_path)


def pnk_executable() -> Path:
    candidate = Path(sys.executable).parent / "pnk"
    if not candidate.exists():
        raise InterfaceError(f"no `pnk` next to {sys.executable}; is Pinakes installed here?")
    return candidate


def run_pnk(args: Sequence[str], *, log: Path | None = None) -> str:
    """Run `pnk <args>`; return its output; raise with the tail of it on a non-zero exit."""
    completed = subprocess.run(
        [str(pnk_executable()), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(output, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(output.strip().splitlines()[-25:])
        raise RuntimeError(f"`pnk {' '.join(args)}` exited {completed.returncode}:\n{tail}")
    return output


def mint_sidecar(document: Path, *, title: str, tags: Sequence[str], source_url: str) -> None:
    """Write a document's sidecar before the first sync, so its title is the publisher's.

    The same move `build_rfc_corpus.py` makes in Pinakes: `title` is the user's field, sync never
    overwrites it, and a title missing at first ingest is the filename for good.
    """
    from pinakes.sidecar import create, sidecar_path, skeleton

    sidecar = dataclasses.replace(
        skeleton(document, title=title),
        tags=tuple(tags),
        provenance={"source": source_url},
    )
    create(sidecar_path(document), sidecar)


@dataclass(slots=True)
class Kb:
    root: Path
    manifest: Any
    backend: Any
    reranker: Any
    connection: Any

    def close(self) -> None:
        self.connection.close()


def open_kb(root: Path, *, offline: bool = True) -> Kb:
    """Open a built KB read-only, loading the models its manifest names.

    Refuses a manifest whose chunking differs from what built the index, as Pinakes' own eval
    does: otherwise a run would be labelled with settings the index was not built under.
    """
    from pinakes import manifest as manifest_module
    from pinakes import store
    from pinakes.embed import load_backend, load_reranker

    manifest = manifest_module.load(root)
    backend = load_backend(manifest.embedding, offline=offline)
    reranker = (
        load_reranker(manifest.rerank, offline=offline)
        if manifest.retrieval.rerank == "local"
        else None
    )
    connection = store.connect_ro(manifest.index_path)
    drift = store.chunking_drift(
        store.get_meta(connection),
        store.chunking_identity(
            headings=manifest.chunking.headings,
            max_tokens=manifest.chunking.max_tokens,
            overlap=manifest.chunking.overlap,
            metadata=manifest.chunking.metadata,
        ),
    )
    if drift:
        connection.close()
        moved = ", ".join(f"{key} {was} -> {now}" for key, (was, now) in sorted(drift.items()))
        raise RuntimeError(
            f"{root}: the index was built under different chunking than the manifest ({moved}). "
            "Run `pnkbench build` again."
        )
    return Kb(root, manifest, backend, reranker, connection)


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one search returned: the passages in rank order, as (path, rerank score)."""

    passages: tuple[tuple[str, float | None], ...]
    confidence: str
    considered: int


def search_kb(kb: Kb, query: str, spec: FilterSpec, *, limit: int) -> Outcome:
    from pinakes.search import Filters, search

    result = search(
        kb.connection,
        kb.manifest,
        query,
        backend=kb.backend,
        reranker=kb.reranker,
        filters=Filters(
            tags=spec.tags,
            path_prefix=spec.path_prefix,
            modified_after=spec.modified_after,
            modified_before=spec.modified_before,
        ),
        limit=limit,
    )
    return Outcome(
        passages=tuple((p.path, p.rerank_score) for p in result.passages),
        confidence=result.confidence,
        considered=result.considered,
    )


def settings_record(kb: Kb) -> dict[str, Any]:
    """The settings a run used, as Pinakes' own eval header records them, plus what resolved."""
    from pinakes.search import resolve_tier

    retrieval = kb.manifest.retrieval
    chunking = kb.manifest.chunking
    thresholds = retrieval.confidence
    embedding = kb.backend.info()
    reranker = kb.reranker.info() if kb.reranker is not None else None
    return {
        "embedding": dataclasses.asdict(embedding),
        "rerank": dataclasses.asdict(reranker) if reranker is not None else None,
        "chunking": {
            "headings": chunking.headings,
            "max_tokens": chunking.max_tokens,
            "overlap": chunking.overlap,
            "metadata": chunking.metadata,
        },
        "retrieval": {
            "candidates_per_source": retrieval.candidates_per_source,
            "fusion": retrieval.fusion,
            "fusion_top_k": retrieval.fusion_top_k,
            "final_k": retrieval.final_k,
            "rerank": retrieval.rerank,
            "vector_tier": retrieval.vector_tier,
            "vector_tier_resolved": resolve_tier(kb.manifest),
        },
        "confidence_thresholds": (
            None
            if thresholds is None
            else {
                "fitted_for": thresholds.fitted_for,
                "low_below": thresholds.low_below,
                "high_above": thresholds.high_above,
            }
        ),
    }
