"""The steps, as functions: build, run, check reproducibility.

A run directory (`runs/<id>/`) is the unit of result: a header saying exactly what produced it, the
rows, and later the calibration, summary and report. The id carries the UTC time, the
configuration and which Pinakes ran, so two runs can never be mistaken for one another.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pnkbench import __version__, kb, layout, multihop_rag, paths, pinakes_api, runner
from pnkbench import config as config_module
from pnkbench import items as items_module
from pnkbench.items import RunItem

Log = Callable[[str], None]


def _say(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def utc_now() -> datetime:
    return datetime.now(UTC)


def git_state() -> dict[str, Any]:
    """The benchmark repository's own commit and whether the tree was clean when it ran.

    `runs/` is left out of the dirtiness check: it holds this tool's own output, including the
    directory of the run being recorded and the results of earlier runs not yet committed, and
    counting them would make every run report itself as dirty.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=paths.repo_root(), capture_output=True, text=True, check=False
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    return {
        "commit": commit or None,
        "dirty": bool(git("status", "--porcelain", "--", ".", ":(exclude)runs"))
        if commit
        else None,
    }


def pinakes_label(provenance: dict[str, Any]) -> str:
    if provenance["source"] == "pypi":
        return f"pypi-{provenance['version']}"
    if provenance["source"] == "git" and provenance["commit"]:
        return f"git-{provenance['commit'][:7]}"
    return provenance["source"]


def new_run_id(config_name: str, provenance: dict[str, Any]) -> str:
    return f"{utc_now():%Y%m%d_%H%M}-{config_name}-{pinakes_label(provenance)}"


def path_index(dataset: multihop_rag.Dataset) -> dict[str, int]:
    return {layout.doc_relpath(article): article.index for article in dataset.articles}


def select_items(dataset: multihop_rag.Dataset, *, sample: int | None) -> list[RunItem]:
    """Every item, or those of `sample` base queries picked by a fixed hash order (a smoke run)."""
    items = items_module.derive_items(dataset)
    if sample is None:
        return items
    ordered = sorted((q.id for q in dataset.queries), key=items_module.calibration_order)
    chosen = set(ordered[:sample])
    return [item for item in items if item.base_id in chosen]


def build(config_name: str, *, log: Log = _say) -> dict[str, Any]:
    dataset = multihop_rag.load()
    return kb.build_kb(dataset, config_module.load(config_name), log=log)


def run(
    config_name: str,
    *,
    workers: int,
    sample: int | None = None,
    resume: Path | None = None,
    log: Log = _say,
) -> Path:
    pinakes_api.require_interfaces()
    dataset = multihop_rag.load()
    config = config_module.load(config_name)
    kb_root = paths.kb_dir(config.name)
    if not (kb_root / "pinakes.toml").exists():
        raise RuntimeError(f"no KB at {kb_root}; run `pnkbench build --config {config.name}`")

    provenance = pinakes_api.provenance()
    previous: dict[str, Any] | None = None
    if resume is not None:
        run_dir = resume
        previous = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if previous["pinakes"]["code_sha256"] != provenance["code_sha256"]:
            raise RuntimeError(
                "cannot resume: the installed Pinakes differs from the one that started this run"
            )
        if previous["config"]["name"] != config.name:
            raise RuntimeError("cannot resume: this run used another configuration")
    else:
        run_dir = paths.runs_dir() / new_run_id(config.name, provenance)
        run_dir.mkdir(parents=True, exist_ok=False)

    items = select_items(dataset, sample=sample)
    index = path_index(dataset)

    handle = pinakes_api.open_kb(kb_root)
    try:
        settings = pinakes_api.settings_record(handle)
        limit = int(handle.manifest.retrieval.fusion_top_k)
        devices = {
            "embedding": str(getattr(getattr(handle.backend, "_model", None), "device", None)),
            "rerank": str(getattr(getattr(handle.reranker, "_model", None), "device", None)),
        }
    finally:
        handle.close()

    if previous is not None:
        changed = [
            name
            for name, was, now in (
                ("settings (models, chunking, retrieval)", previous["settings"], settings),
                ("dataset files", previous["dataset"]["sha256"], dict(dataset.sha256)),
                ("sample", previous["execution"]["sample_base_queries"], sample),
            )
            if was != now
        ]
        if changed:
            raise RuntimeError(
                f"cannot resume: {', '.join(changed)} differ from the run being continued"
            )

    import torch

    header: dict[str, Any] = {
        "schema": 1,
        "run_id": run_dir.name,
        "status": "running",
        "started_utc": utc_now().isoformat(timespec="seconds"),
        "config": {
            "name": config.name,
            "description": config.description,
            "backend": config.backend,
            "chunking_metadata": config.chunking_metadata,
            "chunking_max_tokens": config.chunking_max_tokens,
        },
        "pinakes": provenance,
        "settings": settings,
        "dataset": {
            "source_repo": multihop_rag.read_manifest().source_repo,
            "revision": multihop_rag.read_manifest().revision,
            "sha256": dict(dataset.sha256),
            "articles": len(dataset.articles),
            "queries": len(dataset.queries),
        },
        "benchmark": {"version": __version__, **git_state()},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
            "devices": devices,
        },
        "execution": {
            "workers": workers,
            "passages_per_search": limit,
            "sample_base_queries": sample,
        },
        "items": {
            "total": len(items),
            "by_variant": {
                v: sum(1 for i in items if i.variant == v)
                for v in (items_module.BASE, *items_module.FILTER_VARIANTS)
            },
        },
    }
    if previous is not None:
        # The header is rewritten on a resume, so what the earlier sessions recorded is carried
        # over: the original start, each resume, and the wall time they spent.
        earlier = previous["execution"]
        header["started_utc"] = previous["started_utc"]
        header["execution"]["resumed_utc"] = [
            *earlier.get("resumed_utc", []),
            utc_now().isoformat(timespec="seconds"),
        ]
        header["execution"]["earlier_wall_seconds"] = round(
            earlier.get("earlier_wall_seconds", 0) + earlier.get("wall_seconds", 0), 1
        )
    _write_json(run_dir / "run.json", header)

    started = time.monotonic()
    stats = runner.run_items(
        kb_root,
        items,
        index,
        run_dir / "rows.jsonl",
        workers=workers,
        threads=None,
        limit=limit,
        log=log,
    )
    runner.finalize(run_dir / "rows.jsonl", items)
    header["execution"].update(stats)
    header["execution"]["wall_seconds"] = round(time.monotonic() - started, 1)

    header["reproducibility"] = reproducibility(
        kb_root, items, index, run_dir / "rows.jsonl", limit=limit, log=log
    )
    header["status"] = "complete"
    header["finished_utc"] = utc_now().isoformat(timespec="seconds")
    _write_json(run_dir / "run.json", header)
    return run_dir


def reproducibility(
    kb_root: Path,
    items: Sequence[RunItem],
    index: dict[str, int],
    rows_path: Path,
    *,
    limit: int,
    sample: int = 60,
    log: Log = _say,
) -> dict[str, Any]:
    """Search a spread of items again, in one process, and compare with what the run recorded.

    The run used several worker processes sharing one GPU; this asks whether a single process
    sees the same ranking and the same scores. A difference is reported, not hidden: it bounds
    how much of any later per-question comparison is noise.
    """
    recorded = {row["id"]: row for row in runner.read_rows(rows_path)}
    chosen = [items[i] for i in range(0, len(items), max(1, len(items) // sample))][:sample]
    handle = pinakes_api.open_kb(kb_root)
    order_mismatches = 0
    worst = 0.0
    try:
        for item in chosen:
            again = runner.run_one(handle, index, runner.to_payload(item), limit)
            before = recorded[item.id]["p"]
            after = again["p"]
            if [p[0] for p in before] != [p[0] for p in after]:
                order_mismatches += 1
            for (_, a), (_, b) in zip(before, after, strict=False):
                if a is not None and b is not None:
                    worst = max(worst, abs(a - b))
    finally:
        handle.close()
    result = {
        "checked": len(chosen),
        "ranking_mismatches": order_mismatches,
        "max_abs_score_difference": worst,
    }
    log(f"reproducibility: {result}")
    return result


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    temporary.replace(path)


# --- steps that read a finished run ---------------------------------------------------------


def _context(
    run_dir: Path,
) -> tuple[dict[str, Any], multihop_rag.Dataset, list[RunItem], dict[str, int]]:
    header = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if header.get("status") != "complete":
        raise RuntimeError(
            f"{run_dir.name} is not a complete run (status {header.get('status')!r})"
        )
    dataset = multihop_rag.load()
    items = select_items(dataset, sample=header["execution"]["sample_base_queries"])
    return header, dataset, items, path_index(dataset)


def calibrate_step(run_dir: Path, *, log: Log = _say) -> dict[str, Any]:
    from pnkbench import calibration

    header, dataset, items, index = _context(run_dir)
    rows = runner.read_rows(run_dir / "rows.jsonl")
    log("pinakes.calibrate on the fit half")
    result = calibration.calibrate(
        paths.kb_dir(header["config"]["name"]),
        run_dir,
        items,
        dataset,
        rows,
        index,
        limit=header["execution"]["passages_per_search"],
    )
    _write_json(run_dir / "calibration.json", result)
    return result


def score_step(run_dir: Path) -> dict[str, Any]:
    from pnkbench import scoring

    header, dataset, items, _ = _context(run_dir)
    calibration_file = run_dir / "calibration.json"
    thresholds = (
        json.loads(calibration_file.read_text(encoding="utf-8"))["thresholds"]
        if calibration_file.exists()
        else None
    )
    summary = scoring.score_run(
        runner.read_rows(run_dir / "rows.jsonl"),
        items,
        dataset,
        relpath_of=layout.doc_relpath,
        tags_of=layout.tags_for,
        thresholds=thresholds,
        fusion_top_k=header["settings"]["retrieval"]["fusion_top_k"],
        passages_per_search=header["execution"]["passages_per_search"],
    )
    _write_json(run_dir / "summary.json", summary)
    return summary


def harness_step(run_dir: Path) -> dict[str, Any]:
    from pnkbench import harness

    header, dataset, items, _ = _context(run_dir)
    result = harness.run_harness(paths.kb_dir(header["config"]["name"]), run_dir, dataset, items)
    _write_json(run_dir / "harness.json", result)
    return result


def report_step(run_dir: Path) -> Path:
    from pnkbench import report

    target = run_dir / "report.md"
    target.write_text(report.render_report(run_dir), encoding="utf-8")
    return target


def compare_step(dir_a: Path, dir_b: Path) -> Path:
    from pnkbench import compare, report

    result = compare.compare_runs(dir_a, dir_b, multihop_rag.load())
    target = paths.runs_dir() / f"compare--{dir_a.name}--{dir_b.name}.md"
    target.write_text(report.render_compare(result), encoding="utf-8")
    _write_json(target.with_suffix(".json"), result)
    return target


def run_all(config_name: str, *, workers: int, sample: int | None = None, log: Log = _say) -> Path:
    log(f"== build {config_name}")
    build(config_name, log=log)
    log(f"== run {config_name}")
    run_dir = run(config_name, workers=workers, sample=sample, log=log)
    log("== calibrate")
    calibrate_step(run_dir, log=log)
    log("== score")
    score_step(run_dir)
    log("== harness (Pinakes' own evaluator, eval half)")
    harness_step(run_dir)
    log("== report")
    report_step(run_dir)
    return run_dir
