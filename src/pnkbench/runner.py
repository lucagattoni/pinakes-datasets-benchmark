"""Run every item through Pinakes and record, per item, what came back.

A row is deliberately small and self-sufficient for scoring: the item id, the confidence label
Pinakes reported, and the passages in rank order as `[article index, rerank score]`. No query text
and no passage text, so a committed rows file is a measurement, not a second copy of the corpus.

**Resumable.** Rows are appended and flushed batch by batch; a restart reads what is already there
and runs only what is missing, so a crash costs one batch, not the run. `finalize` then checks the
file against the item list: every id present exactly once, nothing unknown.

**Parallel, with the cost named.** Each worker process loads its own copy of both models. That is
the price of processes over threads (the models are not shared), and it is only worth paying if
the measured throughput rises; `tune` exists to measure that instead of guessing.
"""

from __future__ import annotations

import json
import multiprocessing
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from pnkbench import pinakes_api
from pnkbench.items import FilterSpec, RunItem

BATCH = 8
STALL_SECONDS = 300
"""How long the pool may go without finishing a batch before the run is declared stalled.

A batch of eight takes seconds and the first one adds model loading (about half a minute), so five
minutes is generous. It exists because `multiprocessing.Pool` replaces a worker that dies while
starting up, silently and forever: without a deadline, a broken worker environment looks like a
run that is merely slow.
"""

_KB: pinakes_api.Kb | None = None
_INDEX: dict[str, int] = {}
_LIMIT = 20


def _init_worker(kb_root: str, path_index: dict[str, int], limit: int, threads: int | None) -> None:
    global _KB, _INDEX, _LIMIT
    if threads is not None:
        import torch

        torch.set_num_threads(threads)
    _KB = pinakes_api.open_kb(Path(kb_root))
    _INDEX = path_index
    _LIMIT = limit


def _spec(raw: dict[str, Any]) -> FilterSpec:
    return FilterSpec(
        tags=tuple(raw["tags"]),
        path_prefix=raw["path_prefix"],
        modified_after=raw["after"],
        modified_before=raw["before"],
    )


def run_one(
    kb: pinakes_api.Kb, path_index: dict[str, int], item: dict[str, Any], limit: int
) -> dict[str, Any]:
    started = time.perf_counter()
    outcome = pinakes_api.search_kb(kb, item["query"], _spec(item), limit=limit)
    elapsed = time.perf_counter() - started
    passages: list[list[Any]] = []
    for path, score in outcome.passages:
        if path not in path_index:
            raise RuntimeError(f"{item['id']}: Pinakes returned {path!r}, which is no article")
        passages.append([path_index[path], score])
    return {
        "id": item["id"],
        "base_id": item["base_id"],
        "variant": item["variant"],
        "conf": outcome.confidence,
        "considered": outcome.considered,
        "p": passages,
        "ms": round(elapsed * 1000),
    }


def _work(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assert _KB is not None
    return [run_one(_KB, _INDEX, item, _LIMIT) for item in batch]


def to_payload(item: RunItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "base_id": item.base_id,
        "variant": item.variant,
        "query": item.query,
        "tags": list(item.filters.tags),
        "path_prefix": item.filters.path_prefix,
        "after": item.filters.modified_after,
        "before": item.filters.modified_before,
    }


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Every complete row in `path`; a trailing partial line (a killed write) is ignored."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def run_items(
    kb_root: Path,
    items: Sequence[RunItem],
    path_index: dict[str, int],
    out: Path,
    *,
    workers: int,
    threads: int | None,
    limit: int,
    log: Callable[[str], None] = lambda message: print(message, file=sys.stderr, flush=True),
) -> dict[str, Any]:
    """Append rows for every item not already in `out`. Returns timing and counts."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {row["id"] for row in read_rows(out)}
    pending = [to_payload(item) for item in items if item.id not in done]
    log(f"{len(done)} rows already present; {len(pending)} to run with {workers} worker(s)")
    started = time.monotonic()
    written = 0
    if pending:
        batches = [pending[i : i + BATCH] for i in range(0, len(pending), BATCH)]
        context = multiprocessing.get_context("spawn")
        with (
            out.open("a", encoding="utf-8") as sink,
            context.Pool(
                workers,
                initializer=_init_worker,
                initargs=(str(kb_root), path_index, limit, threads),
            ) as pool,
        ):
            results = pool.imap_unordered(_work, batches)
            for _ in batches:
                try:
                    rows = results.next(timeout=STALL_SECONDS)
                except multiprocessing.TimeoutError:
                    raise RuntimeError(
                        f"no batch finished in {STALL_SECONDS} s after {written} rows. Workers that "
                        "fail while starting are replaced silently, so a changed or broken "
                        "environment looks exactly like this. Nothing is lost: rerun with --resume."
                    ) from None
                for row in rows:
                    sink.write(json.dumps(row, separators=(",", ":")) + "\n")
                sink.flush()
                written += len(rows)
                if written % 200 < BATCH or written == len(pending):
                    rate = written / (time.monotonic() - started)
                    remaining = (len(pending) - written) / rate if rate else 0
                    log(
                        f"{written}/{len(pending)} rows, {rate:.2f}/s, ~{remaining / 60:.0f} min left"
                    )
    return {"ran": written, "skipped": len(done), "seconds": round(time.monotonic() - started, 1)}


def finalize(out: Path, items: Iterable[RunItem]) -> list[dict[str, Any]]:
    """Check `out` against the item list, rewrite it in item order, and return the rows."""
    expected = [item.id for item in items]
    rows = read_rows(out)
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["id"] in by_id:
            raise RuntimeError(f"{out}: item {row['id']} appears twice")
        by_id[row["id"]] = row
    missing = [i for i in expected if i not in by_id]
    unknown = sorted(set(by_id) - set(expected))
    if missing or unknown:
        raise RuntimeError(
            f"{out} does not match the item list: {len(missing)} missing "
            f"(e.g. {missing[:3]}), {len(unknown)} unknown (e.g. {unknown[:3]})"
        )
    ordered = [by_id[i] for i in expected]
    temporary = out.with_suffix(".tmp")
    temporary.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in ordered), encoding="utf-8"
    )
    temporary.replace(out)
    return ordered
