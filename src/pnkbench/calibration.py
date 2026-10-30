"""Fit Pinakes' confidence thresholds on one half of the queries and judge them on the other.

`pinakes.calibrate` fits both thresholds from the top-1 rerank scores of *unanswerable* questions:
`low_below` is their median and `high_above` their 90th percentile. Its own docstring says why a
held-out split matters: fitted on the scored set, the false-confidence rate is "partly a
measurement of the fit". So the queries are split by a fixed hash of their id
(`items.split_of`), the fit half goes to `pinakes.calibrate`, and every threshold-dependent number
in the report is read on the eval half.

Three checks keep this honest, none of them assumed:

* the thresholds Pinakes printed are re-derived here from the stored scores with Pinakes' formula;
* the thresholds are written into the KB's manifest and a sample is searched again, and Pinakes'
  own `confidence` label must equal ours on every one;
* the report shows the labels on the fit half too, so the optimism of fitting on the scored set is
  visible next to the held-out number.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from pnkbench import items as items_module
from pnkbench import kb, pinakes_api, runner, scoring
from pnkbench.items import RunItem
from pnkbench.multihop_rag import Dataset

PARITY_SAMPLE = 600


class CalibrationError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Fit:
    fitted_for: str
    low_below: float
    high_above: float
    output: str


def fit_items(items: Sequence[RunItem], dataset: Dataset) -> list[RunItem]:
    """The fit half's null queries, and a capped, fixed sample of its answerable ones."""
    answerable = {q.id: q.answerable for q in dataset.queries}
    base = [i for i in items if i.variant == items_module.BASE]
    in_fit = [i for i in base if items_module.split_of(i.base_id) == items_module.FIT]
    null = [i for i in in_fit if not answerable[i.base_id]]
    ans = sorted(
        (i for i in in_fit if answerable[i.base_id]),
        key=lambda i: items_module.calibration_order(i.base_id),
    )[: items_module.CALIBRATION_ANSWERABLE_CAP]
    return null + ans


def pinakes_percentile(values: Sequence[float], fraction: float) -> float:
    """`pinakes.calibrate._percentile`, restated so the printed thresholds can be re-derived."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))]


def expected_thresholds(null_top1: Sequence[float]) -> tuple[float, float]:
    low, high = median(null_top1), pinakes_percentile(null_top1, 0.9)
    return (high, low) if low > high else (low, high)


_BLOCK = {
    "fitted_for": re.compile(r'^fitted_for\s*=\s*"(.+)"\s*$', re.M),
    "low_below": re.compile(r"^low_below\s*=\s*(-?[0-9.]+)\s*$", re.M),
    "high_above": re.compile(r"^high_above\s*=\s*(-?[0-9.]+)\s*$", re.M),
}


def parse_block(output: str) -> Fit:
    found = {name: pattern.search(output) for name, pattern in _BLOCK.items()}
    missing = [name for name, match in found.items() if match is None]
    if missing:
        raise CalibrationError(f"`pinakes.calibrate` printed no {missing}:\n{output[-600:]}")
    return Fit(
        fitted_for=found["fitted_for"].group(1),  # type: ignore[union-attr]
        low_below=float(found["low_below"].group(1)),  # type: ignore[union-attr]
        high_above=float(found["high_above"].group(1)),  # type: ignore[union-attr]
        output=output,
    )


def run_pinakes_calibrate(kb_root: Path, questions_yaml: Path) -> Fit:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pinakes.calibrate",
            str(kb_root),
            "--questions",
            str(questions_yaml),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise CalibrationError(
            f"pinakes.calibrate exited {completed.returncode}:\n{completed.stderr[-800:]}"
        )
    return parse_block(completed.stdout)


def calibrate(
    kb_root: Path,
    run_dir: Path,
    items: Sequence[RunItem],
    dataset: Dataset,
    rows: Sequence[Mapping[str, Any]],
    index: dict[str, int],
    *,
    limit: int,
) -> dict[str, Any]:
    from pnkbench import harness

    selected = fit_items(items, dataset)
    questions = run_dir / "calibration-questions.yaml"
    questions.write_text(harness.questions_yaml(dataset, selected), encoding="utf-8")
    fit = run_pinakes_calibrate(kb_root, questions)

    info = scoring.query_info(dataset)
    rows_by_id = {r["id"]: r for r in rows}
    null_scores = [
        s
        for i in selected
        if not info[i.base_id].answerable
        and (s := scoring.top1_score(rows_by_id[i.id])) is not None
    ]
    expected_low, expected_high = expected_thresholds(null_scores)
    cross_check = {
        "null_scores": len(null_scores),
        "pinakes_low_below": fit.low_below,
        "rederived_low_below": expected_low,
        "pinakes_high_above": fit.high_above,
        "rederived_high_above": expected_high,
        # Pinakes prints four decimals, so agreement means agreement to that precision.
        "agrees": abs(fit.low_below - expected_low) < 6e-5
        and abs(fit.high_above - expected_high) < 6e-5,
    }
    if not cross_check["agrees"]:
        raise CalibrationError(
            f"the printed thresholds do not match the stored scores: {cross_check}"
        )

    manifest_file = kb_root / "pinakes.toml"
    manifest_file.write_text(
        kb.set_confidence_thresholds(
            manifest_file.read_text(encoding="utf-8"),
            fitted_for=fit.fitted_for,
            low_below=fit.low_below,
            high_above=fit.high_above,
        ),
        encoding="utf-8",
    )

    parity = parity_check(kb_root, items, index, rows_by_id, fit, limit=limit)
    counts = {
        "fit_null": sum(1 for i in selected if not info[i.base_id].answerable),
        "fit_answerable_used": sum(1 for i in selected if info[i.base_id].answerable),
        "eval_null": sum(
            1
            for q in dataset.queries
            if not q.answerable and items_module.split_of(q.id) == items_module.EVAL
        ),
        "eval_answerable": sum(
            1
            for q in dataset.queries
            if q.answerable and items_module.split_of(q.id) == items_module.EVAL
        ),
    }
    result = {
        "split": {
            "salt": items_module.SPLIT_SALT,
            "rule": "sha256(salt:id)[0] even -> fit",
            **counts,
        },
        "thresholds": {
            "fitted_for": fit.fitted_for,
            "low_below": fit.low_below,
            "high_above": fit.high_above,
        },
        "cross_check_against_stored_scores": cross_check,
        "parity_with_pinakes_labels": parity,
        "pinakes_calibrate_output": fit.output,
    }
    return result


def parity_check(
    kb_root: Path,
    items: Sequence[RunItem],
    index: dict[str, int],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    fit: Fit,
    *,
    limit: int,
) -> dict[str, Any]:
    """Search a fixed spread of base queries again, now that the manifest carries thresholds."""
    base = [i for i in items if i.variant == items_module.BASE]
    step = max(1, len(base) // PARITY_SAMPLE)
    chosen = base[::step][:PARITY_SAMPLE]
    handle = pinakes_api.open_kb(kb_root)
    label_mismatches: list[str] = []
    worst = 0.0
    try:
        for item in chosen:
            again = runner.run_one(handle, index, runner.to_payload(item), limit)
            score = scoring.top1_score(again)
            ours = scoring.confidence_label(score, fit.low_below, fit.high_above)
            if again["conf"] != ours:
                label_mismatches.append(item.id)
            before = scoring.top1_score(rows_by_id[item.id])
            if score is not None and before is not None:
                worst = max(worst, abs(score - before))
    finally:
        handle.close()
    return {
        "checked": len(chosen),
        "label_mismatches": len(label_mismatches),
        "mismatch_ids": label_mismatches[:5],
        "max_abs_top1_difference_vs_recorded": worst,
    }


def thresholds_from(calibration: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    return None if calibration is None else calibration["thresholds"]
