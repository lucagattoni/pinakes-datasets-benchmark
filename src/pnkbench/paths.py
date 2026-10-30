"""Where things live.

One place, so a test can point the whole tool at a temporary directory with `PNKBENCH_ROOT`
instead of every module guessing its own relative path.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_ENV = "PNKBENCH_ROOT"


def repo_root() -> Path:
    override = os.environ.get(ROOT_ENV)
    if override:
        return Path(override).resolve()
    # src/pnkbench/paths.py -> the repository root is two levels above the package directory.
    return Path(__file__).resolve().parents[2]


def dataset_dir(name: str) -> Path:
    """One folder per dataset: its README, its manifest, and (git-ignored) its fetched files."""
    return repo_root() / "datasets" / name


def raw_data_dir(name: str) -> Path:
    """Fetched third-party files. Git-ignored: the manifest next to it is the record."""
    return dataset_dir(name) / "raw"


def build_dir() -> Path:
    """Generated knowledge bases. Git-ignored and always safe to delete."""
    return repo_root() / "build"


def kb_dir(config: str) -> Path:
    return build_dir() / "kb" / config


def runs_dir() -> Path:
    """Committed results: header, rows, summary, report."""
    return repo_root() / "runs"


def configs_dir() -> Path:
    return repo_root() / "configs"
