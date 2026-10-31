"""A configuration is one named way of building the knowledge base.

Two things vary between configurations: the provider stamped by `pnk init --backend`, and the
`[chunking] metadata` setting. Everything else is Pinakes' own default on purpose: the baseline
measures Pinakes as a user would meet it, and a variant changes exactly one thing.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pnkbench import paths

BACKENDS = ("st", "light")
PROVIDERS = {"st": "sentence-transformers", "light": "fastembed"}
"""The provider `pnk init --backend <key>` stamps into the manifest, for both models."""
CHUNKING_METADATA = ("off", "prefix")


class ConfigError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class Config:
    name: str
    description: str
    backend: str
    chunking_metadata: str
    chunking_max_tokens: int | None = None
    """`[chunking] max_tokens`; `None` keeps Pinakes' own default (510).

    `metadata = "prefix"` needs it lowered: Pinakes refuses to build when a chunk plus its document's
    title would overrun the embedding model's window, and says how far to lower it.
    """


def parse(name: str, raw: dict[str, Any]) -> Config:
    unknown = set(raw) - {"description", "backend", "chunking_metadata", "chunking_max_tokens"}
    if unknown:
        raise ConfigError(f"config {name!r}: unknown keys {sorted(unknown)}")
    backend = str(raw.get("backend", "st"))
    if backend not in BACKENDS:
        raise ConfigError(f"config {name!r}: backend must be one of {BACKENDS}, not {backend!r}")
    metadata = str(raw.get("chunking_metadata", "off"))
    if metadata not in CHUNKING_METADATA:
        raise ConfigError(
            f"config {name!r}: chunking_metadata must be one of {CHUNKING_METADATA}, "
            f"not {metadata!r}"
        )
    max_tokens = raw.get("chunking_max_tokens")
    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 1 <= max_tokens <= 510
    ):
        raise ConfigError(
            f"config {name!r}: chunking_max_tokens must be an integer from 1 to 510, not {max_tokens!r}"
        )
    return Config(
        name=name,
        description=str(raw.get("description", "")),
        backend=backend,
        chunking_metadata=metadata,
        chunking_max_tokens=max_tokens,
    )


@dataclass(frozen=True, slots=True)
class ModelPin:
    model: str
    revision: str


@dataclass(frozen=True, slots=True)
class Models:
    embedding: ModelPin
    rerank: ModelPin


def load_models(directory: Path | None = None) -> Models:
    path = (directory or paths.configs_dir()) / "models.toml"
    raw = cast(dict[str, dict[str, Any]], tomllib.loads(path.read_text(encoding="utf-8")))

    def pin(section: str) -> ModelPin:
        table = raw[section]
        revision = str(table["revision"])
        if len(revision) != 40 or set(revision) - set("0123456789abcdef"):
            raise ConfigError(f"models.toml [{section}] revision must be a full 40-hex commit sha")
        return ModelPin(model=str(table["model"]), revision=revision)

    return Models(embedding=pin("embedding"), rerank=pin("rerank"))


@dataclass(frozen=True, slots=True)
class OnnxPin:
    """A fastembed model as the Hub stores it: a repository, a commit of it, one file and its hash.

    fastembed takes no revision, so Pinakes cannot pin these files; the benchmark records which
    ones were loaded and refuses to run on any others.
    """

    source: str
    revision: str
    file: str
    sha256: str


@dataclass(frozen=True, slots=True)
class OnnxPins:
    embedding: OnnxPin
    rerank: OnnxPin


def load_onnx_pins(directory: Path | None = None) -> OnnxPins:
    path = (directory or paths.configs_dir()) / "models.toml"
    raw = cast(
        dict[str, dict[str, dict[str, Any]]], tomllib.loads(path.read_text(encoding="utf-8"))
    )

    def pin(section: str) -> OnnxPin:
        table = raw.get("fastembed", {}).get(section)
        if table is None:
            raise ConfigError(f"models.toml has no [fastembed.{section}] table")
        revision, digest = str(table["revision"]), str(table["sha256"])
        if len(revision) != 40 or set(revision) - set("0123456789abcdef"):
            raise ConfigError(
                f"models.toml [fastembed.{section}] revision must be a full 40-hex commit sha"
            )
        if len(digest) != 64 or set(digest) - set("0123456789abcdef"):
            raise ConfigError(f"models.toml [fastembed.{section}] sha256 must be 64 hex digits")
        return OnnxPin(
            source=str(table["source"]), revision=revision, file=str(table["file"]), sha256=digest
        )

    return OnnxPins(embedding=pin("embedding"), rerank=pin("rerank"))


def load(name: str, directory: Path | None = None) -> Config:
    directory = directory or paths.configs_dir()
    path = directory / f"{name}.toml"
    if not path.is_file():
        available = sorted(p.stem for p in directory.glob("*.toml"))
        raise ConfigError(f"no config named {name!r} in {directory} (available: {available})")
    return parse(name, cast(dict[str, Any], tomllib.loads(path.read_text(encoding="utf-8"))))
