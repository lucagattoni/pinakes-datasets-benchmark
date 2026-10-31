"""The `light` backend: fastembed answers to the same model ids but loads other files.

Fast, no models. What is pinned, what is recorded and what stops a run:

* `configs/models.toml` pins the files fastembed loads (it takes no revision, so Pinakes cannot);
* `pin_models` writes no revision for `light` and refuses a provider other than the backend's;
* `loaded_artefacts` reads what fastembed actually opened, never what the manifest asked for;
* `verify_artefacts` stops a build or a run on any other file;
* `pipeline.run` and `kb.build_kb` record the files, and the run no longer needs torch.

Expected values come from outside the code under test: the Hub's own answers on 20261031 16:22 UTC
for the shipped pins, the FIPS 180 test vector for SHA-256 of "abc", and hand arithmetic.
"""

from __future__ import annotations

import builtins
import dataclasses
import hashlib
import json
import platform
import re
import shutil
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_report_format import _abstention
from test_round2_fast import (
    _fake_pinakes,
    _fake_stack,
    _finished_run,
    _header,
    _header_of,
    _resume_world,
    _row,
    _toy_repo,
)
from toy import make_dataset

from pnkbench import compare, kb, paths, pinakes_api, pipeline, report, runner
from pnkbench import config as config_module
from pnkbench.config import (
    BACKENDS,
    PROVIDERS,
    ConfigError,
    ModelPin,
    Models,
    OnnxPin,
    OnnxPins,
    load_onnx_pins,
)

REAL_CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SHA256_OF_ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

# --- configs/models.toml: the shipped pins -------------------------------------------------------


def test_the_shipped_pins_are_the_files_checked_against_the_hub() -> None:
    pins = load_onnx_pins(REAL_CONFIGS)
    assert pins.embedding == OnnxPin(
        source="Qdrant/bge-small-en-v1.5-onnx-Q",
        revision="aa8f8b060edb00e03bfdd08813a2949946c8ba55",
        file="model_optimized.onnx",
        sha256="51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431",
    )
    assert pins.rerank == OnnxPin(
        source="BAAI/bge-reranker-base",
        revision="2cfc18c9415c912f9d8155881c133215df768a70",
        file="onnx/model.onnx",
        sha256="15b9a8c3da82eddf263df571281166e00e9308fe19d077084b642ebfcaf06d2b",
    )


def test_the_fastembed_pins_leave_the_sentence_transformers_pins_alone() -> None:
    models = config_module.load_models(REAL_CONFIGS)
    assert models.embedding == ModelPin(
        "BAAI/bge-small-en-v1.5", "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    )
    assert models.rerank == ModelPin(
        "BAAI/bge-reranker-base", "2cfc18c9415c912f9d8155881c133215df768a70"
    )


def _models_toml(tmp_path: Path, text: str) -> Path:
    directory = tmp_path / "configs"
    directory.mkdir()
    (directory / "models.toml").write_text(text, encoding="utf-8")
    return directory


def _table(section: str, *, revision: str = "a" * 40, sha256: str = "b" * 64) -> str:
    return (
        f'[fastembed.{section}]\nsource = "x/y"\nrevision = "{revision}"\n'
        f'file = "f.onnx"\nsha256 = "{sha256}"\n'
    )


def test_pins_with_well_formed_values_load(tmp_path: Path) -> None:
    directory = _models_toml(tmp_path, _table("embedding") + _table("rerank", revision="c" * 40))
    pins = load_onnx_pins(directory)
    assert pins.embedding == OnnxPin("x/y", "a" * 40, "f.onnx", "b" * 64)
    assert pins.rerank.revision == "c" * 40


@pytest.mark.parametrize("section", ["embedding", "rerank"])
@pytest.mark.parametrize(
    ("field", "value", "phrase"),
    [
        ("revision", "a" * 39, "revision must be a full 40-hex commit sha"),
        ("revision", "a" * 41, "revision must be a full 40-hex commit sha"),
        ("revision", "A" * 40, "revision must be a full 40-hex commit sha"),
        ("revision", "g" * 40, "revision must be a full 40-hex commit sha"),
        ("sha256", "b" * 63, "sha256 must be 64 hex digits"),
        ("sha256", "b" * 65, "sha256 must be 64 hex digits"),
        ("sha256", "B" * 64, "sha256 must be 64 hex digits"),
        ("sha256", "z" * 64, "sha256 must be 64 hex digits"),
    ],
)
def test_a_malformed_pin_is_refused_and_names_its_table(
    tmp_path: Path, section: str, field: str, value: str, phrase: str
) -> None:
    other = "rerank" if section == "embedding" else "embedding"
    text = _table(other) + _table(section, **{field: value})
    with pytest.raises(ConfigError, match=rf"\[fastembed\.{section}\] {phrase}"):
        load_onnx_pins(_models_toml(tmp_path, text))


@pytest.mark.parametrize("missing", ["embedding", "rerank"])
def test_a_missing_pin_table_is_a_config_error_not_a_key_error(
    tmp_path: Path, missing: str
) -> None:
    present = "rerank" if missing == "embedding" else "embedding"
    with pytest.raises(ConfigError, match=rf"no \[fastembed\.{missing}\] table"):
        load_onnx_pins(_models_toml(tmp_path, _table(present)))


def test_a_models_file_with_no_fastembed_section_is_a_config_error(tmp_path: Path) -> None:
    text = '[embedding]\nmodel = "m"\nrevision = "' + "a" * 40 + '"\n'
    with pytest.raises(ConfigError, match=r"no \[fastembed\.embedding\] table"):
        load_onnx_pins(_models_toml(tmp_path, text))


def test_the_light_configuration_asks_for_fastembed_and_nothing_else() -> None:
    config = config_module.load("baseline-fastembed", REAL_CONFIGS)
    assert (config.backend, config.chunking_metadata, config.chunking_max_tokens) == (
        "light",
        "off",
        None,
    )


def test_every_backend_has_exactly_one_provider() -> None:
    assert PROVIDERS == {"st": "sentence-transformers", "light": "fastembed"}
    assert set(PROVIDERS) == set(BACKENDS)


# --- kb.pin_models for each backend --------------------------------------------------------------


def _manifest(embedding_provider: str, rerank_provider: str, *, model: str = "e-model") -> str:
    return (
        f'[embedding]\nprovider = "{embedding_provider}"\nmodel = "{model}"\n\n'
        f'[rerank]\nprovider = "{rerank_provider}"\nmodel = "r-model"\n'
    )


_PINS = Models(embedding=ModelPin("e-model", "5" * 40), rerank=ModelPin("r-model", "2" * 40))


def test_light_pins_no_revision_and_changes_nothing_in_the_manifest() -> None:
    text = _manifest("fastembed", "fastembed")
    out = kb.pin_models(text, _PINS, backend="light")
    parsed = tomllib.loads(out)
    assert "revision" not in parsed["embedding"]
    assert "revision" not in parsed["rerank"]
    assert out == text


def test_st_still_pins_both_revisions() -> None:
    out = kb.pin_models(_manifest("sentence-transformers", "sentence-transformers"), _PINS)
    parsed = tomllib.loads(out)
    assert parsed["embedding"]["revision"] == "5" * 40
    assert parsed["rerank"]["revision"] == "2" * 40


@pytest.mark.parametrize(
    ("backend", "embedding", "rerank", "table"),
    [
        ("light", "sentence-transformers", "fastembed", "embedding"),
        ("light", "fastembed", "sentence-transformers", "rerank"),
        ("light", "sentence-transformers", "sentence-transformers", "embedding"),
        ("st", "fastembed", "sentence-transformers", "embedding"),
        ("st", "sentence-transformers", "fastembed", "rerank"),
        ("st", "fastembed", "fastembed", "embedding"),
    ],
)
def test_a_provider_other_than_the_backends_is_a_stop(
    backend: str, embedding: str, rerank: str, table: str
) -> None:
    with pytest.raises(kb.BuildError, match=rf"\[{table}\] provider .*backend {backend!r}"):
        kb.pin_models(_manifest(embedding, rerank), _PINS, backend=backend)


def test_a_manifest_that_names_no_provider_is_a_stop() -> None:
    text = '[embedding]\nmodel = "e-model"\n\n[rerank]\nmodel = "r-model"\n'
    with pytest.raises(kb.BuildError, match=r"\[embedding\] provider None"):
        kb.pin_models(text, _PINS, backend="light")


def test_light_still_stops_on_a_different_model_id() -> None:
    text = _manifest("fastembed", "fastembed", model="other-model")
    with pytest.raises(kb.BuildError, match=r"\[embedding\] model 'other-model'"):
        kb.pin_models(text, _PINS, backend="light")


# --- kb.verify_artefacts ------------------------------------------------------------------------

_ONNX = OnnxPins(
    embedding=OnnxPin("a/b", "1" * 40, "m.onnx", "2" * 64),
    rerank=OnnxPin("c/d", "3" * 40, "n.onnx", "4" * 64),
)


def _found() -> dict[str, dict[str, Any]]:
    return {
        "embedding": {
            "source": "a/b",
            "revision": "1" * 40,
            "file": "m.onnx",
            "sha256": "2" * 64,
            "execution_providers": ["CPUExecutionProvider"],
        },
        "rerank": {
            "source": "c/d",
            "revision": "3" * 40,
            "file": "n.onnx",
            "sha256": "4" * 64,
            "execution_providers": ["CPUExecutionProvider"],
        },
    }


def test_the_files_that_were_pinned_pass_whatever_else_is_recorded() -> None:
    kb.verify_artefacts(_found(), _ONNX)


@pytest.mark.parametrize("table", ["embedding", "rerank"])
@pytest.mark.parametrize("field", ["source", "revision", "file", "sha256"])
def test_any_one_field_that_differs_stops_the_run_and_is_named(table: str, field: str) -> None:
    found = _found()
    found[table][field] = "changed"
    pinned = getattr(getattr(_ONNX, table), field)
    with pytest.raises(kb.BuildError) as raised:
        kb.verify_artefacts(found, _ONNX)
    message = str(raised.value)
    assert f"{table} {field}: loaded 'changed', pinned {pinned!r}" in message
    assert message.count("\n  - ") == 1  # only the one difference


@pytest.mark.parametrize("table", ["embedding", "rerank"])
def test_a_model_fastembed_did_not_load_stops_the_run(table: str) -> None:
    found = _found()
    del found[table]
    with pytest.raises(kb.BuildError, match=rf"the {table} model was not loaded by fastembed"):
        kb.verify_artefacts(found, _ONNX)


def test_every_difference_is_listed_in_one_message() -> None:
    found = _found()
    found["embedding"]["sha256"] = "x"
    found["rerank"]["revision"] = "y"
    del found["rerank"]["file"]
    with pytest.raises(kb.BuildError) as raised:
        kb.verify_artefacts(found, _ONNX)
    lines = str(raised.value).split("\n  - ")[1:]
    assert lines == [
        "embedding sha256: loaded 'x', pinned " + repr("2" * 64),
        "rerank revision: loaded 'y', pinned " + repr("3" * 40),
        "rerank file: loaded None, pinned 'n.onnx'",
    ]


# --- pinakes_api.loaded_artefacts: what fastembed actually opened -------------------------------


def _snapshot(tmp_path: Path, name: str, file: str, content: bytes) -> Path:
    directory = tmp_path / "hub" / f"models--{name}" / "snapshots" / ("7" * 40)
    (directory / file).parent.mkdir(parents=True)
    (directory / file).write_bytes(content)
    return directory


def _loaded(
    directory: Path, *, repo: str, file: str, providers: tuple[str, ...] = ("CPUExecutionProvider",)
) -> SimpleNamespace:
    """The shape fastembed builds: wrapper._model.model carries the description, dir and session."""
    session = SimpleNamespace(get_providers=lambda: list(providers))
    inner = SimpleNamespace(
        model_description=SimpleNamespace(sources=SimpleNamespace(hf=repo), model_file=file),
        _model_dir=directory,
        model=session,
    )
    return SimpleNamespace(_model=SimpleNamespace(model=inner))


def _kb(embedding: Any, reranker: Any, *, e: str = "fastembed", r: str = "fastembed") -> Any:
    return SimpleNamespace(
        manifest=SimpleNamespace(
            embedding=SimpleNamespace(provider=e), rerank=SimpleNamespace(provider=r)
        ),
        backend=embedding,
        reranker=reranker,
    )


def test_the_record_names_the_repository_the_snapshot_the_file_and_its_hash(
    tmp_path: Path,
) -> None:
    e_dir = _snapshot(tmp_path, "X--embed", "model.onnx", b"abc")
    r_dir = _snapshot(tmp_path, "Y--rank", "onnx/model.onnx", b"abc")
    handle = _kb(
        _loaded(e_dir, repo="X/embed", file="model.onnx"),
        _loaded(
            r_dir,
            repo="Y/rank",
            file="onnx/model.onnx",
            providers=("CoreMLExecutionProvider", "CPUExecutionProvider"),
        ),
    )
    assert pinakes_api.loaded_artefacts(handle) == {
        "embedding": {
            "source": "X/embed",
            "revision": "7" * 40,
            "file": "model.onnx",
            "sha256": SHA256_OF_ABC,
            "execution_providers": ["CPUExecutionProvider"],
        },
        "rerank": {
            "source": "Y/rank",
            "revision": "7" * 40,
            "file": "onnx/model.onnx",
            "sha256": SHA256_OF_ABC,
            "execution_providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        },
    }


def test_the_hash_covers_the_whole_file_not_its_first_block(tmp_path: Path) -> None:
    content = b"x" * (3 * (1 << 20) + 5) + b"tail"
    directory = _snapshot(tmp_path, "X--embed", "model.onnx", content)
    (directory / "other.onnx").write_bytes(b"not this one")
    handle = _kb(
        _loaded(directory, repo="X/embed", file="model.onnx"), None, r="sentence-transformers"
    )
    assert pinakes_api.loaded_artefacts(handle)["embedding"]["sha256"] == (
        hashlib.sha256(content).hexdigest()
    )


@pytest.mark.parametrize(
    ("e", "r", "has_reranker", "expected"),
    [
        ("sentence-transformers", "sentence-transformers", True, set()),
        ("fastembed", "sentence-transformers", True, {"embedding"}),
        ("sentence-transformers", "fastembed", True, {"rerank"}),
        ("fastembed", "fastembed", True, {"embedding", "rerank"}),
        ("fastembed", "fastembed", False, {"embedding"}),
    ],
)
def test_only_models_that_fastembed_runs_are_recorded(
    tmp_path: Path, e: str, r: str, has_reranker: bool, expected: set[str]
) -> None:
    directory = _snapshot(tmp_path, "X--m", "m.onnx", b"abc")
    model = _loaded(directory, repo="X/m", file="m.onnx")
    handle = _kb(model, model if has_reranker else None, e=e, r=r)
    assert set(pinakes_api.loaded_artefacts(handle)) == expected


def test_a_directory_that_is_no_snapshot_has_no_commit_to_record(tmp_path: Path) -> None:
    directory = tmp_path / "somewhere" / "else"
    directory.mkdir(parents=True)
    (directory / "m.onnx").write_bytes(b"abc")
    handle = _kb(_loaded(directory, repo="X/m", file="m.onnx"), None, r="sentence-transformers")
    with pytest.raises(pinakes_api.InterfaceError, match="not a Hugging Face snapshot directory"):
        pinakes_api.loaded_artefacts(handle)


@pytest.mark.parametrize("missing", ["model_description", "_model_dir", "model"])
def test_a_fastembed_that_moved_its_internals_stops_the_run(tmp_path: Path, missing: str) -> None:
    directory = _snapshot(tmp_path, "X--m", "m.onnx", b"abc")
    loaded = _loaded(directory, repo="X/m", file="m.onnx")
    delattr(loaded._model.model, missing)
    handle = _kb(loaded, None, r="sentence-transformers")
    with pytest.raises(
        pinakes_api.InterfaceError, match="cannot tell which files fastembed loaded"
    ):
        pinakes_api.loaded_artefacts(handle)


def test_a_session_that_cannot_name_its_providers_stops_the_run(tmp_path: Path) -> None:
    directory = _snapshot(tmp_path, "X--m", "m.onnx", b"abc")
    loaded = _loaded(directory, repo="X/m", file="m.onnx")
    loaded._model.model.model = SimpleNamespace()
    handle = _kb(loaded, None, r="sentence-transformers")
    with pytest.raises(
        pinakes_api.InterfaceError, match="cannot tell which files fastembed loaded"
    ):
        pinakes_api.loaded_artefacts(handle)


def test_a_wrapper_with_no_model_at_all_stops_the_run() -> None:
    handle = _kb(SimpleNamespace(), None, r="sentence-transformers")
    with pytest.raises(
        pinakes_api.InterfaceError, match="cannot tell which files fastembed loaded"
    ):
        pinakes_api.loaded_artefacts(handle)


# --- pipeline.environment ------------------------------------------------------------------------


def test_an_environment_without_torch_or_onnxruntime_records_their_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` then raises ImportError
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    record = pipeline.environment()
    assert record["torch"] is None
    assert record["mps_available"] is None
    assert record["onnxruntime"] is None
    assert record["python"] == platform.python_version()
    assert record["platform"] == platform.platform()


@pytest.mark.parametrize("mps", [True, False])
def test_an_environment_with_both_libraries_records_their_versions(
    monkeypatch: pytest.MonkeyPatch, mps: bool
) -> None:
    fake_torch = SimpleNamespace(
        __version__="2.9.9",
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(__version__="1.2.3"))
    record = pipeline.environment()
    assert (record["torch"], record["mps_available"], record["onnxruntime"]) == (
        "2.9.9",
        mps,
        "1.2.3",
    )


def test_an_environment_with_only_onnxruntime_has_no_torch_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(__version__="1.2.3"))
    record = pipeline.environment()
    assert (record["torch"], record["mps_available"], record["onnxruntime"]) == (
        None,
        None,
        "1.2.3",
    )


# --- pipeline.run on the light configuration -----------------------------------------------------


def _record_matching(pin: OnnxPin, providers: list[str]) -> dict[str, Any]:
    return {
        "source": pin.source,
        "revision": pin.revision,
        "file": pin.file,
        "sha256": pin.sha256,
        "execution_providers": providers,
    }


def _light_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A toy repository with a KB for `baseline-fastembed`, and fastembed's files as pinned."""
    _resume_world(tmp_path, monkeypatch)
    paths.kb_dir("baseline-fastembed").mkdir(parents=True)
    (paths.kb_dir("baseline-fastembed") / "pinakes.toml").write_text("")
    _fake_stack(monkeypatch, {"retrieval": {"fusion_top_k": 20}})
    pins = load_onnx_pins(REAL_CONFIGS)
    found = {
        "embedding": _record_matching(pins.embedding, ["CPUExecutionProvider"]),
        "rerank": _record_matching(
            pins.rerank, ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        ),
    }
    monkeypatch.setattr(pinakes_api, "loaded_artefacts", lambda handle: found)
    return found


def test_a_light_run_records_the_files_and_what_they_ran_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = _light_world(tmp_path, monkeypatch)
    run_dir = pipeline.run("baseline-fastembed", workers=1, log=lambda m: None)
    header = _header_of(run_dir)
    assert header["artefacts"] == found
    assert header["config"]["backend"] == "light"
    assert header["environment"]["devices"] == {
        "embedding": "CPUExecutionProvider",
        "rerank": "CoreMLExecutionProvider+CPUExecutionProvider",
    }
    assert header["status"] == "complete"


def test_a_light_run_on_other_files_stops_before_a_single_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = _light_world(tmp_path, monkeypatch)
    found["embedding"]["sha256"] = "0" * 64
    searched: list[object] = []
    monkeypatch.setattr(runner, "run_items", lambda *a, **k: searched.append(1))
    with pytest.raises(kb.BuildError, match=r"embedding sha256: loaded '0{64}'"):
        pipeline.run("baseline-fastembed", workers=1, log=lambda m: None)
    assert searched == []
    assert list(paths.runs_dir().iterdir()) == []  # no directory left to collide with a retry


def test_a_light_run_with_no_fastembed_model_loaded_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _light_world(tmp_path, monkeypatch)
    monkeypatch.setattr(pinakes_api, "loaded_artefacts", lambda handle: {})
    with pytest.raises(kb.BuildError, match="was not loaded by fastembed"):
        pipeline.run("baseline-fastembed", workers=1, log=lambda m: None)


def test_a_sentence_transformers_run_records_no_artefacts_and_is_not_checked_against_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _resume_world(tmp_path, monkeypatch)
    _fake_stack(monkeypatch, {"retrieval": {"fusion_top_k": 20}})
    header = _header_of(pipeline.run("baseline", workers=1, log=lambda m: None))
    assert "artefacts" not in header
    assert header["config"]["backend"] == "st"


def test_a_resume_refuses_when_the_files_fastembed_loads_have_changed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = _light_world(tmp_path, monkeypatch)
    run_dir = pipeline.run("baseline-fastembed", workers=1, log=lambda m: None)
    pins = load_onnx_pins(REAL_CONFIGS)
    # the pins still pass, so only the recorded providers differ: that is a change of model files
    # to the resume check, which compares the whole record
    found["rerank"] = _record_matching(pins.rerank, ["CPUExecutionProvider"])
    with pytest.raises(RuntimeError, match="cannot resume: model files differ"):
        pipeline.run("baseline-fastembed", workers=1, resume=run_dir, log=lambda m: None)


def test_a_resume_with_the_same_files_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _light_world(tmp_path, monkeypatch)
    run_dir = pipeline.run("baseline-fastembed", workers=1, log=lambda m: None)
    assert (
        pipeline.run("baseline-fastembed", workers=1, resume=run_dir, log=lambda m: None) == run_dir
    )


def test_a_run_started_before_artefacts_were_recorded_still_resumes_as_sentence_transformers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _resume_world(tmp_path, monkeypatch)
    _fake_stack(monkeypatch, {"retrieval": {"fusion_top_k": 20}})
    run_dir = pipeline.run("baseline", workers=1, log=lambda m: None)
    header = _header_of(run_dir)
    assert "artefacts" not in header  # the shape of every header written before this change
    assert pipeline.run("baseline", workers=1, resume=run_dir, log=lambda m: None) == run_dir


# --- kb.build_kb on the light configuration ------------------------------------------------------

_LIGHT_MANIFEST = """[kb]
name = "probe"

[embedding]
provider = "fastembed"
model = "e-model"

[chunking]
strategy = "structural"

[retrieval]
rerank = "local"

[rerank]
provider = "fastembed"
model = "r-model"
"""


def _light_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, found: dict[str, Any]
) -> SimpleNamespace:
    """Replace Pinakes; returns what `pnk init` was asked and how often the KB was closed."""
    shutil.copytree(REAL_CONFIGS, tmp_path / "configs")
    init: list[list[str]] = []

    def run_pnk(args: list[str], *, log: Path | None = None) -> str:
        if args[0] == "init":
            init.append(list(args))
            Path(args[1]).mkdir(parents=True, exist_ok=True)
            (Path(args[1]) / "pinakes.toml").write_text(_LIGHT_MANIFEST, encoding="utf-8")
        return ""

    monkeypatch.setenv(paths.ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(pinakes_api, "require_interfaces", lambda: None)
    monkeypatch.setattr(pinakes_api, "run_pnk", run_pnk)
    monkeypatch.setattr(pinakes_api, "mint_sidecar", lambda *a, **k: None)
    monkeypatch.setattr(kb, "load_models", lambda: _PINS)
    monkeypatch.setattr(kb, "verify_kb", lambda dataset, root: {"documents": 12, "chunks": 7})
    closed: list[int] = []
    handle = SimpleNamespace(close=lambda: closed.append(1))
    monkeypatch.setattr(pinakes_api, "open_kb", lambda root: handle)
    monkeypatch.setattr(pinakes_api, "loaded_artefacts", lambda h: found)
    return SimpleNamespace(init=init, closed=closed)


def _build_record(name: str) -> Path:
    return paths.kb_dir(name).parent / f"{name}.build.json"


def test_a_light_build_initialises_with_the_light_backend_pins_no_revision_and_records_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pins = load_onnx_pins(REAL_CONFIGS)
    found = {
        "embedding": _record_matching(pins.embedding, ["CPUExecutionProvider"]),
        "rerank": _record_matching(pins.rerank, ["CPUExecutionProvider"]),
    }
    fake = _light_build(monkeypatch, tmp_path, found)
    config = config_module.load("baseline-fastembed", REAL_CONFIGS)
    report = kb.build_kb(make_dataset(), config, log=lambda m: None)

    assert fake.init == [
        [
            "init",
            str(paths.kb_dir("baseline-fastembed")),
            "--backend",
            "light",
            "--name",
            "mhr-baseline-fastembed",
        ]
    ]
    manifest = tomllib.loads(
        (paths.kb_dir("baseline-fastembed") / "pinakes.toml").read_text(encoding="utf-8")
    )
    assert "revision" not in manifest["embedding"]
    assert "revision" not in manifest["rerank"]
    assert report["model_files"] == found
    written = json.loads(_build_record("baseline-fastembed").read_text())
    assert written["model_files"] == found
    assert fake.closed == [1]  # the KB was closed


def test_a_light_build_on_other_files_stops_and_writes_no_build_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pins = load_onnx_pins(REAL_CONFIGS)
    found = {
        "embedding": _record_matching(pins.embedding, ["CPUExecutionProvider"]),
        "rerank": {**_record_matching(pins.rerank, ["CPUExecutionProvider"]), "revision": "9" * 40},
    }
    fake = _light_build(monkeypatch, tmp_path, found)
    config = config_module.load("baseline-fastembed", REAL_CONFIGS)
    with pytest.raises(kb.BuildError, match=r"rerank revision: loaded '9{40}'"):
        kb.build_kb(make_dataset(), config, log=lambda m: None)
    assert not _build_record("baseline-fastembed").exists()
    assert fake.closed == [1]  # closed even on a stop


def test_a_sentence_transformers_build_opens_no_model_and_records_no_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_pinakes(monkeypatch, tmp_path)
    opened: list[object] = []
    monkeypatch.setattr(pinakes_api, "open_kb", lambda root: opened.append(root))
    report = kb.build_kb(
        make_dataset(), config_module.Config("probe", "", "st", "off"), log=lambda m: None
    )
    assert opened == []
    assert "model_files" not in report


# --- the report: what it says about a run on each provider ---------------------------------------


def test_the_score_scale_is_described_for_the_provider_that_produced_it() -> None:
    st = report._abstention_section(_abstention(), None, "sentence-transformers")
    assert "probabilities between 0 and 1" in st
    assert "logit" not in st.lower()
    fastembed = report._abstention_section(_abstention(), None, "fastembed")
    assert (
        "raw output column without a sigmoid, so these are logits, which can be negative"
        in fastembed
    )
    assert "probabilities" not in fastembed


@pytest.mark.parametrize(("provider", "quoted"), [(None, "None"), ("other", "'other'")])
def test_a_provider_the_report_knows_nothing_about_gets_no_claimed_scale(
    provider: str | None, quoted: str
) -> None:
    text = report._abstention_section(_abstention(), None, provider)
    assert f"does not state the scale of provider {quoted}" in text
    assert "probabilities" not in text
    assert "logit" not in text.lower()


def _artefacts() -> dict[str, dict[str, Any]]:
    return {
        "embedding": {
            "source": "Q/x",
            "revision": "a" * 40,
            "file": "m.onnx",
            "sha256": "b" * 64,
            "execution_providers": ["CPUExecutionProvider"],
        },
        "rerank": {
            "source": "B/y",
            "revision": "c" * 40,
            "file": "n.onnx",
            "sha256": "d" * 64,
            "execution_providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        },
    }


def _rows(text: str) -> list[str]:
    return text.split("\n")[4:]  # past the heading, the blank line, the table's head and separator


def test_the_header_names_the_files_fastembed_loaded_between_the_models_and_the_chunking() -> None:
    header = _header()
    header["artefacts"] = _artefacts()
    rows = _rows(report._header_section(header))
    names = [row[2:].split(" |")[0] for row in rows]
    assert names.index("Model files") == names.index("Reranker") + 1
    assert names.index("Chunking") == names.index("Model files") + 1
    assert (
        "| Model files | embedding: `Q/x` @ `aaaaaaaaaa`, `m.onnx`, sha256 `bbbbbbbbbbbbbbbb`, "
        "ran on CPUExecutionProvider; rerank: `B/y` @ `cccccccccc`, `n.onnx`, sha256 "
        "`dddddddddddddddd`, ran on CoreMLExecutionProvider+CPUExecutionProvider |" in rows
    )


@pytest.mark.parametrize("artefacts", [None, {}])
def test_a_header_with_no_recorded_files_has_no_model_files_row(artefacts: Any) -> None:
    header = _header()
    if artefacts is not None:
        header["artefacts"] = artefacts
    assert "Model files" not in report._header_section(header)


@pytest.mark.parametrize(
    ("torch", "onnxruntime", "expected"),
    [
        ("2.9.9", None, "Python 3.13, torch 2.9.9, plat"),
        (None, "1.2.3", "Python 3.13, onnxruntime 1.2.3, plat"),
        ("2.9.9", "1.2.3", "Python 3.13, torch 2.9.9, onnxruntime 1.2.3, plat"),
        (None, None, "Python 3.13, plat"),
    ],
)
def test_the_environment_row_names_only_the_libraries_the_run_had(
    torch: str | None, onnxruntime: str | None, expected: str
) -> None:
    header = _header()
    header["environment"] = {
        "python": "3.13",
        "torch": torch,
        "onnxruntime": onnxruntime,
        "platform": "plat",
        "devices": {},
    }
    assert f"| Environment | {expected} |" in report._header_section(header)


def test_an_older_header_without_the_onnxruntime_key_is_still_described() -> None:
    header = _header()  # the shape of every header written before the fastembed backend existed
    header["environment"] = {"python": "3.13", "torch": "2.0", "platform": "plat", "devices": {}}
    assert "| Environment | Python 3.13, torch 2.0, plat |" in report._header_section(header)


@pytest.mark.parametrize(
    ("rerank", "expected"),
    [
        ({"model": "r", "revision": None, "provider": "fastembed"}, "fastembed"),
        (
            {"model": "r", "revision": None, "provider": "sentence-transformers"},
            "sentence-transformers",
        ),
        (None, None),
    ],
)
def test_the_report_describes_the_scale_of_the_reranker_the_run_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rerank: Any, expected: str | None
) -> None:
    header = _header()
    header["run_id"] = "r"
    header["settings"]["rerank"] = rerank
    header["settings"]["embedding"]["provider"] = "the-embedding-provider"
    (tmp_path / "run.json").write_text(json.dumps(header))
    summary = {"answerable": {}, "abstention": {}, "filters": {}, "latency_ms": None}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    seen: list[Any] = []
    monkeypatch.setattr(report, "_header_section", lambda h, scored_by=None: "")
    monkeypatch.setattr(report, "_answerable_section", lambda a: "")
    monkeypatch.setattr(report, "_filters_section", lambda f: "")
    monkeypatch.setattr(report, "_latency_section", lambda q, h: "")
    monkeypatch.setattr(
        report, "_abstention_section", lambda ab, calibration, provider: seen.append(provider) or ""
    )
    report.render_report(tmp_path)
    assert seen == [expected]


# --- compare: do the thresholds of one provider mean anything on the other's scores --------------

# The toy dataset's halves (pinned in test_layout_and_items): eval is mhr-0000, 0001, 0004, 0007,
# 0008, 0009 and fit the other four; mhr-0005 and mhr-0007 are the null questions. So the held-out
# half is five answerable questions and one null one, and mhr-0005 (null, fit half) must not count.
_A_THRESHOLDS = {"low_below": 0.3, "high_above": 0.7}  # a probability scale
_B_THRESHOLDS = {"low_below": -2.0, "high_above": 3.0}  # a logit scale
# top-1 scores, with some placed exactly on a threshold: only strictly outside is low or high
_A_SCORES = {0: 0.9, 1: 0.7, 4: 0.3, 8: 0.2, 9: 0.75, 7: 0.1}
_B_SCORES = {0: 4.0, 1: 2.5, 4: 3.0, 8: -2.0, 9: -3.0, 7: 1.5}


def _scored_rows(scores: dict[int, float], fit_score: float) -> list[dict[str, Any]]:
    return [_row(f"mhr-{i:04d}", [[0, scores.get(i, fit_score)]]) for i in range(10)]


def _calibrated(run_dir: Path, thresholds: dict[str, float] | None) -> None:
    if thresholds is not None:
        (run_dir / "calibration.json").write_text(json.dumps({"thresholds": thresholds}))


def _transfer_of(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    a: dict[str, float] | None,
    b: dict[str, float] | None,
) -> dict[str, Any]:
    _toy_repo(tmp_path, monkeypatch)
    run_a = _finished_run("run-a", "baseline", None, _scored_rows(_A_SCORES, 0.5))
    run_b = _finished_run("run-b", "baseline-fastembed", None, _scored_rows(_B_SCORES, 0.0))
    _calibrated(run_a, a)
    _calibrated(run_b, b)
    return compare.compare_runs(run_a, run_b, make_dataset())


def _counts(n: int, high: int = 0, medium: int = 0, low: int = 0) -> dict[str, int]:
    return {"n": n, "high": high, "medium": medium, "low": low, "unknown": 0}


def test_each_runs_held_out_scores_are_labelled_by_its_own_thresholds_and_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transfer = _transfer_of(tmp_path, monkeypatch, _A_THRESHOLDS, _B_THRESHOLDS)[
        "threshold_transfer"
    ]
    assert transfer["a_thresholds"] == _A_THRESHOLDS
    assert transfer["b_thresholds"] == _B_THRESHOLDS
    # A's own: 0.9 high, 0.7 medium (not above 0.7), 0.3 medium (not below 0.3), 0.2 low, 0.75 high
    # B's own: 4.0 high, 2.5 medium, 3.0 medium (not above 3.0), -2.0 medium (not below -2.0), -3.0 low
    # B on A's: 4.0, 2.5, 3.0 high; -2.0, -3.0 low; the null 1.5 is above 0.7, so high
    # A on B's: every probability lies between -2 and 3, so everything is medium
    assert transfer["labels"] == {
        "a_scores_a_thresholds": {
            "answerable": _counts(5, high=2, medium=2, low=1),
            "null": _counts(1, high=0, medium=0, low=1),
        },
        "b_scores_b_thresholds": {
            "answerable": _counts(5, high=1, medium=3, low=1),
            "null": _counts(1, high=0, medium=1, low=0),
        },
        "b_scores_a_thresholds": {
            "answerable": _counts(5, high=3, medium=0, low=2),
            "null": _counts(1, high=1, medium=0, low=0),
        },
        "a_scores_b_thresholds": {
            "answerable": _counts(5, high=0, medium=5, low=0),
            "null": _counts(1, high=0, medium=1, low=0),
        },
    }


@pytest.mark.parametrize(
    ("a", "b"),
    [(None, _B_THRESHOLDS), (_A_THRESHOLDS, None), (None, None)],
)
def test_there_is_no_transfer_to_read_unless_both_runs_were_calibrated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    a: dict[str, float] | None,
    b: dict[str, float] | None,
) -> None:
    assert "threshold_transfer" not in _transfer_of(tmp_path, monkeypatch, a, b)


def test_the_transfer_table_reads_in_the_same_order_as_the_labels_it_comes_from(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _transfer_of(tmp_path, monkeypatch, _A_THRESHOLDS, _B_THRESHOLDS)
    text = report.render_compare(result)
    section = text.split("## Do the confidence thresholds transfer between the two runs?")[1]
    rows = [
        line
        for line in section.split("\n")
        if line.startswith("| ") and "thresholds of" not in line
    ]
    assert rows == [  # the header and separator rows are not in `rows`
        "| A | A | 0.3 | 0.7 | 0.0% (0 of 1) | 100.0% (1 of 1) | 40.0% (2 of 5) | 20.0% (1 of 5) |",
        "| B | B | -2 | 3 | 0.0% (0 of 1) | 0.0% (0 of 1) | 20.0% (1 of 5) | 20.0% (1 of 5) |",
        "| B | A | 0.3 | 0.7 | 100.0% (1 of 1) | 0.0% (0 of 1) | 60.0% (3 of 5) | 40.0% (2 of 5) |",
        "| A | B | -2 | 3 | 0.0% (0 of 1) | 0.0% (0 of 1) | 0.0% (0 of 5) | 0.0% (0 of 5) |",
    ]
    assert "search.py" in section  # it links the rule it names


def test_a_comparison_without_transfer_has_no_transfer_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _transfer_of(tmp_path, monkeypatch, None, None)
    assert "Do the confidence thresholds transfer" not in report.render_compare(result)


@pytest.mark.parametrize(
    ("artefacts", "shared"),
    [
        (None, "one GPU"),
        ({}, "one GPU"),
        ({"embedding": {"execution_providers": ["CPUExecutionProvider"]}}, "the CPU cores"),
        (
            {
                "embedding": {"execution_providers": ["CPUExecutionProvider"]},
                "rerank": {"execution_providers": ["CPUExecutionProvider"]},
            },
            "the CPU cores",
        ),
        (
            {
                "embedding": {"execution_providers": ["CPUExecutionProvider"]},
                "rerank": {
                    "execution_providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                },
            },
            "the machine's cores and accelerators",
        ),
    ],
)
def test_the_latency_note_names_what_the_workers_actually_shared(
    artefacts: Any, shared: str
) -> None:
    header: dict[str, Any] = {"execution": {"workers": 3}}
    if artefacts is not None:
        header["artefacts"] = artefacts
    text = report._latency_section({"p50": 12.0, "p95": 90.0}, header)
    assert f"3 worker(s) sharing {shared} (so under contention): p50 12, p95 90." in text


# --- gaps an independent reviewer's own mutants found (each test below killed one) ---------------


def test_a_pin_cannot_be_changed_after_it_is_read() -> None:
    pin = load_onnx_pins(REAL_CONFIGS).embedding
    with pytest.raises(dataclasses.FrozenInstanceError):
        pin.sha256 = "0" * 64  # type: ignore[misc]


@pytest.mark.parametrize("missing", ["source", "revision", "file", "sha256"])
def test_a_pin_table_missing_a_key_is_a_config_error_naming_the_key(
    tmp_path: Path, missing: str
) -> None:
    body = "".join(
        line + "\n" for line in _table("embedding").splitlines() if not line.startswith(missing)
    )
    with pytest.raises(ConfigError, match=rf"\[fastembed\.embedding\] has no '{missing}'"):
        load_onnx_pins(_models_toml(tmp_path, body + _table("rerank")))


def test_an_empty_pin_table_is_a_table_with_no_keys_not_a_missing_table(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"\[fastembed\.rerank\] has no 'source'"):
        load_onnx_pins(_models_toml(tmp_path, _table("embedding") + "[fastembed.rerank]\n"))


@pytest.mark.parametrize("value", ["3", "1.5", "true", '"x"', "[1]"])
@pytest.mark.parametrize("section", ["embedding", "rerank"])
def test_a_pin_that_is_any_non_table_value_is_refused(
    tmp_path: Path, section: str, value: str
) -> None:
    other = "rerank" if section == "embedding" else "embedding"
    text = f"[fastembed]\n{section} = {value}\n" + _table(other)  # a value where a table belongs
    with pytest.raises(ConfigError, match=rf"no \[fastembed\.{section}\] table"):
        load_onnx_pins(_models_toml(tmp_path, text))


@pytest.mark.parametrize("value", ["3", "1.5", "true", '"x"', "[1]"])
def test_a_fastembed_entry_that_is_any_non_table_value_is_refused(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(ConfigError, match=r"no \[fastembed\.embedding\] table"):
        load_onnx_pins(_models_toml(tmp_path, f"fastembed = {value}\n"))


@pytest.mark.parametrize("key", ["source", "file"])
@pytest.mark.parametrize("value", ['""', "3", "true", '["a"]'])
def test_a_source_or_file_that_is_not_a_non_empty_string_is_refused_at_load(
    tmp_path: Path, key: str, value: str
) -> None:
    body = "".join(
        f"{key} = {value}\n" if line.startswith(key) else line + "\n"
        for line in _table("embedding").splitlines()
    )
    with pytest.raises(
        ConfigError, match=rf"\[fastembed\.embedding\] {key} must be a non-empty string"
    ):
        load_onnx_pins(_models_toml(tmp_path, body + _table("rerank")))


def test_the_provider_stop_says_which_provider_the_backend_should_give() -> None:
    with pytest.raises(kb.BuildError, match=r"backend 'light' should give 'fastembed'"):
        kb.pin_models(_manifest("sentence-transformers", "fastembed"), _PINS, backend="light")
    with pytest.raises(kb.BuildError, match=r"backend 'st' should give 'sentence-transformers'"):
        kb.pin_models(_manifest("fastembed", "fastembed"), _PINS, backend="st")


def test_a_wrong_provider_is_reported_before_a_wrong_model_id() -> None:
    text = _manifest("sentence-transformers", "fastembed", model="other-model")
    with pytest.raises(kb.BuildError, match=r"\[embedding\] provider 'sentence-transformers'"):
        kb.pin_models(text, _PINS, backend="light")


def test_a_record_with_no_fields_is_listed_field_by_field_not_called_unloaded() -> None:
    found = _found()
    found["embedding"] = {}
    with pytest.raises(kb.BuildError) as raised:
        kb.verify_artefacts(found, _ONNX)
    assert "embedding source: loaded None, pinned 'a/b'" in str(raised.value)
    assert "was not loaded" not in str(raised.value)


def test_the_artefact_stop_opens_with_what_it_is_about() -> None:
    found = _found()
    found["embedding"]["sha256"] = "x"
    with pytest.raises(kb.BuildError) as raised:
        kb.verify_artefacts(found, _ONNX)
    assert str(raised.value).startswith(
        "fastembed loaded other files than configs/models.toml pins:\n  - "
    )


def test_a_light_build_closes_the_kb_when_reading_the_files_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _light_build(monkeypatch, tmp_path, {})

    def moved(handle: Any) -> Any:
        raise pinakes_api.InterfaceError("fastembed moved its internals")

    monkeypatch.setattr(pinakes_api, "loaded_artefacts", moved)
    config = config_module.load("baseline-fastembed", REAL_CONFIGS)
    with pytest.raises(pinakes_api.InterfaceError):
        kb.build_kb(make_dataset(), config, log=lambda m: None)
    assert fake.closed == [1]


def test_the_interface_error_names_which_model_could_not_be_read(tmp_path: Path) -> None:
    good = _loaded(_snapshot(tmp_path, "X--m", "m.onnx", b"abc"), repo="X/m", file="m.onnx")
    with pytest.raises(pinakes_api.InterfaceError, match=r"loaded for the embedding model"):
        pinakes_api.loaded_artefacts(_kb(SimpleNamespace(), good))
    with pytest.raises(pinakes_api.InterfaceError, match=r"loaded for the reranker"):
        pinakes_api.loaded_artefacts(_kb(good, SimpleNamespace()))


def test_the_snapshot_error_shows_the_whole_directory(tmp_path: Path) -> None:
    directory = tmp_path / "somewhere" / "else"
    directory.mkdir(parents=True)
    (directory / "m.onnx").write_bytes(b"abc")
    handle = _kb(_loaded(directory, repo="X/m", file="m.onnx"), None, r="sentence-transformers")
    with pytest.raises(
        pinakes_api.InterfaceError,
        match=re.escape(f"from {directory}, which is not a Hugging Face snapshot"),
    ):
        pinakes_api.loaded_artefacts(handle)


def test_the_inner_model_is_not_accepted_in_place_of_the_wrapper(tmp_path: Path) -> None:
    inner = _loaded(_snapshot(tmp_path, "X--m", "m.onnx", b"abc"), repo="X/m", file="m.onnx")
    handle = _kb(inner._model.model, None, r="sentence-transformers")
    with pytest.raises(
        pinakes_api.InterfaceError, match="cannot tell which files fastembed loaded"
    ):
        pinakes_api.loaded_artefacts(handle)


@pytest.mark.parametrize("library", ["torch", "onnxruntime"])
def test_a_library_that_is_installed_but_will_not_import_is_recorded_as_absent(
    monkeypatch: pytest.MonkeyPatch, library: str
) -> None:
    other = "onnxruntime" if library == "torch" else "torch"
    monkeypatch.setitem(sys.modules, other, None)
    real = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == library:
            raise ImportError(f"{library}: dlopen failed")  # not a ModuleNotFoundError
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    assert pipeline.environment()[library] is None


def test_a_resume_refuses_when_the_earlier_header_recorded_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _light_world(tmp_path, monkeypatch)
    run_dir = pipeline.run("baseline-fastembed", workers=1, log=lambda m: None)
    header = _header_of(run_dir)
    del header["artefacts"]
    (run_dir / "run.json").write_text(json.dumps(header))
    with pytest.raises(RuntimeError, match="cannot resume: model files differ"):
        pipeline.run("baseline-fastembed", workers=1, resume=run_dir, log=lambda m: None)


def test_a_refused_light_run_writes_no_header_and_can_be_retried_in_the_same_minute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = _light_world(tmp_path, monkeypatch)
    # the fake clock moves a minute per reading, so without a fixed id the retry could never collide
    monkeypatch.setattr(pipeline, "new_run_id", lambda name, provenance: "same-minute")
    good = dict(found["embedding"])
    found["embedding"]["sha256"] = "0" * 64
    with pytest.raises(kb.BuildError):
        pipeline.run("baseline-fastembed", workers=1, log=lambda m: None)
    assert list(paths.runs_dir().glob("*/run.json")) == []
    found["embedding"] = good
    run_dir = pipeline.run("baseline-fastembed", workers=1, log=lambda m: None)
    assert run_dir.name == "same-minute"
    assert _header_of(run_dir)["status"] == "complete"


def test_a_fresh_run_creates_the_runs_directory_if_it_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _light_world(tmp_path, monkeypatch)
    shutil.rmtree(paths.runs_dir())
    run_dir = pipeline.run("baseline-fastembed", workers=1, log=lambda m: None)
    assert (run_dir / "run.json").is_file()


def test_a_run_closes_the_kb_when_reading_the_files_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _light_world(tmp_path, monkeypatch)
    closed: list[int] = []
    handle = SimpleNamespace(
        manifest=SimpleNamespace(retrieval=SimpleNamespace(fusion_top_k=20)),
        backend=None,
        reranker=None,
        close=lambda: closed.append(1),
    )
    monkeypatch.setattr(pinakes_api, "open_kb", lambda root: handle)

    def moved(h: object) -> object:
        raise pinakes_api.InterfaceError("fastembed moved its internals")

    monkeypatch.setattr(pinakes_api, "loaded_artefacts", moved)
    with pytest.raises(pinakes_api.InterfaceError):
        pipeline.run("baseline-fastembed", workers=1, log=lambda m: None)
    assert closed == [1]


def test_two_fresh_runs_in_the_same_minute_do_not_share_a_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _light_world(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "new_run_id", lambda name, provenance: "fixed-id")
    pipeline.run("baseline-fastembed", workers=1, log=lambda m: None)
    with pytest.raises(FileExistsError):
        pipeline.run("baseline-fastembed", workers=1, log=lambda m: None)


def _transfer_dict(
    null: dict[str, int], a: dict[str, float], b: dict[str, float]
) -> dict[str, Any]:
    keys = (
        "a_scores_a_thresholds",
        "b_scores_b_thresholds",
        "b_scores_a_thresholds",
        "a_scores_b_thresholds",
    )
    return {
        "a_thresholds": a,
        "b_thresholds": b,
        "labels": {k: {"answerable": _counts(2, 1, 1, 0), "null": null} for k in keys},
    }


def test_the_transfer_section_links_the_rule_it_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _transfer_of(tmp_path, monkeypatch, _A_THRESHOLDS, _B_THRESHOLDS)
    assert (
        "(https://github.com/lucagattoni/pinakes/blob/main/src/pinakes/search.py)"
        in report.render_compare(result)
    )


def test_a_group_with_no_questions_is_n_a_not_a_division_by_zero() -> None:
    text = report._transfer_section(_transfer_dict(_counts(0), _A_THRESHOLDS, _B_THRESHOLDS))
    assert "| n/a (0 of 0) | n/a (0 of 0) |" in text


def test_thresholds_are_printed_to_six_significant_digits() -> None:
    fitted = {"low_below": 0.123456, "high_above": 2.345678}
    text = report._transfer_section(_transfer_dict(_counts(1, 0, 1, 0), fitted, fitted))
    assert "| 0.123456 | 2.34568 |" in text  # `:g`: six significant digits, never rounded to 3


def test_the_transfer_columns_are_headed_in_the_order_of_their_cells() -> None:
    text = report._transfer_section(
        _transfer_dict(_counts(1, 0, 1, 0), _A_THRESHOLDS, _B_THRESHOLDS)
    )
    assert (
        "| scores of | thresholds of | low below | high above | null labelled high | "
        "null labelled low | answerable labelled high | answerable labelled low |"
    ) in text


def test_the_latency_note_sees_an_accelerator_wherever_it_sits_in_the_list() -> None:
    header = {
        "execution": {"workers": 2},
        "artefacts": {
            "embedding": {
                "execution_providers": ["CPUExecutionProvider", "CoreMLExecutionProvider"]
            }
        },
    }
    text = report._latency_section({"p50": 1.0}, header)
    assert "sharing the machine's cores and accelerators" in text


def test_a_held_out_question_with_no_passage_is_unknown_not_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _toy_repo(tmp_path, monkeypatch)
    rows_a = _scored_rows(_A_SCORES, 0.5)
    rows_a[8] = _row("mhr-0008", [])  # an answerable held-out question whose search found nothing
    run_a = _finished_run("run-a", "baseline", None, rows_a)
    run_b = _finished_run("run-b", "baseline-fastembed", None, _scored_rows(_B_SCORES, 0.0))
    _calibrated(run_a, _A_THRESHOLDS)
    _calibrated(run_b, _B_THRESHOLDS)
    labels = compare.compare_runs(run_a, run_b, make_dataset())["threshold_transfer"]["labels"]
    # 0.9 high, 0.7 medium, 0.3 medium, nothing found, 0.75 high: the empty search is unknown
    assert labels["a_scores_a_thresholds"]["answerable"] == _counts(5, 2, 2, 0) | {"unknown": 1}
    assert labels["a_scores_b_thresholds"]["answerable"]["unknown"] == 1


def test_the_transfer_covers_only_the_questions_both_runs_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _toy_repo(tmp_path, monkeypatch)
    run_a = _finished_run("run-a", "baseline", None, _scored_rows(_A_SCORES, 0.5))
    rows_b = [r for r in _scored_rows(_B_SCORES, 0.0) if r["id"] != "mhr-0009"]
    run_b = _finished_run("run-b", "baseline-fastembed", None, rows_b)
    _calibrated(run_a, _A_THRESHOLDS)
    _calibrated(run_b, _B_THRESHOLDS)
    labels = compare.compare_runs(run_a, run_b, make_dataset())["threshold_transfer"]["labels"]
    assert labels["a_scores_a_thresholds"]["answerable"]["n"] == 4
    assert labels["b_scores_b_thresholds"]["answerable"]["n"] == 4


def test_release_light_tracks_release_and_conflicts_with_the_other_extras() -> None:
    meta = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    extras = meta["project"]["optional-dependencies"]
    assert extras["release-light"] == [extras["release"][0].replace("[st]", "[light]")]
    conflict = {entry["extra"] for entry in meta["tool"]["uv"]["conflicts"][0]}
    assert conflict == {"release", "release-light", "main"}


# --- compare: do two providers rank the same articles, whatever the scale of their scores --------


def _pool(scale: float, order: list[int] | None = None) -> list[list[Any]]:
    """Ten passages from articles 0..9 (or in `order`), scored on a scale of the caller's choosing."""
    return [[article, scale * (10 - rank)] for rank, article in enumerate(order or range(10))]


def _ranking_rows(
    variants: dict[str, list[int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """A's pools are articles 0..9 in order; B's differ as `variants` says, and always in score."""
    rows_a = [_row(f"mhr-{i:04d}", _pool(0.1)) for i in range(10)]
    rows_b = [_row(f"mhr-{i:04d}", _pool(1.0, variants.get(f"mhr-{i:04d}"))) for i in range(10)]
    return rows_a, rows_b


def _ranking_compare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variants: dict[str, list[int]]
) -> dict[str, Any]:
    _toy_repo(tmp_path, monkeypatch)
    rows_a, rows_b = _ranking_rows(variants)
    run_a = _finished_run("run-a", "baseline", None, rows_a)
    run_b = _finished_run("run-b", "baseline-fastembed", None, rows_b)
    return compare.compare_runs(run_a, run_b, make_dataset())


def test_the_ranking_is_compared_without_the_scores_and_in_two_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    swapped_tail = [0, 1, 2, 3, 4, 5, 6, 7, 9, 8]  # the last two exchanged: outside the first 8
    swapped_head = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]  # the first two exchanged: inside it
    swapped_edge = [0, 1, 2, 3, 4, 5, 6, 8, 7, 9]  # the eighth and ninth: the eighth is inside
    identical = _ranking_compare(
        tmp_path,
        monkeypatch,
        {
            "mhr-0007": swapped_tail,
            "mhr-0008": swapped_head,
            "mhr-0004": swapped_edge,
            "mhr-0003": list(range(9)),
        },
    )["identical_searches"]
    assert identical["of"] == 10
    assert identical["same_passages_and_scores"] == 0  # the scales differ in every search
    # whole pool: all but 0003 (nine passages, not ten), 0004, 0007 and 0008 -> 6
    assert identical["same_articles_in_same_order"] == 6
    # first 8 passages: 0004 (its eighth passage moved) and 0008 differ there; 0003 and 0007
    # differ only beyond the eighth -> 8
    assert identical["same_articles_in_first_8_passages"] == 8


def test_identical_rankings_on_different_scales_count_as_the_same_ranking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identical = _ranking_compare(tmp_path, monkeypatch, {})["identical_searches"]
    assert identical["same_passages_and_scores"] == 0
    assert identical["same_articles_in_same_order"] == 10
    assert identical["same_articles_in_first_8_passages"] == 10


def test_the_comparison_says_how_many_searches_ranked_the_same_articles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _ranking_compare(
        tmp_path,
        monkeypatch,
        {"mhr-0008": [1, 0, 2, 3, 4, 5, 6, 7, 8, 9], "mhr-0003": list(range(9))},
    )
    assert (
        "the same articles in the same order: 8 of 10 over the whole pool, "
        "9 of 10 over the first 8 passages." in report.render_compare(result)
    )


def test_a_comparison_written_before_the_ranking_was_counted_still_renders() -> None:
    from test_round2_fast import _comparison

    result = _comparison()
    assert "same_articles_in_same_order" not in result["identical_searches"]
    assert "Ignoring the scores" not in report.render_compare(result)
