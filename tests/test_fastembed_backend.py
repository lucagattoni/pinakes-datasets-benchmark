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

import hashlib
import json
import platform
import shutil
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from test_round2_fast import _fake_stack, _header_of, _resume_world
from toy import make_dataset

from pnkbench import config as config_module
from pnkbench import kb, paths, pinakes_api, pipeline, runner
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
    from test_round2_fast import _fake_pinakes

    _fake_pinakes(monkeypatch, tmp_path)
    opened: list[object] = []
    monkeypatch.setattr(pinakes_api, "open_kb", lambda root: opened.append(root))
    report = kb.build_kb(
        make_dataset(), config_module.Config("probe", "", "st", "off"), log=lambda m: None
    )
    assert opened == []
    assert "model_files" not in report
