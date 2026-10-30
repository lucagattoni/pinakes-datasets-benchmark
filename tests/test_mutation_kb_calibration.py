"""Tests written against the mutants of `kb.py` and `calibration.py` that an independent review found
still surviving the suite.

The calibration rule is Pinakes' own (`pinakes/calibrate.py`: `low_below` is the median of the
unanswerable top-1 scores, `high_above` their 0.9 percentile taken at index
`round(0.9 * (n - 1))` with Python's round-half-to-even), so the expected values are worked out
from that rule by hand. The fit/eval split is restated here with `hashlib` from the rule the module
prints, `sha256(salt:id)[0] even -> fit`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from toy import make_articles, make_dataset

from pnkbench import calibration, items, kb, layout, paths, pinakes_api, runner
from pnkbench.config import Config, ModelPin, Models
from pnkbench.multihop_rag import Dataset, Evidence, Query

MANIFEST = """[kb]
name = "probe"

[chunking]
strategy = "structural"
overlap = 64

[retrieval]
rerank = "local"

[rerank]
model = "BAAI/bge-reranker-base"

[budget]
timezone = "UTC"
"""

_REFUSED = (kb.BuildError, tomllib.TOMLDecodeError)


# --- kb.set_key ---------------------------------------------------------------------------


@pytest.mark.parametrize("literal", ['"unterminated', "[1, 2", "1 = 1"])
def test_a_literal_that_breaks_the_manifest_is_refused_not_returned(literal: str) -> None:
    with pytest.raises(_REFUSED):
        kb.set_key(MANIFEST, "chunking", "metadata", literal)


def test_replacing_a_key_whose_value_spans_several_lines_is_refused() -> None:
    # only the first line of the old value would be removed, leaving the rest as orphans
    text = MANIFEST.replace('strategy = "structural"', 'metadata = [\n  "a",\n  "b",\n]')
    with pytest.raises(_REFUSED):
        kb.set_key(text, "chunking", "metadata", '"prefix"')


def test_the_last_table_of_a_manifest_is_edited_through_its_final_line() -> None:
    out = kb.set_key(MANIFEST, "budget", "timezone", '"Europe/Rome"')
    assert tomllib.loads(out)["budget"] == {"timezone": "Europe/Rome"}
    assert out.count("timezone") == 1
    unterminated = kb.set_key(MANIFEST.rstrip("\n"), "budget", "timezone", '"Europe/Rome"')
    assert tomllib.loads(unterminated)["budget"] == {"timezone": "Europe/Rome"}


def test_a_table_is_found_by_its_exact_header_not_by_a_longer_name_that_starts_the_same() -> None:
    text = "[chunking-extra]\nfoo = 1\n\n" + MANIFEST
    parsed = tomllib.loads(kb.set_chunking_metadata(text, "prefix"))
    assert parsed["chunking"]["metadata"] == "prefix"
    assert parsed["chunking-extra"] == {"foo": 1}


@pytest.mark.parametrize("value", ["prefix", "Prefix", 'a "quoted" word', "tab\there", "naïve"])
def test_the_chunking_metadata_value_is_written_exactly_as_given(value: str) -> None:
    parsed = tomllib.loads(kb.set_chunking_metadata(MANIFEST, value))
    assert parsed["chunking"]["metadata"] == value


# --- kb.set_confidence_thresholds ---------------------------------------------------------


def test_replacing_thresholds_drops_the_whole_old_block_even_across_a_blank_line() -> None:
    old = '[retrieval.confidence]\nfitted_for = "old"\n\nlow_below = 1.0\nhigh_above = 2.0\n\n[rerank]'
    text = MANIFEST.replace("[rerank]", old)
    out = kb.set_confidence_thresholds(text, fitted_for="new", low_below=-1.0, high_above=0.5)
    parsed = tomllib.loads(out)
    # the old keys after the blank line must not be left behind in [retrieval]
    assert parsed["retrieval"] == {
        "rerank": "local",
        "confidence": {"fitted_for": "new", "low_below": -1.0, "high_above": 0.5},
    }
    assert out.count("low_below") == 1 and out.count("high_above") == 1
    assert parsed["rerank"] == {"model": "BAAI/bge-reranker-base"}


def test_thresholds_are_written_so_that_they_read_back_to_the_same_floats() -> None:
    out = kb.set_confidence_thresholds(
        MANIFEST,
        fitted_for="BAAI/bge-reranker-base@abc123",
        low_below=-4.123456789012345,
        high_above=0.1 + 0.2,
    )
    assert tomllib.loads(out)["retrieval"]["confidence"] == {
        "fitted_for": "BAAI/bge-reranker-base@abc123",
        "low_below": -4.123456789012345,
        "high_above": 0.30000000000000004,
    }
    assert out.rstrip("\n").splitlines()[-3].startswith('fitted_for = "BAAI')  # appended last


def test_thresholds_that_cannot_be_read_back_are_refused() -> None:
    with pytest.raises(_REFUSED):  # a quote inside the string breaks the TOML
        kb.set_confidence_thresholds(MANIFEST, fitted_for='m"@r', low_below=-1.0, high_above=1.0)
    with pytest.raises(kb.BuildError, match="did not take"):  # nan is valid TOML but never equal
        kb.set_confidence_thresholds(
            MANIFEST, fitted_for="m@r", low_below=float("nan"), high_above=1.0
        )


# --- calibration: the thresholds ----------------------------------------------------------


def test_the_low_threshold_is_the_median_not_the_mean() -> None:
    # [1, 2, 10]: median 2 (the mean would be 4.33); 0.9 rank round(0.9 * 2) = 2, so 10
    assert calibration.expected_thresholds([10.0, 1.0, 2.0]) == (2.0, 10.0)
    # an even count: the median is the mean of the two middle values; round(0.9 * 3) = 3
    assert calibration.expected_thresholds([1.0, 2.0, 3.0, 100.0]) == (2.5, 100.0)


def test_the_high_threshold_is_the_ninetieth_percentile_not_the_ninety_fifth() -> None:
    values = [float(v) for v in range(21)]  # 0..20: median 10; round(0.9 * 20) = 18, not 19
    assert calibration.expected_thresholds(values) == (10.0, 18.0)


def test_the_percentile_rank_is_round_half_to_even_of_fraction_times_n_minus_one() -> None:
    values = [6.0, 1.0, 5.0, 2.0, 4.0, 3.0]  # sorted 1..6, so n - 1 = 5
    assert calibration.pinakes_percentile(values, 0.0) == 1.0
    assert calibration.pinakes_percentile(values, 0.5) == 3.0  # 2.5 rounds to 2: the third value
    assert calibration.pinakes_percentile(values, 0.7) == 5.0  # 3.5 rounds to 4: the fifth value
    assert calibration.pinakes_percentile(values, 0.9) == 5.0  # 4.5 rounds to 4: the fifth value
    assert calibration.pinakes_percentile(values, 1.0) == 6.0
    assert calibration.pinakes_percentile([7.0], 0.9) == 7.0


def test_the_thresholds_of_a_cached_run_are_read_from_its_calibration() -> None:
    assert calibration.thresholds_from(None) is None
    assert calibration.thresholds_from({"thresholds": {"low_below": -1.0, "high_above": 2.0}}) == {
        "low_below": -1.0,
        "high_above": 2.0,
    }


# --- calibration: reading what Pinakes printed --------------------------------------------


def test_parse_block_reads_a_last_line_with_no_newline_after_it() -> None:
    printed = (
        'fitted on 3 answerable\n[retrieval.confidence]\nfitted_for = "m@abc"\n'
        "low_below  = -4.1250\nhigh_above = 2.7512"
    )
    fit = calibration.parse_block(printed)
    assert (fit.fitted_for, fit.low_below, fit.high_above) == ("m@abc", -4.125, 2.7512)
    assert fit.output == printed


def test_parse_block_reads_a_fitted_for_that_is_the_last_line() -> None:
    printed = 'low_below  = -1.5000\nhigh_above = 0.2500\nfitted_for = "BAAI/bge-reranker-base@abc"'
    fit = calibration.parse_block(printed)
    assert fit.fitted_for == "BAAI/bge-reranker-base@abc"
    assert (fit.low_below, fit.high_above) == (-1.5, 0.25)


def test_parse_block_names_what_is_missing_and_quotes_only_the_last_600_characters() -> None:
    printed = "x" * 700 + "THE-END"
    with pytest.raises(calibration.CalibrationError) as caught:
        calibration.parse_block(printed)
    message = str(caught.value)
    assert "printed no ['fitted_for', 'low_below', 'high_above']" in message
    quoted = message.split(":\n", 1)[1]
    assert quoted == printed[-600:] and quoted.endswith("THE-END")


# --- calibration: which questions go to Pinakes -------------------------------------------


def _dataset_of(answerable: int, null: int) -> Dataset:
    articles = make_articles()
    queries = []
    for n in range(answerable + null):
        if n < answerable:
            pair = (n % 11, n % 11 + 1)
            evidence = tuple(Evidence(articles[i].title, articles[i].url, "a fact") for i in pair)
            queries.append(Query(f"mhr-{n:04d}", n, f"q{n}", "x", "comparison_query", evidence))
        else:
            queries.append(
                Query(f"mhr-{n:04d}", n, f"q{n}", "Insufficient information.", "null_query", ())
            )
    return Dataset(articles=tuple(articles), queries=tuple(queries), sha256={})


def _in_fit_half(base_id: str) -> bool:
    return hashlib.sha256(f"pnkbench-split-v1:{base_id}".encode()).digest()[0] % 2 == 0


def _sample_order(base_id: str) -> str:
    return hashlib.sha256(f"pnkbench-split-v1:order:{base_id}".encode()).hexdigest()


def test_fit_items_take_every_fit_null_and_the_150_first_answerable_in_hash_order() -> None:
    dataset = _dataset_of(answerable=400, null=60)
    chosen = calibration.fit_items(items.derive_items(dataset), dataset)

    fit_answerable = [q.id for q in dataset.queries if q.answerable and _in_fit_half(q.id)]
    fit_null = [q.id for q in dataset.queries if not q.answerable and _in_fit_half(q.id)]
    assert len(fit_answerable) > 150, "the fixture must exceed the cap for the cap to matter"
    assert fit_null, "the fixture must contain fit-half null questions"

    answerable_ids = {q.id for q in dataset.queries if q.answerable}
    chosen_answerable = sorted(i.id for i in chosen if i.id in answerable_ids)
    expected = sorted(sorted(fit_answerable, key=_sample_order)[:150])
    assert len(chosen_answerable) == 150
    assert chosen_answerable == expected
    assert sorted(i.id for i in chosen if i.id not in answerable_ids) == sorted(fit_null)
    assert all(i.variant == items.BASE for i in chosen)


# --- kb.set_key: where a key lands ---------------------------------------------------------


def test_a_key_set_in_an_empty_table_stays_in_that_table() -> None:
    text = '[chunking]\n[retrieval]\nrerank = "local"\n'
    parsed = tomllib.loads(kb.set_key(text, "chunking", "metadata", '"prefix"'))
    assert parsed == {"chunking": {"metadata": "prefix"}, "retrieval": {"rerank": "local"}}


def test_a_new_key_goes_after_the_last_key_and_before_the_comments_and_blank_that_follow() -> None:
    # placement is cosmetic to TOML; the tidy place is directly under the table's last key
    text = (
        '[chunking]\nstrategy = "x"\n# a note about the next table\n\n[retrieval]\nrerank = "l"\n'
    )
    out = kb.set_key(text, "chunking", "metadata", '"prefix"')
    assert out == (
        '[chunking]\nstrategy = "x"\nmetadata = "prefix"\n# a note about the next table\n\n'
        '[retrieval]\nrerank = "l"\n'
    )


def test_lines_before_the_first_table_survive_a_threshold_edit() -> None:
    text = "# generated by pnk init\nschema = 1\n\n" + MANIFEST
    out = kb.set_confidence_thresholds(text, fitted_for="m@r", low_below=-1.0, high_above=1.0)
    assert out.startswith("# generated by pnk init\nschema = 1\n\n[kb]\n")
    assert tomllib.loads(out)["schema"] == 1


# --- kb.build_kb: what is written, and what is refused -------------------------------------

_BUILD_MANIFEST = """[kb]
name = "probe"

[embedding]
model = "e-model"

[chunking]
strategy = "structural"

[retrieval]
rerank = "local"

[rerank]
model = "r-model"
"""
_PINS = Models(embedding=ModelPin("e-model", "5" * 40), rerank=ModelPin("r-model", "2" * 40))


class _Stop(Exception):
    pass


def _fake_pinakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, list[Any]]:
    """Replace everything that needs Pinakes with recorders; the repository root is `tmp_path`."""
    seen: dict[str, list[Any]] = {"pnk": [], "sidecars": []}

    def run_pnk(args: list[str], *, log: Path | None = None) -> str:
        seen["pnk"].append((list(args), log))
        if args[0] == "init":
            Path(args[1]).mkdir(parents=True, exist_ok=True)
            (Path(args[1]) / "pinakes.toml").write_text(_BUILD_MANIFEST, encoding="utf-8")
        return ""

    def mint_sidecar(document: Path, *, title: str, tags: Any, source_url: str) -> None:
        seen["sidecars"].append((document, title, tuple(tags), source_url))

    monkeypatch.setenv(paths.ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(pinakes_api, "require_interfaces", lambda: None)
    monkeypatch.setattr(pinakes_api, "run_pnk", run_pnk)
    monkeypatch.setattr(pinakes_api, "mint_sidecar", mint_sidecar)
    monkeypatch.setattr(kb, "load_models", lambda: _PINS)
    monkeypatch.setattr(
        kb, "verify_kb", lambda dataset, root: {"documents": len(dataset.articles), "chunks": 7}
    )
    return seen


def test_build_writes_every_article_byte_for_byte_with_its_publication_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _fake_pinakes(monkeypatch, tmp_path)
    dataset = make_dataset()
    odd = replace(dataset.articles[2], body="Café line one\r\nline two\n\nlast")
    # article 3 shares article 0's folder (same category and source), as most real ones do
    twin = replace(
        dataset.articles[3],
        category=dataset.articles[0].category,
        source=dataset.articles[0].source,
    )
    dataset = replace(dataset, articles=(*dataset.articles[:2], odd, twin, *dataset.articles[4:]))
    folder = layout.doc_relpath(dataset.articles[0]).rsplit("/", 1)[0]
    assert layout.doc_relpath(twin).rsplit("/", 1)[0] == folder
    config = Config("probe", "a test", "st", "prefix")
    root = paths.kb_dir("probe")
    report = kb.build_kb(dataset, config, log=lambda message: None)

    for article in dataset.articles:
        document = root / layout.doc_relpath(article)
        assert document.read_bytes() == article.body.encode("utf-8")
        assert document.stat().st_mtime == article.published_at.timestamp()
    assert [(t, tags, url) for _, t, tags, url in seen["sidecars"]] == [
        (a.title, layout.tags_for(a), a.url) for a in dataset.articles
    ]
    assert [d for d, *_ in seen["sidecars"]] == [
        root / layout.doc_relpath(a) for a in dataset.articles
    ]
    # init, then sync, with Pinakes' own flags
    assert seen["pnk"][0][0] == ["init", str(root), "--backend", "st", "--name", "mhr-probe"]
    assert seen["pnk"][1] == (
        ["sync", "--kb", str(root), "--offline"],
        root.parent / "probe.sync.log",
    )

    manifest = tomllib.loads((root / "pinakes.toml").read_text(encoding="utf-8"))
    assert manifest["embedding"]["revision"] == "5" * 40
    assert manifest["rerank"]["revision"] == "2" * 40
    assert manifest["chunking"]["metadata"] == "prefix"

    digest = hashlib.sha256((root / "pinakes.toml").read_bytes()).hexdigest()
    assert report["manifest_sha256"] == digest
    assert (report["documents"], report["chunks"], report["config"]) == (12, 7, "probe")
    assert report["root"] == str(root) and report["sync_seconds"] >= 0
    text = (root.parent / "probe.build.json").read_text(encoding="utf-8")
    assert json.loads(text) == report
    assert text.endswith("}\n") and '\n  "documents": 12,' in text  # indented two spaces


def test_build_leaves_the_manifest_without_a_metadata_key_when_it_is_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_pinakes(monkeypatch, tmp_path)
    kb.build_kb(make_dataset(), Config("probe", "", "st", "off"), log=lambda message: None)
    manifest = tomllib.loads((paths.kb_dir("probe") / "pinakes.toml").read_text(encoding="utf-8"))
    assert "metadata" not in manifest["chunking"]


def test_build_removes_an_earlier_build_but_refuses_a_root_outside_the_build_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _fake_pinakes(monkeypatch, tmp_path)
    config = Config("probe", "", "st", "off")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("precious")
    with pytest.raises(kb.BuildError, match="refusing to build outside"):
        kb.build_kb(make_dataset(), config, root=elsewhere, log=lambda message: None)
    assert (elsewhere / "keep.txt").read_text() == "precious" and seen["pnk"] == []

    # inside the build directory an earlier build is wiped before the new one starts
    root = paths.kb_dir("probe")
    root.mkdir(parents=True)
    (root / "stale.txt").write_text("old")
    messages: list[str] = []
    kb.build_kb(make_dataset(), config, log=messages.append)
    assert not (root / "stale.txt").exists()
    assert messages[0] == f"removing the previous build at {root}"


def test_build_stops_when_two_articles_would_share_a_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_pinakes(monkeypatch, tmp_path)
    dataset = make_dataset()
    twin = replace(dataset.articles[1], index=99)
    dataset = replace(dataset, articles=(*dataset.articles, twin))
    with pytest.raises(kb.BuildError, match="two articles map to one path"):
        kb.build_kb(dataset, Config("probe", "", "st", "off"), log=lambda message: None)


def test_under_means_inside_the_directory_once_paths_are_resolved(tmp_path: Path) -> None:
    build = tmp_path / "build"
    build.mkdir()
    assert kb._under(build, build / "kb" / "baseline")
    assert not kb._under(build, tmp_path / "build-evil" / "kb")  # a longer name, not a child
    assert not kb._under(build, build / ".." / "elsewhere")


# --- kb.verify_kb: the check that the index is the dataset -------------------------------------


def _index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **changes: Any) -> Path:
    """A SQLite index holding the toy articles as Pinakes would. `changes` maps a document path
    (or '*' for all) to column overrides, 'failures' / 'chunks' to counts, 'drop' to paths."""
    dataset = make_dataset()
    database = tmp_path / "index.db"
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE documents (path, title, mtime, metadata, state, source_type)")
    connection.execute("CREATE TABLE chunks (id)")
    connection.execute("CREATE TABLE failures (id)")
    for article in dataset.articles:
        path = layout.doc_relpath(article)
        if path in changes.get("drop", ()):
            continue
        row = {
            "title": article.title,
            "mtime": article.published_at.timestamp(),
            "metadata": json.dumps({"tags": list(layout.tags_for(article))}),
            "state": "active",
            "source_type": "text",
        }
        row.update(changes.get("*", {}))
        row.update(changes.get(path, {}))
        connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
            (path, row["title"], row["mtime"], row["metadata"], row["state"], row["source_type"]),
        )
    for extra in changes.get("extra", ()):
        connection.execute(
            "INSERT INTO documents VALUES (?, 't', 0, '{}', 'active', 'text')", (extra,)
        )
    connection.executemany("INSERT INTO chunks VALUES (?)", [(n,) for n in range(30)])
    connection.executemany(
        "INSERT INTO failures VALUES (?)", [(n,) for n in range(changes.get("failures", 0))]
    )
    connection.commit()
    connection.close()
    monkeypatch.setattr(pinakes_api, "index_path", lambda root: database)
    return database


def test_a_faithful_index_verifies_and_reports_its_documents_and_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _index(tmp_path, monkeypatch)
    assert kb.verify_kb(make_dataset(), tmp_path) == {"documents": 12, "chunks": 30}


def test_a_failure_row_is_a_problem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _index(tmp_path, monkeypatch, failures=2)
    with pytest.raises(kb.BuildError, match="2 rows in the failures table"):
        kb.verify_kb(make_dataset(), tmp_path)


def _path_of(n: int) -> str:
    return layout.doc_relpath(make_dataset().articles[n])


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"state": "stale"}, ": state stale"),
        ({"source_type": "pdf"}, ": source_type pdf"),
        ({"title": "Another title"}, ": title 'Another title'"),
        (
            {"metadata": json.dumps({"tags": ["technology", "source-the-verge"]})},
            ": tags ('technology', 'source-the-verge')",
        ),
        ({"metadata": json.dumps({})}, ": tags ()"),
        ({"mtime": 1696118400.0 + 34200 + 1.5}, ": mtime "),
    ],
)
def test_each_way_an_indexed_document_can_differ_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: dict[str, Any], fragment: str
) -> None:
    _index(tmp_path, monkeypatch, **{_path_of(0): change})
    with pytest.raises(kb.BuildError) as caught:
        kb.verify_kb(make_dataset(), tmp_path)
    message = str(caught.value)
    assert message.startswith("the built KB is not the dataset:\n  - ")
    assert f"{_path_of(0)}{fragment}" in message


def test_a_modification_time_may_differ_by_one_second_but_no_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stamp = make_dataset().articles[0].published_at.timestamp()
    for offset in (1.0, -1.0, 0.5):
        _index(tmp_path, monkeypatch, **{_path_of(0): {"mtime": stamp + offset}})
        assert kb.verify_kb(make_dataset(), tmp_path)["documents"] == 12
    _index(tmp_path, monkeypatch, **{_path_of(0): {"mtime": stamp - 1.5}})
    with pytest.raises(kb.BuildError, match="mtime"):
        kb.verify_kb(make_dataset(), tmp_path)


def test_a_missing_and_an_unexpected_document_are_each_listed_three_at_most(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dropped = [_path_of(n) for n in range(5)]
    extras = [f"docs/x/{n}.txt" for n in range(5)]
    _index(tmp_path, monkeypatch, drop=dropped, extra=extras)
    with pytest.raises(kb.BuildError) as caught:
        kb.verify_kb(make_dataset(), tmp_path)
    message = str(caught.value)
    assert f"missing {sorted(dropped)[:3]}" in message
    assert f"unexpected {sorted(extras)[:3]}" in message


def test_only_the_first_ten_problems_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _index(tmp_path, monkeypatch, **{"*": {"title": "wrong"}})  # all twelve titles are wrong
    with pytest.raises(kb.BuildError) as caught:
        kb.verify_kb(make_dataset(), tmp_path)
    assert str(caught.value).count("\n  - ") == 10


# --- calibration: the run against Pinakes' tool ---------------------------------------------


def test_pinakes_calibrate_is_run_with_this_interpreter_and_its_failure_is_reported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    printed = (
        '[retrieval.confidence]\nfitted_for = "m@r"\nlow_below  = -1.5000\nhigh_above = 0.2500\n'
    )
    calls: list[tuple[list[str], dict[str, Any]]] = []
    outcomes = [
        subprocess.CompletedProcess([], 0, stdout=printed, stderr=""),
        subprocess.CompletedProcess([], 3, stdout="", stderr="x" * 900 + "TAIL"),
    ]

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return outcomes[len(calls) - 1]

    monkeypatch.setattr(calibration.subprocess, "run", fake_run)
    questions = tmp_path / "q.yaml"
    fit = calibration.run_pinakes_calibrate(tmp_path, questions)
    assert (fit.fitted_for, fit.low_below, fit.high_above) == ("m@r", -1.5, 0.25)
    assert calls[0][0] == [
        sys.executable,
        "-m",
        "pinakes.calibrate",
        str(tmp_path),
        "--questions",
        str(questions),
    ]
    assert calls[0][1] == {"capture_output": True, "text": True, "check": False}
    with pytest.raises(calibration.CalibrationError) as caught:
        calibration.run_pinakes_calibrate(tmp_path, questions)
    message = str(caught.value)
    assert message.startswith("pinakes.calibrate exited 3:\n") and message.endswith("xxxxTAIL")
    assert len(message.split("\n", 1)[1]) == 800  # only the last 800 characters of stderr


def _id(n: int) -> str:
    return f"mhr-{n:04d}"


def _calibration_world(tmp_path: Path):
    """40 answerable and 40 null queries, a stored top-1 score for each, and a KB directory."""
    articles = make_articles()
    queries = []
    for n in range(80):
        if n < 40:
            evidence = tuple(
                Evidence(articles[i].title, articles[i].url, "fact") for i in (n % 11, n % 11 + 1)
            )
            queries.append(Query(_id(n), n, f"q{n}", "x", "comparison_query", evidence))
        else:
            queries.append(Query(_id(n), n, f"q{n}", "none", "null_query", ()))
    dataset = Dataset(articles=tuple(articles), queries=tuple(queries), sha256={})
    derived = items.derive_items(dataset)
    rows = []
    for item in derived:
        if item.variant != items.BASE:
            continue
        n = int(item.id[4:])
        passages = [[0, 2.0]] if n < 40 else [[0, -9.0 + 0.173 * (n - 40)]]
        rows.append(
            {
                "id": item.id,
                "base_id": item.id,
                "variant": "base",
                "conf": "unknown",
                "considered": 20,
                "p": passages,
                "ms": 1,
            }
        )
    fit_nulls = [q.id for q in queries if not q.answerable and _in_fit_half(q.id)]
    # one fit-half null question returned nothing: it has no score and takes no part
    silent = fit_nulls[0]
    for row in rows:
        if row["id"] == silent:
            row["p"] = []
    kb_root = tmp_path / "kb"
    kb_root.mkdir()
    (kb_root / "pinakes.toml").write_text(_BUILD_MANIFEST, encoding="utf-8")
    scores = sorted(r["p"][0][1] for r in rows if r["id"] in fit_nulls and r["id"] != silent)
    return dataset, derived, rows, kb_root, fit_nulls, scores


def _calibrate(monkeypatch, tmp_path, fit, *, parity=None):
    dataset, derived, rows, kb_root, fit_nulls, scores = _calibration_world(tmp_path)
    monkeypatch.setattr(calibration, "run_pinakes_calibrate", lambda root, questions: fit(scores))
    monkeypatch.setattr(
        calibration,
        "parity_check",
        lambda *a, **k: parity or {"label_mismatches": 0, "checked": 3, "mismatch_ids": []},
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    result = calibration.calibrate(
        kb_root, run_dir, derived, dataset, rows, {"docs/a.txt": 0}, limit=20
    )
    return result, run_dir, kb_root, fit_nulls, scores


def _printed(scores: list[float], low_error: float = 0.0, high_error: float = 0.0):
    low, high = calibration.expected_thresholds(scores)
    return calibration.Fit("m@r", low + low_error, high + high_error, "the tool's output")


def test_calibrate_writes_the_thresholds_into_the_manifest_and_reports_the_split_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, run_dir, kb_root, fit_nulls, scores = _calibrate(
        monkeypatch, tmp_path, lambda s: _printed(s)
    )
    low, high = calibration.expected_thresholds(scores)
    assert len(scores) == len(fit_nulls) - 1  # the silent question has no score
    cross = result["cross_check_against_stored_scores"]
    assert cross == {
        "null_scores": len(scores),
        "pinakes_low_below": low,
        "rederived_low_below": low,
        "pinakes_high_above": high,
        "rederived_high_above": high,
        "agrees": True,
    }
    block = tomllib.loads((kb_root / "pinakes.toml").read_text(encoding="utf-8"))
    assert block["retrieval"]["confidence"] == {
        "fitted_for": "m@r",
        "low_below": low,
        "high_above": high,
    }
    assert result["thresholds"] == {"fitted_for": "m@r", "low_below": low, "high_above": high}
    assert result["pinakes_calibrate_output"] == "the tool's output"
    assert result["parity_with_pinakes_labels"]["label_mismatches"] == 0
    # 80 queries split by the fixed hash; every fit-half answerable question is below the cap
    fit_answerable = [n for n in range(40) if _in_fit_half(_id(n))]
    eval_answerable = [n for n in range(40) if not _in_fit_half(_id(n))]
    eval_null = [n for n in range(40, 80) if not _in_fit_half(_id(n))]
    assert result["split"] == {
        "salt": "pnkbench-split-v1",
        "rule": "sha256(salt:id)[0] even -> fit",
        "fit_null": len(fit_nulls),
        "fit_answerable_used": len(fit_answerable),
        "eval_null": len(eval_null),
        "eval_answerable": len(eval_answerable),
    }
    # the questions handed to Pinakes are written beside the run
    assert (run_dir / "calibration-questions.yaml").read_text(encoding="utf-8").startswith("#")


@pytest.mark.parametrize(
    ("low_error", "high_error", "agrees"),
    [
        (5.9e-5, 0.0, True),  # Pinakes prints four decimals: a rounding error is not a mismatch
        (-5.9e-5, 5.9e-5, True),
        (6.1e-5, 0.0, False),
        (-6.1e-5, 0.0, False),
        (0.0, 6.1e-5, False),
        (0.0, -6.1e-5, False),
    ],
)
def test_the_printed_thresholds_must_agree_with_the_stored_scores_to_four_decimals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    low_error: float,
    high_error: float,
    agrees: bool,
) -> None:
    def fit(scores: list[float]) -> calibration.Fit:
        return _printed(scores, low_error, high_error)

    if agrees:
        result, *_ = _calibrate(monkeypatch, tmp_path, fit)
        assert result["cross_check_against_stored_scores"]["agrees"] is True
    else:
        with pytest.raises(calibration.CalibrationError, match="do not match the stored scores"):
            _calibrate(monkeypatch, tmp_path, fit)


def test_a_disagreement_between_pinakes_labels_and_ours_stops_the_calibration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parity = {"label_mismatches": 2, "checked": 10, "mismatch_ids": ["mhr-0001", "mhr-0002"]}
    with pytest.raises(calibration.CalibrationError) as caught:
        _calibrate(monkeypatch, tmp_path, lambda s: _printed(s), parity=parity)
    assert "differed from ours on 2 of 10 repeated searches" in str(caught.value)
    assert "['mhr-0001', 'mhr-0002']" in str(caught.value)


# --- calibration.parity_check ---------------------------------------------------------------


class _Handle:
    closed = False

    def close(self) -> None:
        self.closed = True


def _parity(
    monkeypatch: pytest.MonkeyPatch,
    base_ids: list[str],
    again: dict[str, dict[str, Any]],
    recorded: dict[str, list[list[Any]]],
    fit: calibration.Fit,
) -> tuple[dict[str, Any], list[str], _Handle]:
    handle = _Handle()
    visited: list[str] = []
    monkeypatch.setattr(pinakes_api, "open_kb", lambda root: handle)

    def run_one(kb_handle: Any, index: Any, payload: dict[str, Any], limit: int) -> dict[str, Any]:
        visited.append(payload["id"])
        return again[payload["id"]]

    monkeypatch.setattr(runner, "run_one", run_one)
    run_items = [items.RunItem(i, i, items.BASE, "q", items.FilterSpec(), 1) for i in base_ids]
    rows_by_id = {i: {"p": recorded[i]} for i in base_ids}
    result = calibration.parity_check(Path("."), run_items, {}, rows_by_id, fit, limit=20)
    return result, visited, handle


def test_parity_searches_every_base_item_of_a_small_run_and_closes_the_kb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [_id(n) for n in range(10)]
    fit = calibration.Fit("m@r", -3.0, 1.0, "")
    # the recorded top-1 score and the repeated one differ by 0.25, 0.5 and 0.125; two labels differ
    again = {i: {"p": [[0, -5.0]], "conf": "low"} for i in ids}
    recorded = {i: [[0, -5.0]] for i in ids}
    recorded[ids[1]], again[ids[1]]["p"] = [[0, -5.0]], [[0, -4.75]]
    recorded[ids[2]], again[ids[2]]["p"] = [[0, -5.0]], [[0, -5.5]]
    recorded[ids[3]], again[ids[3]]["p"] = [[0, -5.0]], [[0, -4.875]]
    again[ids[4]]["conf"] = "high"
    again[ids[5]]["conf"] = "medium"
    result, visited, handle = _parity(monkeypatch, ids, again, recorded, fit)
    assert visited == ids and handle.closed
    assert result == {
        "checked": 10,
        "label_mismatches": 2,
        "mismatch_ids": [ids[4], ids[5]],
        "max_abs_top1_difference_vs_recorded": 0.5,
    }


def test_parity_takes_every_second_item_of_a_run_of_1300_and_stops_at_600(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [_id(n) for n in range(1300)]
    fit = calibration.Fit("m@r", -3.0, 1.0, "")
    again = {i: {"p": [[0, -5.0]], "conf": "low"} for i in ids}
    recorded = {i: [[0, -5.0]] for i in ids}
    result, visited, _ = _parity(monkeypatch, ids, again, recorded, fit)
    assert result["checked"] == 600
    assert visited == [_id(n) for n in range(0, 1200, 2)]


def test_parity_ignores_a_missing_score_when_measuring_the_difference_and_names_five_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [_id(n) for n in range(8)]
    fit = calibration.Fit("m@r", -3.0, 1.0, "")
    again = {i: {"p": [[0, -5.0]], "conf": "high"} for i in ids}  # every label disagrees
    recorded = {i: [[0, -5.0]] for i in ids}
    again[ids[0]]["p"] = []  # nothing came back this time: no score, label "unknown" differs too
    recorded[ids[1]] = []  # nothing was recorded
    result, _, _ = _parity(monkeypatch, ids, again, recorded, fit)
    assert result["label_mismatches"] == 8
    assert result["mismatch_ids"] == ids[:5]
    assert result["max_abs_top1_difference_vs_recorded"] == 0.0
