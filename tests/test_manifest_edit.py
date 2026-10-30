from __future__ import annotations

import tomllib

import pytest

from pnkbench import kb
from pnkbench.config import ModelPin, Models

# The manifest `pnk init --backend st` stamps (Pinakes 0.32.8), trimmed to what the edits touch.
STAMPED = """[kb]
name     = "probe"
id       = "01M66YWJJSA29YVVZ18878NN7J"
template = "notes@1.2"

[sources]
roots   = ["docs/"]
include = ["**/*.md", "**/*.txt"]

[embedding]
provider = "sentence-transformers"
model    = "BAAI/bge-small-en-v1.5"
dim      = 384
# revision = "…"   # pin the HF commit sha once you are settled

[chunking]
strategy   = "structural"
max_tokens = 510                  # <= the model's window minus special tokens
overlap    = 64

[retrieval]
candidates_per_source = 50
rerank                = "local"

# [retrieval.confidence]
# fitted_for = "BAAI/bge-reranker-base@<revision>"
# low_below  = 0.31

[rerank]
provider = "sentence-transformers"
model    = "BAAI/bge-reranker-base"

[budget]
timezone          = "UTC"
"""

MODELS = Models(
    embedding=ModelPin("BAAI/bge-small-en-v1.5", "5" * 40),
    rerank=ModelPin("BAAI/bge-reranker-base", "2" * 40),
)


def test_metadata_is_inserted_inside_the_chunking_table_only() -> None:
    out = kb.set_chunking_metadata(STAMPED, "prefix")
    parsed = tomllib.loads(out)
    assert parsed["chunking"]["metadata"] == "prefix"
    assert parsed["chunking"]["overlap"] == 64 and parsed["retrieval"]["rerank"] == "local"
    assert "metadata" not in parsed["retrieval"] and "metadata" not in parsed["embedding"]
    assert out.count("metadata") == 1


def test_every_other_line_is_untouched() -> None:
    out = kb.set_chunking_metadata(STAMPED, "prefix")
    kept = [line for line in out.splitlines() if "metadata" not in line]
    assert kept == STAMPED.splitlines()


def test_setting_a_key_twice_replaces_it() -> None:
    once = kb.set_chunking_metadata(STAMPED, "prefix")
    twice = kb.set_chunking_metadata(once, "off")
    assert tomllib.loads(twice)["chunking"]["metadata"] == "off"
    assert twice.count("metadata") == 1


def test_a_commented_out_key_is_not_mistaken_for_the_key() -> None:
    out = kb.set_key(STAMPED, "embedding", "revision", '"abc"')
    assert tomllib.loads(out)["embedding"]["revision"] == "abc"
    assert "# revision" in out  # the comment survives


def test_missing_table_is_an_error() -> None:
    with pytest.raises(kb.BuildError, match=r"no \[nope\] table"):
        kb.set_key(STAMPED, "nope", "k", "1")


def test_pin_models_sets_both_revisions() -> None:
    parsed = tomllib.loads(kb.pin_models(STAMPED, MODELS))
    assert parsed["embedding"]["revision"] == "5" * 40
    assert parsed["rerank"]["revision"] == "2" * 40


def test_pin_models_stops_when_pinakes_stamps_a_different_model() -> None:
    other = STAMPED.replace("BAAI/bge-reranker-base", "BAAI/bge-reranker-v2-m3")
    with pytest.raises(kb.BuildError, match="Review the pin"):
        kb.pin_models(other, MODELS)


def test_thresholds_are_appended_and_replace_an_earlier_block() -> None:
    first = kb.set_confidence_thresholds(STAMPED, fitted_for="m@r", low_below=-4.5, high_above=1.25)
    block = tomllib.loads(first)["retrieval"]["confidence"]
    assert block == {"fitted_for": "m@r", "low_below": -4.5, "high_above": 1.25}
    second = kb.set_confidence_thresholds(first, fitted_for="m@r2", low_below=-3.0, high_above=0.5)
    assert tomllib.loads(second)["retrieval"]["confidence"]["fitted_for"] == "m@r2"
    assert (
        second.count("[retrieval.confidence]") == 2
    )  # the commented template line + one live table


def test_thresholds_leave_the_other_tables_alone() -> None:
    out = kb.set_confidence_thresholds(STAMPED, fitted_for="m@r", low_below=-1.0, high_above=1.0)
    parsed, before = tomllib.loads(out), tomllib.loads(STAMPED)
    for table in ("kb", "sources", "embedding", "chunking", "rerank", "budget"):
        assert parsed[table] == before[table]


def test_replacing_thresholds_keeps_the_tables_that_follow_them() -> None:
    with_block = STAMPED.replace(
        "[rerank]",
        '[retrieval.confidence]\nfitted_for = "old"\nlow_below = 1.0\nhigh_above = 2.0\n\n[rerank]',
    )
    out = kb.set_confidence_thresholds(with_block, fitted_for="new", low_below=-1.0, high_above=0.5)
    parsed = tomllib.loads(out)
    assert parsed["retrieval"]["confidence"]["fitted_for"] == "new"
    assert parsed["rerank"]["model"] == "BAAI/bge-reranker-base"
    assert parsed["budget"]["timezone"] == "UTC"
