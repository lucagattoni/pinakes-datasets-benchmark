"""Real Pinakes, real models, a 12-article corpus. Run with `uv run pytest -m slow`.

This is the one place the filter machinery is exercised against Pinakes itself rather than against
rows written by hand: build the KB through the production code path, search every derived item,
and score the result with the production scorer.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from toy import make_dataset

pytestmark = pytest.mark.slow

pytest.importorskip("pinakes")

from pnkbench import config, items, kb, layout, paths, pinakes_api, pipeline, runner, scoring  # noqa: E402

REAL_CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("repo")
    shutil.copytree(REAL_CONFIGS, root / "configs")
    mp = pytest.MonkeyPatch()
    mp.setenv(paths.ROOT_ENV, str(root))
    dataset = make_dataset()
    report = kb.build_kb(dataset, config.load("baseline"), log=lambda m: None)
    handle = pinakes_api.open_kb(paths.kb_dir("baseline"))
    yield dataset, report, handle
    handle.close()
    mp.undo()


def test_the_built_index_is_exactly_the_dataset(built) -> None:
    dataset, report, _ = built
    assert report["documents"] == len(dataset.articles) == 12
    assert report["chunks"] >= 36  # three paragraphs each


def test_a_distinctive_word_finds_its_article_first(built) -> None:
    dataset, _, handle = built
    index = pipeline.path_index(dataset)
    for article in dataset.articles:
        payload = {
            "id": "x",
            "base_id": "x",
            "variant": "base",
            "query": f"zebra{article.index}",
            "tags": [],
            "path_prefix": None,
            "after": None,
            "before": None,
        }
        row = runner.run_one(handle, index, payload, 20)
        assert row["p"][0][0] == article.index, f"zebra{article.index} -> {row['p'][0]}"


def test_pinakes_applies_every_derived_filter_correctly(built) -> None:
    dataset, _, handle = built
    index = pipeline.path_index(dataset)
    derived = items.derive_items(dataset)
    rows = [runner.run_one(handle, index, runner.to_payload(i), 20) for i in derived]
    report = scoring.filter_report(
        {r["id"]: r for r in rows},
        derived,
        dataset,
        scoring.query_info(dataset),
        relpath_of=layout.doc_relpath,
        tags_of=layout.tags_for,
        fusion_top_k=20,
    )
    assert set(report) >= {items.F_CAT_TAG, items.F_CAT_PATH, items.F_SRC_TAG, items.F_DATE}
    for variant in items.FILTER_VARIANTS:
        assert report[variant]["soundness_violations"] == 0, report[variant]
    assert report["equivalence_tag_vs_path_prefix"]["mismatches"] == 0
    assert report["equivalence_tag_vs_path_prefix"]["pairs"] >= 3


def test_a_filter_that_matches_nothing_returns_nothing(built) -> None:
    _, _, handle = built
    outcome = pinakes_api.search_kb(
        handle, "zebra3", items.FilterSpec(tags=("no-such-tag",)), limit=20
    )
    assert outcome.passages == ()


def test_the_date_window_uses_the_publication_time_set_on_each_file(built) -> None:
    dataset, _, handle = built
    window = items.FilterSpec(
        modified_after=dataset.articles[2].published_at.timestamp(),
        modified_before=dataset.articles[4].published_at.timestamp(),
    )
    outcome = pinakes_api.search_kb(handle, "article", window, limit=20)
    index = pipeline.path_index(dataset)
    assert {index[path] for path, _ in outcome.passages} <= {2, 3, 4}
    assert {index[path] for path, _ in outcome.passages} == {2, 3, 4}
