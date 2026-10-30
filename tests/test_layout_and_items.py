from __future__ import annotations

from collections import Counter

import pytest
from toy import make_articles, make_dataset

from pnkbench import items, layout


def test_slug_is_ascii_hyphenated_and_never_empty() -> None:
    assert layout.slug("Hello, World! It's 2023…") == "hello-world-it-s-2023"
    assert layout.slug("Zürich Café") == "zurich-cafe"
    assert layout.slug("???") == "x"
    assert "_" not in layout.slug("snake_case_title")


def test_slug_truncation_leaves_no_trailing_hyphen() -> None:
    assert not layout.slug("word " * 30, 12).endswith("-")
    assert len(layout.slug("word " * 30, 12)) <= 12


def test_paths_are_unique_stable_and_underscore_free() -> None:
    articles = make_articles()
    paths = [layout.doc_relpath(a) for a in articles]
    assert len(set(paths)) == len(paths)
    assert paths == [layout.doc_relpath(a) for a in articles]
    assert all("_" not in p for p in paths), "an underscore is a LIKE wildcard in path_prefix"
    assert all(p.startswith("docs/") and p.endswith(".txt") for p in paths)


def test_doc_path_changes_with_url_not_title_alone() -> None:
    a = make_articles()[0]
    from dataclasses import replace

    other = replace(a, url=a.url + "x")
    assert layout.doc_relpath(a) != layout.doc_relpath(other)


def test_tags_and_prefix_agree_with_the_path() -> None:
    for article in make_articles():
        path = layout.doc_relpath(article)
        assert path.startswith(layout.category_prefix(article))
        assert layout.tags_for(article) == (
            layout.category_tag(article),
            layout.source_tag(article),
        )


def test_split_is_deterministic_and_balanced() -> None:
    ids = [f"mhr-{i:04d}" for i in range(2000)]
    sides = [items.split_of(i) for i in ids]
    assert sides == [items.split_of(i) for i in ids]
    share = Counter(sides)[items.FIT] / len(ids)
    assert 0.45 < share < 0.55


def test_base_item_exists_for_every_query_and_only_answerable_get_variants() -> None:
    dataset = make_dataset()
    derived = items.derive_items(dataset)
    base = [i for i in derived if i.variant == items.BASE]
    assert [i.id for i in base] == [q.id for q in dataset.queries]
    null_ids = {q.id for q in dataset.queries if not q.answerable}
    assert not [i for i in derived if i.base_id in null_ids and i.variant != items.BASE]


def test_every_filter_variant_admits_all_its_gold_articles() -> None:
    from pnkbench.scoring import admits

    dataset = make_dataset()
    queries = {q.id: q for q in dataset.queries}
    checked = 0
    for item in items.derive_items(dataset):
        if item.variant == items.BASE:
            continue
        for article in dataset.gold(queries[item.base_id]):
            assert admits(
                item.filters, article, layout.tags_for(article), layout.doc_relpath(article)
            ), f"{item.id} would exclude gold article {article.index}"
            checked += 1
    assert checked > 10


def test_a_variant_exists_only_when_it_narrows_the_corpus() -> None:
    dataset = make_dataset()
    total = len(dataset.articles)
    for item in items.derive_items(dataset):
        if item.variant != items.BASE:
            assert item.allowed < total, item.id
    # q9 spans the first and last day, so its date window admits everything: no f-date variant.
    assert "mhr-0009:f-date" not in {i.id for i in items.derive_items(dataset)}


def test_category_variants_come_in_matching_pairs() -> None:
    derived = items.derive_items(make_dataset())
    tag = {i.base_id for i in derived if i.variant == items.F_CAT_TAG}
    path = {i.base_id for i in derived if i.variant == items.F_CAT_PATH}
    assert tag == path and tag, "tag and path spellings must be created together"


def test_source_variant_requires_one_shared_source() -> None:
    derived = items.derive_items(make_dataset())
    ids = {i.base_id for i in derived if i.variant == items.F_SRC_TAG}
    # q6 pairs articles 0 and 4: both come from index % 4 == 0, i.e. one source.
    assert "mhr-0006" in ids
    assert "mhr-0000" not in ids  # articles 0 and 3 come from different sources


def test_date_window_is_whole_utc_days() -> None:
    derived = {i.id: i for i in items.derive_items(make_dataset())}
    item = derived["mhr-0008:f-date"]  # articles 10 and 11: 11 and 12 October, 09:30 UTC
    from datetime import UTC, datetime

    assert item.filters.modified_after == datetime(2023, 10, 11, 0, 0, tzinfo=UTC).timestamp()
    assert item.filters.modified_before > datetime(2023, 10, 12, 23, 59, tzinfo=UTC).timestamp()
    assert item.allowed == 2


def test_gold_lists_a_repeated_article_once() -> None:
    dataset = make_dataset()
    query = next(q for q in dataset.queries if q.id == "mhr-0003")
    assert len(query.evidence) == 3
    assert [a.index for a in dataset.gold(query)] == [5, 8]


@pytest.mark.parametrize("bad", ["", "x"])
def test_item_id_for_base_is_the_query_id(bad: str) -> None:
    assert items.item_id("mhr-0001" + bad, items.BASE) == "mhr-0001" + bad
    assert items.item_id("mhr-0001", items.F_DATE) == "mhr-0001:f-date"


def test_split_assignments_are_pinned() -> None:
    """The documented split counts depend on these exact assignments; flipping the rule moves them."""
    assert {f"mhr-{i:04d}": items.split_of(f"mhr-{i:04d}") for i in range(8)} == {
        "mhr-0000": "eval",
        "mhr-0001": "eval",
        "mhr-0002": "fit",
        "mhr-0003": "fit",
        "mhr-0004": "eval",
        "mhr-0005": "fit",
        "mhr-0006": "fit",
        "mhr-0007": "eval",
    }


def test_no_category_or_source_variant_when_the_filter_would_admit_everything() -> None:
    from dataclasses import replace

    dataset = make_dataset()
    flat = replace(
        dataset,
        articles=tuple(replace(a, category="sports", source="The Verge") for a in dataset.articles),
    )
    variants = {i.variant for i in items.derive_items(flat)}
    assert items.F_CAT_TAG not in variants and items.F_CAT_PATH not in variants
    assert items.F_SRC_TAG not in variants


def test_path_ends_with_eight_hex_digits_of_the_url_hash() -> None:
    import hashlib

    article = make_articles()[0]
    digest = hashlib.sha1(article.url.encode(), usedforsecurity=False).hexdigest()[:8]
    assert layout.doc_relpath(article).endswith(f"-{digest}.txt")


def test_category_prefix_ends_with_a_slash_so_it_cannot_match_a_longer_name() -> None:
    prefix = layout.category_prefix(make_articles()[0])
    assert prefix.endswith("/")
    assert not "docs/sports-extra/x.txt".startswith(prefix)
