"""Tests written against the mutants of `items.py` and `layout.py` that an independent review found
still surviving the suite.

Expected values come from the module docstrings, the dataset README's layout table, or arithmetic
by hand. The digests below were produced by the system `shasum`, not by the code under test:

    printf %s 'pnkbench-split-v1:order:mhr-0000' | shasum -a 256
    printf %s 'https://example.test/news/0' | shasum -a 1
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta, timezone

from toy import make_articles, make_dataset

from pnkbench import items, layout
from pnkbench.admission import admits
from pnkbench.multihop_rag import Dataset

# --- slugs and paths ----------------------------------------------------------------------


def test_a_cut_that_lands_after_a_hyphen_leaves_no_trailing_hyphen() -> None:
    assert layout.slug("word " * 30, 10) == "word-word"  # "word-word-" cut at 10, hyphen dropped
    assert layout.slug("aaaa bbbb", 5) == "aaaa"  # "aaaa-" cut at 5


def test_leading_and_trailing_separators_never_survive_into_a_slug() -> None:
    assert layout.slug("--Hello--") == "hello"
    assert layout.slug("(Updated) news!") == "updated-news"
    assert layout.slug("!!!") == "x"  # nothing left: the placeholder, not an empty name


def test_a_slug_is_cut_at_its_limit_exactly() -> None:
    assert layout.slug("a" * 40, 40) == "a" * 40
    assert layout.slug("a" * 41, 40) == "a" * 40
    assert layout.slug("a" * 200) == "a" * 60  # the default limit is the title's, 60


def test_non_ascii_letters_fold_to_ascii_and_symbols_without_a_fold_disappear() -> None:
    assert layout.slug("Zürich Café") == "zurich-cafe"
    assert layout.slug("Ångström — naïve") == "angstrom-naive"
    assert layout.slug("¿Qué pasa?") == "que-pasa"  # the inverted mark folds to nothing


def test_category_and_source_are_cut_at_forty_and_the_title_at_sixty() -> None:
    article = replace(make_articles()[0], category="c" * 50, source="s" * 50, title="t" * 100)
    assert layout.category_slug(article) == "c" * 40
    assert layout.source_slug(article) == "s" * 40
    digest = "7bcd5054"  # sha1 of https://example.test/news/0, first eight hex digits
    assert layout.doc_relpath(article) == f"docs/{'c' * 40}/{'s' * 40}/{'t' * 60}-{digest}.txt"


def test_two_toy_articles_have_exactly_the_path_the_readme_table_describes() -> None:
    articles = make_articles()
    # docs/<category>/<source>/<title-slug>-<8 hex of sha1(url)>.txt
    assert layout.doc_relpath(articles[0]) == (
        "docs/sports/the-verge/article-number-0-a-story-about-topic-0-7bcd5054.txt"
    )
    assert layout.doc_relpath(articles[7]) == (
        "docs/technology/the-guardian/article-number-7-a-story-about-topic-7-37204985.txt"
    )


def test_tags_are_the_category_then_source_with_the_readmes_source_prefix() -> None:
    article = make_articles()[0]  # category sports, source The Verge
    assert layout.category_tag(article) == "sports"
    assert layout.source_tag(article) == "source-the-verge"
    assert layout.tags_for(article) == ("sports", "source-the-verge")
    assert layout.category_prefix(article) == "docs/sports/"


# --- the fixed orderings ------------------------------------------------------------------


def test_calibration_order_is_the_pinned_digest_of_the_salted_id() -> None:
    assert items.calibration_order("mhr-0000") == (
        "69b55af8efa90ef9fd3b4f1fd3e7d2a519b2da200c2b36897a8440c0abb42e36"
    )
    assert items.calibration_order("mhr-0001") == (
        "c5649ff0cd618829cac54a6485922354c0dcd14a663e3005aa664921af658a9a"
    )
    assert items.calibration_order("mhr-0002") == (
        "23ed9398329a4b424dc0b1191883a3e0fc76fb02f6e99899e661c3dc84abec78"
    )
    # so the calibration sample is taken in the order of these hex strings
    ids = ["mhr-0000", "mhr-0001", "mhr-0002"]
    assert sorted(ids, key=items.calibration_order) == ["mhr-0002", "mhr-0000", "mhr-0001"]


def test_the_calibration_answerable_sample_is_capped_at_150() -> None:
    assert items.CALIBRATION_ANSWERABLE_CAP == 150


# --- derived filter variants --------------------------------------------------------------


def _brute_allowed(dataset: Dataset, item: items.RunItem) -> int:
    """How many articles the item's own filter admits, counted article by article."""
    return sum(
        admits(item.filters, a, layout.tags_for(a), layout.doc_relpath(a)) for a in dataset.articles
    )


def _every_variant_counts_what_its_filter_admits(dataset: Dataset) -> dict[str, items.RunItem]:
    derived = {i.id: i for i in items.derive_items(dataset)}
    variants = [i for i in derived.values() if i.variant != items.BASE]
    assert variants
    for item in variants:
        assert item.allowed == _brute_allowed(dataset, item), item.id
    return derived


def test_admitted_counts_on_the_toy_dataset_by_hand() -> None:
    derived = {i.id: i for i in items.derive_items(make_dataset())}
    # twelve articles over three categories (4 each) and four sources (3 each)
    for query in ("mhr-0000", "mhr-0002", "mhr-0003"):
        assert derived[f"{query}:f-cat-tag"].allowed == 4
        assert derived[f"{query}:f-cat-path"].allowed == 4
    assert derived["mhr-0006:f-src-tag"].allowed == 3
    # one article a day from 1 October: a window over articles lo..hi admits hi - lo + 1
    windows = {"0000": 4, "0001": 3, "0002": 4, "0003": 4, "0006": 5, "0008": 2}
    assert {q: derived[f"mhr-{q}:f-date"].allowed for q in windows} == windows
    assert "mhr-0004:f-date" not in derived  # gold spans article 0 to 11: the whole corpus


def test_a_date_window_includes_an_article_stamped_exactly_on_either_edge() -> None:
    dataset = make_dataset()
    articles = list(dataset.articles)
    # mhr-0001's gold are articles 1, 2, 3. Put the first at midnight sharp (the window's start)
    # and the last on the final microsecond of its day (the window's end).
    articles[1] = replace(articles[1], published_at=datetime(2023, 10, 2, 0, 0, tzinfo=UTC))
    articles[3] = replace(
        articles[3], published_at=datetime(2023, 10, 4, 23, 59, 59, 999999, tzinfo=UTC)
    )
    derived = _every_variant_counts_what_its_filter_admits(
        replace(dataset, articles=tuple(articles))
    )
    window = derived["mhr-0001:f-date"]
    assert window.allowed == 3  # articles 1, 2 and 3, edges included
    assert window.filters.modified_after == datetime(2023, 10, 2, tzinfo=UTC).timestamp()
    assert (
        window.filters.modified_before
        == datetime.combine(date(2023, 10, 4), time.max, UTC).timestamp()
    )


def test_the_admitted_count_does_not_depend_on_the_order_of_the_articles() -> None:
    dataset = make_dataset()
    # publication dates run backwards through the article list
    articles = tuple(
        replace(a, published_at=datetime(2023, 10, 12 - a.index, 9, 30, tzinfo=UTC))
        for a in dataset.articles
    )
    derived = _every_variant_counts_what_its_filter_admits(replace(dataset, articles=articles))
    # mhr-0000's gold are articles 0 and 3, published 12 and 9 October: a window of 9..12 October
    assert derived["mhr-0000:f-date"].allowed == 4


def test_a_window_is_cut_on_utc_days_whatever_offset_the_stamp_carries() -> None:
    dataset = make_dataset()
    articles = list(dataset.articles)
    # mhr-0008's gold are articles 10 and 11.
    # article 10: 12 Oct 03:00 at UTC+14 is 11 Oct 13:00 UTC (the local day is a day later)
    articles[10] = replace(
        articles[10],
        published_at=datetime(2023, 10, 12, 3, 0, tzinfo=timezone(timedelta(hours=14))),
    )
    # article 11: 12 Oct 22:00 at UTC-5 is 13 Oct 03:00 UTC (the local day is a day earlier)
    articles[11] = replace(
        articles[11],
        published_at=datetime(2023, 10, 12, 22, 0, tzinfo=timezone(timedelta(hours=-5))),
    )
    derived = _every_variant_counts_what_its_filter_admits(
        replace(dataset, articles=tuple(articles))
    )
    window = derived["mhr-0008:f-date"]
    assert window.filters.modified_after == datetime(2023, 10, 11, tzinfo=UTC).timestamp()
    assert (
        window.filters.modified_before
        == datetime.combine(date(2023, 10, 13), time.max, UTC).timestamp()
    )
    assert window.allowed == 2
