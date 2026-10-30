"""The things a run asks Pinakes: one base item per dataset query, plus oracle-filter variants.

**A filter variant is an oracle, and says so.** It is derived from the gold articles' own metadata,
so the filter *cannot* exclude a gold article (checked, not assumed). That makes it a test of the
filter machinery (is it applied before retrieval, does it admit exactly what it should, do two
spellings of one filter agree), not of whether a user could have guessed the filter. The dataset
ships no filter questions; these are derived, and every report labels them so.

Variants, each created only when it genuinely narrows the corpus:

    f-cat-tag   `tags=[<category>]`, when every gold article shares a category
    f-cat-path  `path_prefix='docs/<category>/'`: the same admitted set by a different code path
    f-src-tag   `tags=[source-<source>]`, when every gold article shares a source
    f-date      `modified_after/before` = the gold articles' publication days (UTC), widened to
                whole days
"""

from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, time

from pnkbench import layout
from pnkbench.multihop_rag import Dataset

BASE = "base"
F_CAT_TAG = "f-cat-tag"
F_CAT_PATH = "f-cat-path"
F_SRC_TAG = "f-src-tag"
F_DATE = "f-date"
FILTER_VARIANTS = (F_CAT_TAG, F_CAT_PATH, F_SRC_TAG, F_DATE)

SPLIT_SALT = "pnkbench-split-v1"
FIT, EVAL = "fit", "eval"
CALIBRATION_ANSWERABLE_CAP = 150
"""How many answerable questions go to `pinakes.calibrate`. It prints their score range and fits
nothing from them: both thresholds come from the unanswerable scores alone (calibrate.py)."""


@dataclass(frozen=True, slots=True)
class FilterSpec:
    tags: tuple[str, ...] = ()
    path_prefix: str | None = None
    modified_after: float | None = None
    modified_before: float | None = None

    def any_set(self) -> bool:
        return bool(
            self.tags
            or self.path_prefix
            or self.modified_after is not None
            or self.modified_before is not None
        )


@dataclass(frozen=True, slots=True)
class RunItem:
    id: str
    base_id: str
    variant: str
    query: str
    filters: FilterSpec
    allowed: int
    """How many of the corpus's articles the filter admits, from the dataset's own metadata."""


def item_id(base_id: str, variant: str) -> str:
    return base_id if variant == BASE else f"{base_id}:{variant}"


def split_of(base_id: str) -> str:
    """`fit` or `eval`, fixed by a hash of the id so nobody can choose which side a query lands on."""
    digest = hashlib.sha256(f"{SPLIT_SALT}:{base_id}".encode()).digest()
    return FIT if digest[0] % 2 == 0 else EVAL


def calibration_order(base_id: str) -> str:
    """A fixed, arbitrary ordering used to pick the calibration answerable sample."""
    return hashlib.sha256(f"{SPLIT_SALT}:order:{base_id}".encode()).hexdigest()


def _day_bounds(first: datetime, last: datetime) -> tuple[float, float]:
    start = datetime.combine(first.astimezone(UTC).date(), time.min, UTC)
    end = datetime.combine(last.astimezone(UTC).date(), time.max, UTC)
    return start.timestamp(), end.timestamp()


def derive_items(dataset: Dataset) -> list[RunItem]:
    articles = dataset.articles
    total = len(articles)
    by_category: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for article in articles:
        by_category[layout.category_tag(article)] = (
            by_category.get(layout.category_tag(article), 0) + 1
        )
        by_source[layout.source_tag(article)] = by_source.get(layout.source_tag(article), 0) + 1
    stamps = sorted(article.published_at.timestamp() for article in articles)

    items: list[RunItem] = []
    for query in dataset.queries:
        items.append(RunItem(query.id, query.id, BASE, query.text, FilterSpec(), total))
        if not query.answerable:
            continue
        gold = dataset.gold(query)

        categories = {layout.category_tag(a) for a in gold}
        if len(categories) == 1:
            (category,) = categories
            allowed = by_category[category]
            if allowed < total:
                items.append(
                    RunItem(
                        item_id(query.id, F_CAT_TAG),
                        query.id,
                        F_CAT_TAG,
                        query.text,
                        FilterSpec(tags=(category,)),
                        allowed,
                    )
                )
                items.append(
                    RunItem(
                        item_id(query.id, F_CAT_PATH),
                        query.id,
                        F_CAT_PATH,
                        query.text,
                        FilterSpec(path_prefix=layout.category_prefix(gold[0])),
                        allowed,
                    )
                )

        sources = {layout.source_tag(a) for a in gold}
        if len(sources) == 1:
            (source,) = sources
            allowed = by_source[source]
            if allowed < total:
                items.append(
                    RunItem(
                        item_id(query.id, F_SRC_TAG),
                        query.id,
                        F_SRC_TAG,
                        query.text,
                        FilterSpec(tags=(source,)),
                        allowed,
                    )
                )

        after, before = _day_bounds(
            min(a.published_at for a in gold), max(a.published_at for a in gold)
        )
        admitted = bisect.bisect_right(stamps, before) - bisect.bisect_left(stamps, after)
        if admitted < total:
            for article in gold:
                if not after <= article.published_at.timestamp() <= before:
                    raise ValueError(f"{query.id}: the date window excludes a gold article")
            items.append(
                RunItem(
                    item_id(query.id, F_DATE),
                    query.id,
                    F_DATE,
                    query.text,
                    FilterSpec(modified_after=after, modified_before=before),
                    admitted,
                )
            )
    return items
