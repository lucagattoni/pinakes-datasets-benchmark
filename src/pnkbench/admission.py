"""What a filter admits, in one place: the definition both the question deriver and the scorer use.

Pinakes' semantics (`search._filter_sql`): tags AND together, `path_prefix` is a prefix of the
KB-relative path, and the date window bounds the file's modification time, both ends inclusive.
Here the modification time is the article's publication time, which the KB builder sets.

Kept apart from `items` and `scoring` so that deriving a filter and checking a filter cannot drift
apart: `items.derive_items` refuses to create a variant that excludes one of its own gold articles,
and `scoring.filter_report` checks every result against the same function.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pnkbench.multihop_rag import Article


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


def admits(spec: FilterSpec, article: Article, tags: Sequence[str], relpath: str) -> bool:
    """Whether Pinakes' filter semantics admit `article`."""
    if any(tag not in tags for tag in spec.tags):
        return False
    if spec.path_prefix and not relpath.startswith(spec.path_prefix):
        return False
    stamp = article.published_at.timestamp()
    if spec.modified_after is not None and stamp < spec.modified_after:
        return False
    return not (spec.modified_before is not None and stamp > spec.modified_before)
