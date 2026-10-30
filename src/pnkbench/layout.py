"""How an article becomes a file: its path, its tags, the prefix that selects its category.

Pure functions, shared by the KB builder and the question deriver so the two cannot disagree about
where a document lives. **No underscore appears in any generated path.** Pinakes' `path_prefix`
filter is a SQL `LIKE` (`d.path LIKE 'prefix%'`), where `_` matches any single character, so an
underscore in a prefix would quietly widen the filter; hyphens are not special.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from pnkbench.multihop_rag import Article

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
TITLE_SLUG_LIMIT = 60
DOCS_ROOT = "docs"


def slug(text: str, limit: int = TITLE_SLUG_LIMIT) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    cleaned = _NON_ALNUM.sub("-", folded.lower()).strip("-")
    return cleaned[:limit].rstrip("-") or "x"


def category_slug(article: Article) -> str:
    return slug(article.category, 40)


def source_slug(article: Article) -> str:
    return slug(article.source, 40)


def doc_relpath(article: Article) -> str:
    """`docs/<category>/<source>/<title>-<8 hex of sha1(url)>.txt`, relative to the KB root.

    The URL hash makes the name unique without leaning on the title slug, and it is stable: the
    URL is part of the pinned dataset.
    """
    digest = hashlib.sha1(article.url.encode(), usedforsecurity=False).hexdigest()[:8]
    return (
        f"{DOCS_ROOT}/{category_slug(article)}/{source_slug(article)}/"
        f"{slug(article.title)}-{digest}.txt"
    )


def category_tag(article: Article) -> str:
    return category_slug(article)


def source_tag(article: Article) -> str:
    return f"source-{source_slug(article)}"


def tags_for(article: Article) -> tuple[str, ...]:
    return (category_tag(article), source_tag(article))


def category_prefix(article: Article) -> str:
    return f"{DOCS_ROOT}/{category_slug(article)}/"
