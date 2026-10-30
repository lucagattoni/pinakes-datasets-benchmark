"""A tiny hand-made dataset with the structure of MultiHop-RAG, for tests that need no network.

Twelve articles in three categories and four sources, published over twelve days; ten queries:
answerable ones with two to four evidence items (one listing an article twice, as 168 real ones
do) and two null ones. Small enough to check by eye, structured enough that every filter variant,
both splits and every gold size occur.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pnkbench.multihop_rag import Article, Dataset, Evidence, Query

_CATEGORIES = ["sports", "technology", "business"]
_SOURCES = ["The Verge", "Sporting News", "TechCrunch", "The Guardian"]


def make_articles() -> list[Article]:
    articles = []
    for i in range(12):
        articles.append(
            Article(
                index=i,
                title=f"Article number {i}: a story about topic {i}",
                author=None if i % 5 == 0 else f"Author {i}",
                source=_SOURCES[i % 4],
                category=_CATEGORIES[i % 3],
                url=f"https://example.test/news/{i}",
                published_at=datetime(2023, 10, 1 + i, 9, 30, tzinfo=UTC),
                body=(
                    f"Opening paragraph of article {i}.\n\n"
                    f"The distinctive fact of article {i} is the word zebra{i}.\n\n"
                    f"Closing paragraph of article {i}."
                ),
            )
        )
    return articles


def _evidence(articles: list[Article], *indices: int) -> tuple[Evidence, ...]:
    return tuple(
        Evidence(title=articles[i].title, url=articles[i].url, fact=f"fact {i}") for i in indices
    )


def make_dataset() -> Dataset:
    a = make_articles()
    queries = [
        # articles 0 and 3: both `sports`, from two different sources
        Query("mhr-0000", 0, "q0 sports pair", "yes", "comparison_query", _evidence(a, 0, 3)),
        # articles 1, 2, 3: three categories, three sources
        Query("mhr-0001", 1, "q1 mixed trio", "x", "inference_query", _evidence(a, 1, 2, 3)),
        # articles 1 and 4: both `technology`
        Query("mhr-0002", 2, "q2 tech pair", "no", "temporal_query", _evidence(a, 1, 4)),
        # one article listed twice: the gold set has two members, not three
        Query("mhr-0003", 3, "q3 repeated", "yes", "comparison_query", _evidence(a, 5, 5, 8)),
        Query("mhr-0004", 4, "q4 quartet", "y", "inference_query", _evidence(a, 0, 4, 8, 11)),
        Query("mhr-0005", 5, "q5 null", "Insufficient information.", "null_query", ()),
        Query("mhr-0006", 6, "q6 same source", "z", "comparison_query", _evidence(a, 0, 4)),
        Query("mhr-0007", 7, "q7 null two", "Insufficient information.", "null_query", ()),
        Query("mhr-0008", 8, "q8 late pair", "yes", "temporal_query", _evidence(a, 10, 11)),
        Query("mhr-0009", 9, "q9 whole span", "n", "comparison_query", _evidence(a, 0, 11)),
    ]
    return Dataset(articles=tuple(a), queries=tuple(queries), sha256={})
