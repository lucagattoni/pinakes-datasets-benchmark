"""Round three of the mutation review, for the code that needs the real models.

The comparison of two configurations that chunk differently, end to end: two real knowledge bases
built from the same toy corpus by Pinakes itself, one with the default `max_tokens` and one with a
lower one, both searched, then `pipeline.compare_step`. Exactly one article is long enough to be
cut differently, and the comparison must name exactly that one and set aside the questions that
need it. Run with `pytest -m slow tests/test_round3_slow.py` (about a minute).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from toy import make_dataset

pytestmark = pytest.mark.slow

pytest.importorskip("pinakes")

from pnkbench import compare, layout, paths, pinakes_api, pipeline  # noqa: E402
from pnkbench import multihop_rag as mh  # noqa: E402

REAL_CONFIGS = Path(__file__).resolve().parents[1] / "configs"

LOW_MAX_TOKENS = 300
LONG_ARTICLE = 8  # in the gold of mhr-0003 {5, 8} and mhr-0004 {0, 4, 8, 11}
# 30 copies of a 13-word sentence: 390 words, one paragraph. The bge tokenizer counts 14 tokens per
# copy (measured on 10, 14, 18, 20 and 25 copies: 140, 196, 252, 280 and 350 tokens), so about 420
# in all: one chunk under the default cap of 510, two under 300.
SENTENCE = "Quarterly revenue rose three percent as analysts expected steady demand across regional markets."
LONG_PARAGRAPH = " ".join([SENTENCE] * 30)


def _dataset_with_one_long_article() -> Any:
    dataset = make_dataset()
    article = dataset.articles[LONG_ARTICLE]
    body = (
        f"Opening paragraph of article {article.index}.\n\n{LONG_PARAGRAPH}\n\nClosing paragraph."
    )
    articles = list(dataset.articles)
    articles[LONG_ARTICLE] = dataclasses.replace(article, body=body)
    return dataclasses.replace(dataset, articles=tuple(articles))


def _write_dataset_like_the_real_one(root: Path, dataset: Any) -> None:
    corpus = [
        {
            "title": a.title,
            "author": a.author,
            "source": a.source,
            "category": a.category,
            "url": a.url,
            "published_at": a.published_at.isoformat(),
            "body": a.body,
        }
        for a in dataset.articles
    ]
    by_title = dataset.article_by_title()
    queries = [
        {
            "query": q.text,
            "answer": q.answer,
            "question_type": q.qtype,
            "evidence_list": [
                {
                    "title": e.title,
                    "url": e.url,
                    "fact": e.fact,
                    "author": None,
                    "source": by_title[e.title].source,
                    "category": by_title[e.title].category,
                    "published_at": by_title[e.title].published_at.isoformat(),
                }
                for e in q.evidence
            ],
        }
        for q in dataset.queries
    ]
    folder = root / "datasets" / mh.DATASET
    (folder / "raw").mkdir(parents=True)
    files = []
    for name, role, payload in (
        ("corpus.json", "corpus", corpus),
        ("MultiHopRAG.json", "queries", queries),
    ):
        data = json.dumps(payload).encode()
        (folder / "raw" / name).write_bytes(data)
        files.append(
            {
                "path": name,
                "role": role,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    (folder / "manifest.json").write_text(
        json.dumps(
            {"source_repo": "owner/toy", "revision": "a" * 40, "licence": "n/a", "files": files}
        )
    )


def _quiet(message: str) -> None:
    pass


@pytest.fixture(scope="module")
def two_configurations(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """A toy repository whose corpus has one long article, and two configurations built and run."""
    root = tmp_path_factory.mktemp("repo")
    shutil.copytree(REAL_CONFIGS, root / "configs")
    (root / "configs" / "lower-max-tokens.toml").write_text(
        'description = "The baseline with max_tokens lowered."\n'
        'backend = "st"\n'
        'chunking_metadata = "off"\n'
        f"chunking_max_tokens = {LOW_MAX_TOKENS}\n"
    )
    patch = pytest.MonkeyPatch()
    patch.setenv(paths.ROOT_ENV, str(root))
    dataset = _dataset_with_one_long_article()
    _write_dataset_like_the_real_one(root, dataset)
    (root / "runs").mkdir()
    for name in ("baseline", "lower-max-tokens"):
        pipeline.build(name, log=_quiet)
    runs = {
        name: pipeline.run(name, workers=1, log=_quiet) for name in ("baseline", "lower-max-tokens")
    }
    yield {"dataset": dataset, "runs": runs}
    patch.undo()


def _index(config_name: str) -> Path:
    return pinakes_api.index_path(paths.kb_dir(config_name))


def test_only_the_long_article_is_cut_differently_by_the_two_real_indexes(
    two_configurations: dict[str, Any],
) -> None:
    dataset = two_configurations["dataset"]
    long_path = layout.doc_relpath(dataset.articles[LONG_ARTICLE])
    roomy, tight = (
        compare.chunk_spans(_index("baseline")),
        compare.chunk_spans(_index("lower-max-tokens")),
    )
    assert len(roomy[long_path]) == 3  # opening, the one long paragraph, closing
    assert len(tight[long_path]) >= 4  # the long paragraph alone is now cut in two or more
    others = set(roomy) - {long_path}
    assert len(others) == 11 and all(roomy[p] == tight[p] for p in others)
    assert compare.articles_chunked_differently(
        _index("baseline"), _index("lower-max-tokens"), pipeline.path_index(dataset)
    ) == frozenset({LONG_ARTICLE})


def test_the_comparison_sets_aside_the_one_article_and_the_questions_that_need_it(
    two_configurations: dict[str, Any],
) -> None:
    runs = two_configurations["runs"]
    target = pipeline.compare_step(runs["baseline"], runs["lower-max-tokens"])
    aside = json.loads(target.with_suffix(".json").read_text())["without_set_aside_articles"]
    assert aside["articles"] == 1
    # article 8 is in the gold of mhr-0003 and mhr-0004; the other six answerable questions stay
    assert (aside["questions_set_aside"], aside["questions_kept"]) == (2, 6)
    text = target.read_text()
    assert "Read without the articles the two configurations chunk differently" in text
    assert "which splits 1 article" in text
    assert "chunking max_tokens | default | 300" in text
