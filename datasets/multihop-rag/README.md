# MultiHop-RAG in this repository

This page covers what is **specific to this repository**: how the dataset is pinned, how it becomes a
Pinakes knowledge base, which questions are derived from it, and what to be careful about when
reading the results. For the dataset itself (how it was built, its query types, its statistics),
read the sources instead of a copy of them:

| | |
|---|---|
| Paper | Tang and Yang, *MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries*, [arXiv:2401.15391](https://arxiv.org/abs/2401.15391) |
| Data card | [`yixuantt/MultiHopRAG`](https://huggingface.co/datasets/yixuantt/MultiHopRAG) on the Hugging Face Hub |
| Authors' repository | [`yixuantt/MultiHop-RAG`](https://github.com/yixuantt/MultiHop-RAG) |

In one sentence: a corpus of English news articles and questions whose evidence is spread over two to
four of them; `null` questions are the ones the corpus cannot answer, and declining is the right
behaviour.

## Pin and integrity

[`manifest.json`](manifest.json) is the single record: Hub repository, **full commit revision**, the
size and SHA-256 of both files, and the licence with the evidence for it (ODC-By). Nothing about it
is repeated here, so it cannot drift.

`pnkbench fetch` downloads at that revision into `raw/` (git-ignored) and compares what arrives with
the manifest. A mismatch stops it and leaves no partial file; a file already on disk is verified, never
replaced. `pnkbench` re-checks both files on every load. The checksums were first recorded by an
earlier fetch (20261010) and this repository's own fresh download reproduced them (20261030).

## What loading checks

[`multihop_rag.py`](../../src/pnkbench/multihop_rag.py) refuses a dataset in which article titles or URLs
repeat, in which an evidence item names no article or carries another URL than its article's, in which a
`null` query carries evidence or any other query carries fewer than two items, or in which a timestamp
has no timezone. These are the properties every later stage leans on. The shape is also asserted by a
test (skipped until the data is fetched):

| | |
|---|---|
| Articles / queries | 609 / 2,556 |
| Query types | comparison 856, inference 816, temporal 583, null 301 |
| Distinct gold articles per answerable query | 2: 1,169 · 3: 774 · 4: 312 |

## From an article to a file in the knowledge base

| In the dataset | In the knowledge base |
|---|---|
| `body` | the file's content, byte for byte: `docs/<category>/<source>/<title-slug>-<8 hex of sha1(url)>.txt` |
| `title` | the sidecar's `title`, written **before the first sync** so it is the publisher's title, not a filename |
| `category`, `source` | sidecar tags: `<category>` and `source-<source>` |
| `url` | sidecar `provenance.source` |
| `published_at` | the file's **modification time**, so Pinakes' date filters mean the publication date |
| `author` | not used |

`.txt` rather than `.md` because the articles have no headings and the text is left exactly as
published. No underscore appears in any generated path: Pinakes' `path_prefix` filter is a SQL `LIKE`,
where `_` matches any character. The layout is in [`layout.py`](../../src/pnkbench/layout.py); how
Pinakes then chunks the text is in [its design notes](https://github.com/lucagattoni/pinakes/blob/main/docs/DESIGN.md#46-chunking-and-tokens).

## Gold articles

A query's **gold** is the set of distinct articles its `evidence_list` names. 168 queries list one article twice,
so the gold set is smaller than the evidence count for them. A retrieval counts as complete only if
every gold article is among the results ([how the measures are defined](../../README.md#what-is-measured)).
The evidence sentences (`fact`) are not used: relevance is judged at the article level.

## Items derived from the dataset

The dataset ships queries, not filter questions. This repository derives **8,497 items** from the 2,556
queries ([`items.py`](../../src/pnkbench/items.py)):

| Variant | What it is | Items | Share of the corpus it admits (mean) |
|---|---|---|---|
| `base` | the query as written | 2,556 | 100% |
| `f-cat-tag` | tag filter on the category, when every gold article shares one | 1,544 | 28.1% |
| `f-cat-path` | the same set of articles selected by `path_prefix` instead | 1,544 | 28.1% |
| `f-src-tag` | tag filter on the source, when every gold article shares one | 598 | 12.8% |
| `f-date` | publication window of the gold articles, widened to whole UTC days | 2,255 | 36.9% |

### Filter variants

Each is an **oracle**: it is built from the gold articles' own metadata, so it cannot exclude a gold
article (checked for every item, not assumed), and it is created only when it narrows the corpus. That
makes it a test of the *filter machinery* (is it applied before retrieval, does it admit exactly the
right articles, do two spellings of one filter agree), not of whether a user would have chosen it. The
two category variants admit the same articles by different code paths, so they must return identical
results.

## Split

Queries are split in two halves by a fixed hash of the query id ([`items.split_of`](../../src/pnkbench/items.py)),
so nobody can choose which side a query lands on. The **fit** half (1,115 answerable, 147 null) is where Pinakes' confidence
thresholds are fitted; the **eval** half (1,140 answerable, 154 null) is where every threshold-dependent
number is read. Measures that need no thresholds are given on all queries.

## Caveats

- **Generated by a language model.** Per the paper, GPT-4 generated the queries and answers and a model extracted the
  evidence. They were not checked here and can be wrong or leading.
- **The null queries are written differently from the others.** Measured on all 2,556 queries (20261030):

  | Surface feature | answerable (2,255) | null (301) |
  |---|---|---|
  | mentions a "letter" ("which single letter…") | 0 | 144 (47.8%) |
  | contains "Considering the information" | 2 (0.1%) | 184 (61.1%) |
  | names no news outlet | 14 (0.6%) | 211 (70.1%) |

  A score that separates null from answerable questions may therefore be reading the wording, not the
  content. Every abstention result is also given on the null queries that have none of these features.
- **Article-level relevance.** Pinakes retrieves passages. A retrieval counts when it returns a passage *of* a gold
  article, not the evidence sentence itself.
- **The embedding model predates the corpus.** The earliest article is from 26 September 2023; the
  FlagEmbedding README dates `bge-*-v1.5` to 12 September 2023. The reranker's training data and date were not
  established, and the models' training data is unstated by their authors.
- **Licence and attribution.** The dataset is licensed ODC-By 1.0 (evidence in [`manifest.json`](manifest.json)); the news text comes from
  many publishers whose own terms were not checked. The articles are therefore never committed here, and neither is the
  dataset's question text: the files that carry it (`calibration-questions.yaml`, `harness/questions.yaml`) are regenerated
  by the pipeline and git-ignored. What is committed from a run is numbers, article indices and ids. This repository
  contains information derived from MultiHop-RAG (Tang and Yang, 2024, [arXiv:2401.15391](https://arxiv.org/abs/2401.15391)),
  made available under the [Open Data Commons Attribution License 1.0](https://opendatacommons.org/licenses/by/1-0/).
