# Run `20261030_1910-metadata-prefix-pypi-0.32.8`

## What produced this

|  |  |
|---|---|
| Pinakes | 0.32.8 from pypi; installed-code hash `ad0db0547b7e247b` |
| Configuration | `metadata-prefix`: The baseline plus [chunking] metadata = "prefix" (each chunk's document title is embedded with it) and max_tokens = 459, which prefix requires. |
| Embedding | `BAAI/bge-small-en-v1.5@5c38ec7c40`, 384 dims, sentence-transformers |
| Reranker | `BAAI/bge-reranker-base@2cfc18c941`, sentence-transformers |
| Chunking | headings = none, max_tokens = 459, overlap = 64, metadata = prefix |
| Retrieval | candidates_per_source = 50, fusion = rrf, fusion_top_k = 20, final_k = 8, rerank = local, vector_tier = auto, vector_tier_resolved = numpy |
| Dataset | `yixuantt/MultiHopRAG` @ `71ac0d0bd1`: 609 articles, 2556 queries |
| Items run | 8497 (base 2556, f-cat-tag 1544, f-cat-path 1544, f-src-tag 598, f-date 2255) |
| Execution | 3 worker(s), 3029.4 s wall, 20 passages per search; devices {'embedding': 'mps:0', 'rerank': 'mps:0'} |
| Reproducibility | 60 items searched again in one process: 0 rankings differed, largest score difference 0.00e+00 |
| Benchmark repo | v0.1.0, commit `946bef9e23` |
| Scored by | v0.1.0, commit `6b21f15d69`, 2026-10-30T20:28:55+00:00 |
| Environment | Python 3.13.18, torch 2.15.0, macOS-27.2-arm64-arm-64bit-Mach-O |

## Answerable questions

The primary measure is **all-evidence recall**: a question counts only when *every* gold article is retrieved. Pinakes' harness counts it when *any* is.

|  | result (95% Wilson interval) |
|---|---|
| every gold article in the first 8 passages (what `pnk search` returns by default) | 42.3% (40.2–44.3, n=2255) |
| every gold article in the first 5 passages | 33.3% (31.4–35.3, n=2255) |
| every gold article in the first 20 passages (the reranker's whole pool) | 55.7% (53.6–57.7, n=2255) |
| every gold article in the first 3 distinct articles | 32.2% (30.3–34.1, n=2255) |
| every gold article in the first 5 distinct articles | 43.8% (41.7–45.8, n=2255) |
| every gold article in the first 10 distinct articles | 52.8% (50.7–54.8, n=2255) |
| *any* gold article in the first 5 passages (Pinakes' harness definition) | 91.4% (90.2–92.5, n=2255) |
| *any* gold article in the first 8 passages (Pinakes' harness definition) | 95.7% (94.8–96.5, n=2255) |
| mean share of gold articles found in the first 10 articles | 78.7% |
| MRR of the first gold article | 0.804 |
| nDCG@10 (articles, binary) | 0.712 |
| distinct articles among the 20 passages (median, p10–p90) | 9 (4–13) |

### By question type

| type | n | all in first 8 passages | all in first 20 passages | any in first 5 passages | MRR |
|---|---|---|---|---|---|
| comparison_query | 856 | 59.0% (55.7–62.2, n=856) | 71.8% (68.7–74.8, n=856) | 93.6% (91.7–95.0, n=856) | 0.840 |
| inference_query | 816 | 25.2% (22.4–28.3, n=816) | 40.8% (37.5–44.2, n=816) | 93.4% (91.5–94.9, n=816) | 0.804 |
| temporal_query | 583 | 41.5% (37.6–45.6, n=583) | 52.7% (48.6–56.7, n=583) | 85.6% (82.5–88.2, n=583) | 0.749 |

### By number of gold articles

| gold articles | n | all in first 8 passages | all in first 20 passages | any in first 5 passages | MRR |
|---|---|---|---|---|---|
| 2 | 1169 | 65.8% (63.0–68.4, n=1169) | 76.1% (73.6–78.5, n=1169) | 93.7% (92.1–94.9, n=1169) | 0.852 |
| 3 | 774 | 20.8% (18.1–23.8, n=774) | 38.0% (34.6–41.5, n=774) | 87.3% (84.8–89.5, n=774) | 0.752 |
| 4 | 312 | 7.4% (5.0–10.8, n=312) | 22.8% (18.5–27.7, n=312) | 93.3% (89.9–95.6, n=312) | 0.749 |

## Abstention: can the score tell a null question from an answerable one?

The AUROC is the probability that a random answerable question's top rerank score beats a random null question's (0.5 = no separation, 1.0 = perfect). It needs no thresholds.

| comparison | AUROC (95% bootstrap interval; n) |
|---|---|
| answerable vs null, all questions | 0.944 (0.928–0.956; 2255 vs 301) |
| answerable vs null, eval half | 0.935 (0.913–0.956; 1140 vs 154) |
| answerable vs null, fit half | 0.952 (0.932–0.969; 1115 vs 147) |
| null vs answerable *found* (all gold in the first 10 articles) | 0.959 (0.945–0.970; 1190 vs 301) |
| **null without any style tell** vs answerable, all questions | 0.943 (0.879–0.982; 2255 vs 18) |

**Read the first rows with care.** The null queries differ from the answerable ones in their wording (see [the dataset notes](../../datasets/multihop-rag/README.md#caveats)), so a score gap may be reading style. The bold row keeps only null queries with none of these tells:

| tell | answerable queries | null queries |
|---|---|---|
| asks for a letter | 0 | 144 |
| opens with considering the information | 2 | 184 |
| names no outlet | 14 | 211 |
| null queries with none of the three | n/a | 18 |

Top-1 rerank score. The scale belongs to the provider: sentence-transformers applies a sigmoid to a single-label cross-encoder, so these are probabilities between 0 and 1:

|  | min | p10 | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|---|---|
| answerable | 0.00 | 0.17 | 0.48 | 0.88 | 0.99 | 1.00 | 1.00 |
| null | 0.00 | 0.00 | 0.00 | 0.01 | 0.06 | 0.35 | 0.99 |

### Confidence labels

Thresholds fitted by `pinakes.calibrate` on the **fit half** (147 null questions): `low_below = 0.0113`, `high_above = 0.1688`. The **eval half** was not used in the fit.

|  | false confidence: null labelled high | false abstain (Pinakes' definition: any gold in first 5 passages, labelled low) | false abstain (all gold in first 10 articles, labelled low) |
|---|---|---|---|
| eval half (held out) | 16.2% (11.2–22.9, n=154) | 0.9% (0.5–1.6, n=1140) | 0.2% (0.0–0.6, n=1140) |
| fit half (fitted on) | 10.9% (6.8–16.9, n=147) | 0.4% (0.2–1.0, n=1115) | 0.0% (0.0–0.3, n=1115) |

Checks: the printed thresholds were re-derived from 147 stored scores (agree: True); 600 searches repeated with the thresholds in the manifest gave 0 label mismatches against ours.

## Filters (oracle variants)

Each variant filters on the gold articles' own metadata, so it cannot exclude a gold article. It tests the filter machinery, not whether a user would have chosen the filter ([how they are derived](../../datasets/multihop-rag/README.md#filter-variants)).

| variant | n | share of corpus admitted | soundness violations | not applied before retrieval | all gold in first 10 articles, unfiltered | same, filtered | filtered better / worse | sign test p |
|---|---|---|---|---|---|---|---|---|
| `f-cat-tag` | 1544 | 28.1% | 0 | 0 | 59.5% (57.0–61.9, n=1544) | 62.6% (60.1–64.9, n=1544) | +55 / -7 | <0.0001 |
| `f-cat-path` | 1544 | 28.1% | 0 | 0 | 59.5% (57.0–61.9, n=1544) | 62.6% (60.1–64.9, n=1544) | +55 / -7 | <0.0001 |
| `f-src-tag` | 598 | 12.8% | 0 | 0 | 71.6% (67.8–75.0, n=598) | 81.1% (77.8–84.0, n=598) | +59 / -2 | <0.0001 |
| `f-date` | 2255 | 36.9% | 0 | 0 | 52.8% (50.7–54.8, n=2255) | 67.7% (65.8–69.6, n=2255) | +344 / -7 | <0.0001 |

Tag and path-prefix spellings of the same category: 1544 pairs, **0 returned different passages or scores**.

## Pinakes' own harness (`python -m pinakes.eval`, k = 5)

2370 questions ({'base': 1294, 'f-cat-tag': 782, 'f-src-tag': 294}). any-of scoring: a hit is any one expected article among the first k passages; multi-hop questions carry no scripted hops; eval half only.

| metric | value |
|---|---|
| questions | 2370 |
| recall_at_k | 0.9368 |
| mrr | 0.8245 |
| rerank_precision | 0.7437 |
| false_abstain | 0.0063 |
| false_confidence | 0.1623 |
| confidence_coverage | 1.0000 |

By kind:

| kind | recall@k |
|---|---|
| filter | 0.9582 |
| multi-hop | 0.9167 |
| no-answer | 1.0000 |

## Latency

Per search, in milliseconds, with 3 worker(s) sharing one GPU (so under contention): min 287, p10 693, p25 812, median 969, p75 1278, p90 1634, max 2272.

## Reading this

The dataset's own caveats (LLM-generated queries, templated null questions, yes/no answers) are in [its README](../../datasets/multihop-rag/README.md#caveats). Pinakes chunks plain text by paragraph by design ([DESIGN §4.6](https://github.com/lucagattoni/pinakes/blob/main/docs/DESIGN.md)), which is why twenty passages can come from few articles.
