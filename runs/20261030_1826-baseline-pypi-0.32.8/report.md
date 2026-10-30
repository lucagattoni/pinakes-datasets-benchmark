# Run `20261030_1826-baseline-pypi-0.32.8`

## What produced this

|  |  |
|---|---|
| Pinakes | 0.32.8 from pypi; installed-code hash `ad0db0547b7e247b` |
| Configuration | `baseline`: Pinakes as a user meets it: sentence-transformers provider, every default, no title in the embedded text. |
| Embedding | `BAAI/bge-small-en-v1.5@5c38ec7c40`, 384 dims, sentence-transformers |
| Reranker | `BAAI/bge-reranker-base@2cfc18c941`, sentence-transformers |
| Chunking | headings = none, max_tokens = 510, overlap = 64, metadata = off |
| Retrieval | candidates_per_source = 50, fusion = rrf, fusion_top_k = 20, final_k = 8, rerank = local, vector_tier = auto, vector_tier_resolved = numpy |
| Dataset | `yixuantt/MultiHopRAG` @ `71ac0d0bd1`: 609 articles, 2556 queries |
| Items run | 8497 (base 2556, f-cat-tag 1544, f-cat-path 1544, f-src-tag 598, f-date 2255) |
| Execution | 3 worker(s), 2356.2 s wall, 20 passages per search; devices {'embedding': 'mps:0', 'rerank': 'mps:0'} |
| Reproducibility | 60 items searched again in one process: 0 rankings differed, largest score difference 0.00e+00 |
| Benchmark repo | v0.1.0, commit `c0b97d5b16` |
| Scored by | v0.1.0, commit `6b21f15d69`, 2026-10-30T20:28:43+00:00 |
| Environment | Python 3.13.18, torch 2.15.0, macOS-27.2-arm64-arm-64bit-Mach-O |

## Answerable questions

The primary measure is **all-evidence recall**: a question counts only when *every* gold article is retrieved. Pinakes' harness counts it when *any* is.

|  | result (95% Wilson interval) |
|---|---|
| every gold article in the first 8 passages (what `pnk search` returns by default) | 43.4% (41.3–45.4, n=2255) |
| every gold article in the first 5 passages | 33.8% (31.9–35.8, n=2255) |
| every gold article in the first 20 passages (the reranker's whole pool) | 61.0% (58.9–63.0, n=2255) |
| every gold article in the first 3 distinct articles | 32.0% (30.1–34.0, n=2255) |
| every gold article in the first 5 distinct articles | 43.5% (41.4–45.5, n=2255) |
| every gold article in the first 10 distinct articles | 57.5% (55.5–59.5, n=2255) |
| *any* gold article in the first 5 passages (Pinakes' harness definition) | 91.5% (90.3–92.6, n=2255) |
| *any* gold article in the first 8 passages (Pinakes' harness definition) | 96.8% (96.0–97.5, n=2255) |
| mean share of gold articles found in the first 10 articles | 81.6% |
| MRR of the first gold article | 0.802 |
| nDCG@10 (articles, binary) | 0.722 |
| distinct articles among the 20 passages (median, p10–p90) | 10 (6–15) |

### By question type

| type | n | all in first 8 passages | all in first 20 passages | any in first 5 passages | MRR |
|---|---|---|---|---|---|
| comparison_query | 856 | 60.3% (57.0–63.5, n=856) | 74.3% (71.3–77.1, n=856) | 92.6% (90.7–94.2, n=856) | 0.830 |
| inference_query | 816 | 25.2% (22.4–28.3, n=816) | 46.2% (42.8–49.6, n=816) | 94.0% (92.1–95.4, n=816) | 0.817 |
| temporal_query | 583 | 43.9% (39.9–48.0, n=583) | 62.1% (58.1–65.9, n=583) | 86.3% (83.2–88.8, n=583) | 0.742 |

### By number of gold articles

| gold articles | n | all in first 8 passages | all in first 20 passages | any in first 5 passages | MRR |
|---|---|---|---|---|---|
| 2 | 1169 | 67.1% (64.3–69.7, n=1169) | 80.1% (77.7–82.3, n=1169) | 93.7% (92.1–94.9, n=1169) | 0.848 |
| 3 | 774 | 22.1% (19.3–25.1, n=774) | 45.0% (41.5–48.5, n=774) | 87.3% (84.8–89.5, n=774) | 0.747 |
| 4 | 312 | 7.4% (5.0–10.8, n=312) | 29.2% (24.4–34.4, n=312) | 93.6% (90.3–95.8, n=312) | 0.767 |

## Abstention: can the score tell a null question from an answerable one?

The AUROC is the probability that a random answerable question's top rerank score beats a random null question's (0.5 = no separation, 1.0 = perfect). It needs no thresholds.

| comparison | AUROC (95% bootstrap interval; n) |
|---|---|
| answerable vs null, all questions | 0.944 (0.929–0.957; 2255 vs 301) |
| answerable vs null, eval half | 0.935 (0.912–0.954; 1140 vs 154) |
| answerable vs null, fit half | 0.954 (0.934–0.971; 1115 vs 147) |
| null vs answerable *found* (all gold in the first 10 articles) | 0.956 (0.945–0.967; 1297 vs 301) |
| **null without any style tell** vs answerable, all questions | 0.945 (0.882–0.982; 2255 vs 18) |

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
| answerable | 0.00 | 0.17 | 0.49 | 0.89 | 0.99 | 1.00 | 1.00 |
| null | 0.00 | 0.00 | 0.00 | 0.01 | 0.06 | 0.34 | 0.99 |

### Confidence labels

Thresholds fitted by `pinakes.calibrate` on the **fit half** (147 null questions): `low_below = 0.0124`, `high_above = 0.1565`. The **eval half** was not used in the fit.

|  | false confidence: null labelled high | false abstain (Pinakes' definition: any gold in first 5 passages, labelled low) | false abstain (all gold in first 10 articles, labelled low) |
|---|---|---|---|
| eval half (held out) | 16.2% (11.2–22.9, n=154) | 1.1% (0.6–1.8, n=1140) | 0.4% (0.1–0.9, n=1140) |
| fit half (fitted on) | 10.9% (6.8–16.9, n=147) | 0.5% (0.2–1.2, n=1115) | 0.2% (0.0–0.7, n=1115) |

Checks: the printed thresholds were re-derived from 147 stored scores (agree: True); 600 searches repeated with the thresholds in the manifest gave 0 label mismatches against ours.

## Filters (oracle variants)

Each variant filters on the gold articles' own metadata, so it cannot exclude a gold article. It tests the filter machinery, not whether a user would have chosen the filter ([how they are derived](../../datasets/multihop-rag/README.md#filter-variants)).

| variant | n | share of corpus admitted | soundness violations | not applied before retrieval | all gold in first 10 articles, unfiltered | same, filtered | filtered better / worse | sign test p |
|---|---|---|---|---|---|---|---|---|
| `f-cat-tag` | 1544 | 28.1% | 0 | 0 | 63.5% (61.0–65.8, n=1544) | 66.7% (64.3–69.0, n=1544) | +65 / -15 | <0.0001 |
| `f-cat-path` | 1544 | 28.1% | 0 | 0 | 63.5% (61.0–65.8, n=1544) | 66.7% (64.3–69.0, n=1544) | +65 / -15 | <0.0001 |
| `f-src-tag` | 598 | 12.8% | 0 | 0 | 75.1% (71.5–78.4, n=598) | 83.9% (80.8–86.7, n=598) | +56 / -3 | <0.0001 |
| `f-date` | 2255 | 36.9% | 0 | 0 | 57.5% (55.5–59.5, n=2255) | 72.7% (70.9–74.5, n=2255) | +360 / -17 | <0.0001 |

Tag and path-prefix spellings of the same category: 1544 pairs, **0 returned different passages or scores**.

## Pinakes' own harness (`python -m pinakes.eval`, k = 5)

2370 questions ({'base': 1294, 'f-cat-tag': 782, 'f-src-tag': 294}). any-of scoring: a hit is any one expected article among the first k passages; multi-hop questions carry no scripted hops; eval half only.

| metric | value |
|---|---|
| questions | 2370 |
| recall_at_k | 0.9350 |
| mrr | 0.8167 |
| rerank_precision | 0.7319 |
| false_abstain | 0.0068 |
| false_confidence | 0.1623 |
| confidence_coverage | 1.0000 |

By kind:

| kind | recall@k |
|---|---|
| filter | 0.9554 |
| multi-hop | 0.9158 |
| no-answer | 1.0000 |

## Latency

Per search, in milliseconds, with 3 worker(s) sharing one GPU (so under contention): min 268, p10 529, p25 630, median 769, p75 1009, p90 1246, max 1692.

## Reading this

The dataset's own caveats (LLM-generated queries, templated null questions, yes/no answers) are in [its README](../../datasets/multihop-rag/README.md#caveats). Pinakes chunks plain text by paragraph by design ([DESIGN §4.6](https://github.com/lucagattoni/pinakes/blob/main/docs/DESIGN.md)), which is why twenty passages can come from few articles.
