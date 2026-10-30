# pinakes-datasets-benchmark

Measures how well [Pinakes](https://github.com/lucagattoni/pinakes) searches a corpus when the right
answers are known: it builds a Pinakes knowledge base from a public dataset that ships ground-truth
evidence, asks it every question the dataset contains, and scores what comes back. The first dataset is
**MultiHop-RAG**. Every result records which Pinakes, which models, which dataset revision and which
commit of this repository produced it.

## Results

First full run, **Pinakes 0.32.8 from PyPI with its defaults** on MultiHop-RAG: 2,255 answerable and 301 null questions, 8,497 searches in all.
Everything below is read from the committed run
[`20261030_1826-baseline-pypi-0.32.8`](runs/20261030_1826-baseline-pypi-0.32.8/report.md); intervals are 95% ([how each measure is defined](#what-is-measured)).

**Finding some of the evidence is easy; finding all of it is not.** A question needs two to four articles.

| In the first… | at least one gold article (Pinakes' own harness counts this a hit) | **every** gold article |
|---|---|---|
| 5 passages | 91.5% | 33.8% |
| 8 passages (what `pnk search` returns by default) | 96.8% | **43.4%** (41.3–45.4) |
| 20 passages (the reranker's whole pool) | not computed | 61.0% (58.9–63.0) |

- **By how much evidence a question needs**, every gold article in the first 8 passages: two articles 67.1%, three 22.1%, four 7.4%. By type: comparison 60.3%, temporal 43.9%, inference 25.2%.
- **Where the other 56.6% go.** For 39.0% of answerable questions at least one gold article is missing from the whole 20-passage pool; for a further 17.6% every gold article is in the pool but not all in the first 8.
- **Abstention.** The top rerank score separates null from answerable questions well: AUROC 0.944 (0.929–0.957). The null questions are worded differently from the rest, so this was re-checked on the 18 that have none of those tells: 0.945 (0.882–0.982), so it is not an artefact of wording (a small sample, hence the wide interval).
- **Confidence labels.** Thresholds fitted on one half of the questions and read on the other: a null question is labelled high-confidence 16.2% of the time (11.2–22.9, n=154) against 10.9% on the half the thresholds were fitted on, which is the optimism `pinakes.calibrate` itself warns about. An answerable question whose evidence was found (by Pinakes' any-article definition) is labelled low 1.1% of the time.
- **Filters.** Across 5,941 filtered searches: 0 returned an article the filter excludes, 0 showed the symptom of a filter applied after retrieval (a candidate pool smaller than 20 although at least 20 articles were admitted), and the tag and the path-prefix spelling of the same category returned identical results in all 1,544 pairs. Filtering helped far more questions than it hurt but is not monotone: with an oracle filter, every gold article in the first 10 articles went from 63.5% to 66.7% for a category (+65 questions, −15), from 75.1% to 83.9% for a source (+56, −3) and from 57.5% to 72.7% for a date window (+360, −17).
- **The metadata-prefix variant does not help.** Embedding each chunk's title (which also requires `max_tokens = 459`) leaves the first 8 passages at 42.3% against 43.4% (79 questions gained, 104 lost, p = 0.076) and loses ground in the pool: 55.7% against 61.0% (71 gained, 191 lost, p < 0.0001). Its 20 passages cover fewer distinct articles (median 9 against 10). Setting aside the 11 articles that the lower `max_tokens` re-splits, and the 48 questions that need them, changes nothing. [Paired comparison](runs/compare--20261030_1826-baseline-pypi-0.32.8--20261030_1910-metadata-prefix-pypi-0.32.8.md).
- **Pinakes' own harness agrees where it overlaps.** `python -m pinakes.eval` on the held-out half reports a false-confidence rate of 0.162, the same figure as above; its recall (0.935) is high because it counts any one article.
- **Pinakes `main` against the release.** The 0.32.8 release and `main` at `b1bad68` both report version 0.32.8, but the installed Python is byte-identical (the same code hash, `ad0db0547b7e…`), and on a **freshly rebuilt** knowledge base 200 of 200 sampled searches returned the same passages in the same order with the same scores ([comparison](runs/compare--20261030_1826-baseline-pypi-0.32.8--20261030_2030-baseline-git-b1bad68.md); that sample was run from a scratch checkout, so its header records no benchmark commit). The same check shows that rebuilding is deterministic.
- **The same searches, repeated in one process** after the run, returned identical rankings and scores for 60 of 60; on a separate 147-item probe, rankings and scores were also identical whether 1, 2, 3 or 4 worker processes ran.

### What this shows about Pinakes

- **A paragraph is a chunk, by design** ([DESIGN §4.6](https://github.com/lucagattoni/pinakes/blob/main/docs/DESIGN.md)). On plain-text news this gives 28,748 chunks for 609 articles, median 38 tokens, so 20 passages cover about 10 articles (median; 6 to 15 at the 10th to 90th percentile). That is the context for the gap between "some" and "all" above.
- **The rerank score scale depends on the provider.** With `sentence-transformers` the reranker returns sigmoid probabilities between 0 and 1 (read from Pinakes' provider code and the sentence-transformers source, and seen in every one of this run's scores), while Pinakes' own calibration notes record values of about −0.28 and −7.85. A confidence threshold is stored with `fitted_for = model@revision`, which names neither provider nor scale. Whether a threshold fitted under one provider is silently misapplied under the other was **not run here** (no `fastembed` environment); it is worth checking.
- **`metadata = "prefix"` needs a lower `max_tokens`**: Pinakes refuses to build at the default 510 for 37 of these 609 documents, and names 459 as the limit that fits them all.

### Limits

One dataset, whose questions and answers were generated by a language model; one machine; default settings throughout (`final_k` and the candidate pool sizes were not tuned); the filters are oracles built from the gold articles' own metadata, so they test the machinery and not a user's choice of filter; latencies were measured with three workers sharing one GPU and say little about a single query. Not run yet: a `fastembed` baseline, Pinakes' graph channel, and the paid `--deep` loop.

## The two things being measured

**The dataset: MultiHop-RAG.** A corpus of 609 English news articles and 2,556 questions whose evidence is
spread over two to four of them; 301 are *null* questions the corpus cannot answer, where declining is
correct. For the dataset itself, read its
[paper](https://arxiv.org/abs/2401.15391) and [data card](https://huggingface.co/datasets/yixuantt/MultiHopRAG).
What is specific to this repository (the pin, how articles become files, the derived filter questions,
the fit/eval split, and what to be careful about) is in
**[`datasets/multihop-rag/README.md`](datasets/multihop-rag/README.md)**.

**The system: Pinakes.** A portable, agent-first knowledge base: one directory is one KB. Its free search
pipeline is a metadata filter, then BM25 and dense vectors in parallel, reciprocal rank fusion, a local
cross-encoder rerank, and cited passages with a confidence signal. Read its
[design](https://github.com/lucagattoni/pinakes/blob/main/docs/DESIGN.md) (retrieval is §4, chunking §4.6,
its own quality gates §7), the [manifest reference](https://github.com/lucagattoni/pinakes/blob/main/docs/MANIFEST.md)
and the [CLI reference](https://github.com/lucagattoni/pinakes/blob/main/docs/CLI.md). Specific to this repository:

| | |
|---|---|
| Version under test | `release`: [0.32.8 from PyPI](https://pypi.org/project/pinakes/0.32.8/), or `main`: a pinned commit of Pinakes' `main` (see [switching](#switching-the-pinakes-under-test)) |
| Models | the defaults, `BAAI/bge-small-en-v1.5` (embedding) and `BAAI/bge-reranker-base` (rerank), run through the `sentence-transformers` provider and pinned to full Hub commits in [`configs/models.toml`](configs/models.toml) |
| Settings | Pinakes' own defaults: 510-token chunks with 64 overlap, 50 BM25 and 50 vector candidates fused by RRF, the best 20 reranked. The only variant changes `[chunking] metadata` |
| How it is called | in-process through `pinakes.search.search` on a read-only index, 20 passages per question; every call to Pinakes' internals is in one file, [`pinakes_api.py`](src/pnkbench/pinakes_api.py), which checks the installed build before a run |
| Chunking | by design a plain-text article becomes **one chunk per paragraph** ([DESIGN §4.6](https://github.com/lucagattoni/pinakes/blob/main/docs/DESIGN.md)): 609 articles become 28,748 chunks, median 38 tokens, 70.8% under 64 tokens. So 20 passages may come from only a few articles |

## What is measured

Defined once, here; the reports name these and link back.

- **Gold articles** of a question: the distinct articles its evidence names (2, 3 or 4 of them).
- **All-evidence recall** (the primary measure): the question counts only if **every** gold article is
  among the results. Counted two ways: *in the first k passages* (what `pnk search` shows a user; Pinakes returns
  8 by default) and *in the first k distinct articles* (the ranking quality, whatever the chunking).
- **Any-evidence recall**: at least one gold article is among the results. This is what Pinakes' own harness
  calls a hit ([`eval.py`](https://github.com/lucagattoni/pinakes/blob/main/src/pinakes/eval.py)), so it is
  reported under its own name and never as the headline.
- **MRR** of the first gold article; **nDCG@10** over articles; mean **coverage** (share of gold found).
- **Abstention.** The AUROC of the top rerank score for telling answerable from null questions (0.5 is no
  separation); and, once Pinakes' confidence thresholds are fitted on the fit half, the held-out **false-confidence** rate
  (null labelled high) and **false-abstain** rate (an answerable question that was found, labelled low), by Pinakes' own definitions.
- **Filters** (oracle variants, [how derived](datasets/multihop-rag/README.md#filter-variants)). Four checks:
  *soundness* (every returned article satisfies the filter), *applied before retrieval* (a post-filter returns
  fewer candidates than the pool), *equivalence* (the tag and the path-prefix spelling of one category return identical results), and *effect* (all-evidence recall, paired with the unfiltered question).
- **Paired comparison** between two runs: the questions one found and the other missed, with an exact sign test.

Intervals are 95%: Wilson for proportions, bootstrap (500 resamples, fixed seed) for AUROC.

## How it works

`pnkbench fetch` → `build` → `run` → `calibrate` → `score` → `harness` → `report`, or `pnkbench all`.

| Step | What it does | Code |
|---|---|---|
| `fetch` | downloads the dataset at the pinned revision; refuses anything that differs from the manifest | [`multihop_rag.py`](src/pnkbench/multihop_rag.py) |
| `build` | writes one file and sidecar per article, runs `pnk init` and `pnk sync`, then verifies the index holds exactly the dataset | [`kb.py`](src/pnkbench/kb.py), [`layout.py`](src/pnkbench/layout.py) |
| `run` | searches every item with 3 worker processes; resumable; a stall raises instead of waiting; re-searches a sample in one process to check reproducibility | [`runner.py`](src/pnkbench/runner.py), [`pipeline.py`](src/pnkbench/pipeline.py) |
| `calibrate` | runs `python -m pinakes.calibrate` on the fit half, re-derives its thresholds from the stored scores, writes them into the KB manifest, and checks Pinakes' own confidence labels against ours | [`calibration.py`](src/pnkbench/calibration.py) |
| `score` | every metric above, from the rows file alone | [`scoring.py`](src/pnkbench/scoring.py) |
| `harness` | feeds the eval half to `python -m pinakes.eval`, so the numbers sit beside the ones Pinakes publishes for its own corpora | [`harness.py`](src/pnkbench/harness.py) |
| `report`, `compare` | Markdown from the JSON; a paired comparison of two runs | [`report.py`](src/pnkbench/report.py), [`compare.py`](src/pnkbench/compare.py) |

A run lives in `runs/<UTC time>-<config>-<pinakes>/`: `run.json` (everything that produced it), `rows.jsonl`
(one row per search: passages as article index and score, no text), `summary.json`, `calibration.json`,
`harness.json`, `report.md`.

## Choices

Three were the maintainer's, asked before building: **Pinakes 0.32.8 as the baseline with a switch for `main`**;
**the `sentence-transformers` provider** (not `fastembed`); and **all four extras** (Pinakes' harness
export, the metadata-prefix variant, filter questions derived from metadata, confidence calibration).

Settled from sources, not asked: Python 3.13 and `uv` (Pinakes requires 3.13); the model weights pinned to the Hub
`main` commits that were verified equal to the cached ones (Pinakes records no revision unless the manifest names one);
one `.txt` file per article; a fixed hash split; 3 worker processes, measured (1 worker 2.12 items/s, 2 → 2.81, 3 → 3.01,
4 → 3.02, so 3), with rankings and scores bit-identical across 1 to 4 processes.

## Reproduce

Needs Python 3.13, [`uv`](https://docs.astral.sh/uv/), a network for the first run (the dataset, and the models if not
already cached; the reranker's PyTorch weights are about 1 GB) and an Apple-silicon GPU or patience: a full run is 8,497 searches.

```bash
uv sync --extra release --group dev
```

```bash
uv run pnkbench fetch
```

```bash
.venv/bin/pnkbench all --config baseline --workers 3
```

Start long jobs from `.venv/bin/pnkbench`, not `uv run`: `uv run` may re-sync the environment, and a worker that
dies while the environment changes is replaced silently by `multiprocessing`. (A first run here stalled that way; the
runner now raises after 300 s without a finished batch, and `--resume <run dir>` continues.)

```bash
.venv/bin/pnkbench all --config metadata-prefix --workers 3
```

```bash
.venv/bin/pnkbench compare runs/<baseline run> runs/<metadata-prefix run>
```

Tests: `uv run pytest` (no network, no models), and `uv run pytest -m slow` for the end-to-end tests against real Pinakes.

### Switching the Pinakes under test

The two extras name the same package from two sources, so they cannot share an environment. `--extra release` installs
the pinned PyPI release; `--extra main` installs the Pinakes commit pinned in [`pyproject.toml`](pyproject.toml)
(`[tool.uv.sources]`). Change either pin deliberately and run `uv lock`.

```bash
uv sync --extra main --group dev
```

The version string alone cannot tell a release from the commit after it, so each run also records a SHA-256 of the
installed Pinakes code, and `compare` states both builds at the top.

## Layout

| | |
|---|---|
| `src/pnkbench/` | the tool |
| `datasets/multihop-rag/` | the dataset's README and `manifest.json`; `raw/` is fetched and git-ignored |
| `configs/` | the configurations (`baseline`, `metadata-prefix`) and the model pins |
| `runs/` | committed results, one directory per run, and the paired comparisons; the two files carrying the dataset's question text are regenerated and git-ignored |
| `tests/` | unit tests on a synthetic corpus, parity tests against Pinakes' own code, and the slow end-to-end tests |
| `build/` | generated knowledge bases; git-ignored, safe to delete |

## Licence

The code is licensed under the [Apache License 2.0](LICENSE), as Pinakes is. The dataset is not redistributed here: it is
fetched from its authors under ODC-By 1.0 (see [its notes](datasets/multihop-rag/README.md#caveats) for attribution and
what is and is not committed), and the models are downloaded from the Hugging Face Hub under their own licences.
