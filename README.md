# pinakes-datasets-benchmark

Measures how well [Pinakes](https://github.com/lucagattoni/pinakes) searches a corpus when the right
answers are known: it builds a Pinakes knowledge base from a public dataset that ships ground-truth
evidence, asks it every question the dataset contains, and scores what comes back. The first dataset is
**MultiHop-RAG**. Every result records which Pinakes, which models, which dataset revision and which
commit of this repository produced it.

<!-- RESULTS -->

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

Three were the owner's, asked before building: **Pinakes 0.32.8 as the baseline with a switch for `main`**;
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
| `runs/` | committed results, one directory per run, and the paired comparisons |
| `tests/` | unit tests on a synthetic corpus, parity tests against Pinakes' own code, and the slow end-to-end tests |
| `build/` | generated knowledge bases; git-ignored, safe to delete |
