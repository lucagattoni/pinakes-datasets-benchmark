# pinakes-datasets-benchmark: project instructions

What this is, how to run it and what each measure means: [`README.md`](README.md). Adding a dataset: [`datasets/README.md`](datasets/README.md).

## 🛑 This repository is PUBLIC

- **Never commit a dataset's text** (articles, questions, evidence). It is fetched, checksummed and git-ignored; a run commits numbers, article indices and ids. The two files that carry question text stay ignored: `runs/*/calibration-questions.yaml`, `runs/*/harness/questions.yaml` ([why](datasets/multihop-rag/README.md#caveats)).
- Run the global PRIMARY CHECK before every push. Name no private repository, path or host in a file or a commit message.

## Running

- **Start any job longer than a minute from the worktree's `.venv/bin/pnkbench`, never `uv run`.** A re-sync of the environment kills the worker processes, `multiprocessing` replaces them silently, and the run stalls with no error ([README, Reproduce](README.md#reproduce)).
- One pipeline at a time: they share one GPU, and overlapping runs make every latency meaningless.
- The Pinakes extras `release` and `main` cannot share an environment. Say so in the commit when a pin changes; a run records the Pinakes it used, with a hash of its code.

## Results are records

- Never edit a file under `runs/` by hand: regenerate with `pnkbench score` or `report <run dir>`. A run's header names the commit that searched, its summary the commit that scored.
- A run that lands updates README's Results in the same commit; every figure there is read from that run's report.
- A measure is defined once, in README ([What is measured](README.md#what-is-measured)): change the definition there, then the code.

## Code and tests

- Only `src/pnkbench/pinakes_api.py` imports Pinakes. Keep it so.
- Run `pytest -m slow` (it loads the real models) before landing a change to the pipeline.
- **A test's expected value comes from the spec or hand arithmetic, never from running the code and pasting its output.** New code is not "tested" until a fresh reviewer's own mutants have been run against it: the author's 43 of 43 killed, a reviewer's 193 still left 98 alive.

<!-- Provenance, 20261031. Public-repo rules: the user made the repo public on 20261030 after a pre-publication check that found one private repository name in the history; the user chose to leave it there. Long-job rule: a first baseline run stalled when `uv run` re-synced the venv under its workers. One-pipeline rule: the latencies in the first reports were taken with a second job contending. Mutant figures: three independent review rounds on this repo, 20261030. -->
