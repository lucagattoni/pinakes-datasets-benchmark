"""`pnkbench`: fetch, build, run, calibrate, score, report, compare."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pnkbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch", help="download MultiHop-RAG at the pinned revision and verify it")

    build = sub.add_parser("build", help="build the knowledge base for one configuration")
    build.add_argument("--config", default="baseline")

    run = sub.add_parser("run", help="run every item through Pinakes; writes runs/<id>/")
    run.add_argument("--config", default="baseline")
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--sample", type=int, default=None, help="only this many base queries")
    run.add_argument("--resume", type=Path, default=None, help="a run directory to continue")

    for name, text in (
        ("calibrate", "fit confidence thresholds on the fit half; check them against Pinakes"),
        ("score", "compute every metric from the rows; writes summary.json"),
        ("harness", "run Pinakes' own evaluator over the eval half; writes harness.json"),
        ("report", "render report.md from the run's JSON"),
    ):
        step = sub.add_parser(name, help=text)
        step.add_argument("run_dir", type=Path)

    compare = sub.add_parser("compare", help="compare two runs question by question")
    compare.add_argument("run_a", type=Path)
    compare.add_argument("run_b", type=Path)

    everything = sub.add_parser("all", help="build, run, calibrate, score, harness, report")
    everything.add_argument("--config", default="baseline")
    everything.add_argument("--workers", type=int, default=2)
    everything.add_argument("--sample", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "fetch":
        from pnkbench import multihop_rag

        for path in multihop_rag.fetch():
            print(f"verified {path}")
        return 0
    if args.command == "build":
        from pnkbench import pipeline

        print(pipeline.build(args.config))
        return 0
    if args.command == "run":
        from pnkbench import pipeline

        print(
            pipeline.run(args.config, workers=args.workers, sample=args.sample, resume=args.resume)
        )
        return 0
    from pnkbench import pipeline

    if args.command == "calibrate":
        pipeline.calibrate_step(args.run_dir)
    elif args.command == "score":
        pipeline.score_step(args.run_dir)
    elif args.command == "harness":
        pipeline.harness_step(args.run_dir)
    elif args.command == "report":
        print(pipeline.report_step(args.run_dir))
    elif args.command == "compare":
        print(pipeline.compare_step(args.run_a, args.run_b))
    elif args.command == "all":
        print(pipeline.run_all(args.config, workers=args.workers, sample=args.sample))
    else:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
