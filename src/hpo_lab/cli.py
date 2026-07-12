"""Command-line entry point for reproducible benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from hpo_lab.config import ExperimentConfig
from hpo_lab.plots import plot_convergence, plot_quality_vs_time, plot_seed_stability
from hpo_lab.search import available_methods, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hpo-lab",
        description="Run a fair, reproducible HPO benchmark.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="benchmark",
        choices=("benchmark",),
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "quick", "full"),
        default="quick",
    )
    parser.add_argument(
        "--dataset",
        choices=("diabetes", "breast_cancer"),
        default="diabetes",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=available_methods(),
        dest="methods",
        help="Optimizer to include; repeat this flag to select multiple.",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/latest"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExperimentConfig.for_mode(args.mode, n_jobs=args.n_jobs)
    result = run_benchmark(
        config,
        dataset_name=args.dataset,
        methods=args.methods,
    )
    output_dir = result.save(args.output_dir)
    for name, plotter in (
        ("convergence.png", plot_convergence),
        ("quality-vs-time.png", plot_quality_vs_time),
        ("seed-stability.png", plot_seed_stability),
    ):
        figure, _ = plotter(result)
        figure.savefig(output_dir / name, dpi=160, bbox_inches="tight")

    printable = result.summary.copy()
    numeric = printable.select_dtypes("number").columns
    printable[numeric] = printable[numeric].round(4)
    print(printable.to_string(index=False))
    print(f"\nArtifacts: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
