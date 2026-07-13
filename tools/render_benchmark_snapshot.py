#!/usr/bin/env python
"""Render the tracked quick benchmark into both README files."""

from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RUNS_PATH = ROOT / "benchmarks" / "quick" / "runs.csv"
CONFIG_PATH = ROOT / "benchmarks" / "quick" / "config.json"
ENVIRONMENT_PATH = ROOT / "benchmarks" / "quick" / "environment.json"
START = "<!-- benchmark-snapshot:start -->"
END = "<!-- benchmark-snapshot:end -->"

GUIDANCE = {
    "en": {
        "Grid": "Small discrete spaces; transparent baseline",
        "TPE": "Mixed and conditional spaces",
        "GP": "Low-dimensional, expensive objectives",
        "CMA-ES": "Continuous parameters with interactions",
        "Random": "Space scouting and a strong baseline",
        "Baseline": "Untuned reference",
    },
    "ko": {
        "Grid": "작은 이산 공간과 투명한 기준선",
        "TPE": "혼합형·조건부 탐색공간",
        "GP": "저차원·고비용 목적함수",
        "CMA-ES": "상호작용하는 연속형 파라미터",
        "Random": "탐색 범위 점검과 강한 기준선",
        "Baseline": "튜닝하지 않은 참조점",
    },
}


def load_snapshot() -> tuple[list[dict[str, str]], dict, dict]:
    with RUNS_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    environment = json.loads(ENVIRONMENT_PATH.read_text(encoding="utf-8"))
    validate(rows, config)
    return rows, config, environment


def validate(rows: list[dict[str, str]], config: dict) -> None:
    seeds = set(config["experiment"]["seeds"])
    grouped = group_by_method(rows)
    methods = {"Baseline", "Grid", "Random", "TPE", "GP", "CMA-ES"}
    if set(grouped) != methods:
        raise ValueError("Benchmark snapshot has an unexpected method set.")
    expected_pairs = {(method, seed) for method in methods for seed in seeds}
    actual_pairs = [(row["method"], int(row["seed"])) for row in rows]
    if len(actual_pairs) != len(set(actual_pairs)):
        raise ValueError("Each method/seed pair must occur exactly once.")
    if set(actual_pairs) != expected_pairs:
        raise ValueError("Every method must contain each configured seed exactly once.")

    experiment = config["experiment"]
    optimizer_fits = (
        experiment["trials_per_method"] * experiment["cv_folds"] + 1
    )
    baseline_fits = experiment["cv_folds"] + 1
    for row in rows:
        expected_fits = baseline_fits if row["method"] == "Baseline" else optimizer_fits
        try:
            actual_fits = Decimal(row["fit_count"])
        except InvalidOperation as error:
            raise ValueError("Fit counts must be numeric.") from error
        if actual_fits != expected_fits:
            raise ValueError(
                f"{row['method']} seed {row['seed']} must report "
                f"{expected_fits} fits."
            )


def group_by_method(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    return grouped


def metrics(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped = group_by_method(rows)
    baseline = {
        int(row["seed"]): float(row["holdout_loss"])
        for row in grouped["Baseline"]
    }
    output = []
    for method, method_rows in grouped.items():
        losses = [float(row["holdout_loss"]) for row in method_rows]
        improvements = (
            []
            if method == "Baseline"
            else [
                (baseline[int(row["seed"])] - float(row["holdout_loss"]))
                / baseline[int(row["seed"])]
                * 100
                for row in method_rows
            ]
        )
        output.append(
            {
                "method": method,
                "mean_loss": mean(losses),
                "min_loss": min(losses),
                "max_loss": max(losses),
                "improvement": mean(improvements) if improvements else None,
                "seconds": mean(float(row["search_seconds"]) for row in method_rows),
                "fits": int(mean(float(row["fit_count"]) for row in method_rows)),
            }
        )
    return sorted(output, key=lambda row: (row["method"] == "Baseline", row["mean_loss"]))


def render(language: str, rows: list[dict[str, str]], config: dict, environment: dict) -> str:
    values = metrics(rows)
    experiment = config["experiment"]
    dataset = config["dataset"]
    seeds = ", ".join(str(seed) for seed in experiment["seeds"])
    python = environment["python"]
    commit = environment["source_commit"]
    commit_link = (
        "https://github.com/hyeonsangjeon/Hyperparameters-Optimization/commit/"
        f"{commit}"
    )
    if language == "ko":
        heading = "## ⚡ 재현 가능한 벤치마크 스냅샷"
        setup = (
            f"**{dataset['name']} · {dataset['samples']}개 샘플 · {dataset['features']}개 특성 · "
            f"{dataset['model']} · 방법당 {experiment['trials_per_method']} trials · "
            f"{experiment['cv_folds']}-fold CV · seeds {seeds}**"
        )
        headers = (
            "| 방법 | Holdout MSE ↓ | 관측 범위 | Baseline 대비 개선 ↑ | "
            "탐색시간 ↓ | Fit 수 | 적합한 상황 |\n"
            "|---|---:|---:|---:|---:|---:|---|"
        )
        note = (
            "> **해석:** 이것은 보편적 순위표가 아니라 동일 예산에서 얻은 재현 가능한 "
            "스냅샷입니다. 두 seed만 사용했으므로 평균과 관측 범위를 함께 보세요. "
            "탐색시간은 하드웨어와 병렬 설정에 따라 달라집니다."
        )
        source = (
            f"원본: [`runs.csv`](benchmarks/quick/runs.csv) · "
            f"[설정](benchmarks/quick/config.json) · "
            f"[환경](benchmarks/quick/environment.json) (Python {python})<br>\n"
            f"기준 커밋: [`{commit[:7]}`]({commit_link}) · "
            "재현: `uv run hpo-lab benchmark --mode quick`"
        )
    else:
        heading = "## ⚡ Reproducible benchmark snapshot"
        setup = (
            f"**{dataset['name']} · {dataset['samples']} samples · {dataset['features']} features · "
            f"{dataset['model']} · {experiment['trials_per_method']} trials/method · "
            f"{experiment['cv_folds']}-fold CV · seeds {seeds}**"
        )
        headers = (
            "| Method | Holdout MSE ↓ | Observed range | Improvement vs baseline ↑ | "
            "Search time ↓ | Fits | Good fit |\n"
            "|---|---:|---:|---:|---:|---:|---|"
        )
        note = (
            "> **Interpretation:** This is a reproducible equal-budget snapshot, not a "
            "universal leaderboard. With only two seeds, read the mean together with the "
            "observed range. Search time varies by hardware and parallel settings."
        )
        source = (
            f"Source: [`runs.csv`](benchmarks/quick/runs.csv) · "
            f"[config](benchmarks/quick/config.json) · "
            f"[environment](benchmarks/quick/environment.json) (Python {python})<br>\n"
            f"Source commit: [`{commit[:7]}`]({commit_link}) · "
            "Reproduce: `uv run hpo-lab benchmark --mode quick`"
        )

    table_rows = []
    for row in values:
        improvement = "—" if row["improvement"] is None else f"+{row['improvement']:.1f}%"
        table_rows.append(
            f"| **{row['method']}** | {row['mean_loss']:.1f} | "
            f"{row['min_loss']:.1f}–{row['max_loss']:.1f} | {improvement} | "
            f"{row['seconds']:.2f}s | {row['fits']} | {GUIDANCE[language][row['method']]} |"
        )
    return "\n\n".join(
        [heading, setup, headers + "\n" + "\n".join(table_rows), note, source]
    )


def update_readme(path: Path, rendered: str) -> str:
    content = path.read_text(encoding="utf-8")
    if START not in content or END not in content:
        raise ValueError(f"{path.name} is missing benchmark snapshot markers.")
    prefix, remainder = content.split(START, 1)
    _, suffix = remainder.split(END, 1)
    return f"{prefix}{START}\n\n{rendered}\n\n{END}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows, config, environment = load_snapshot()
    stale = []
    for language, filename in (("en", "README.md"), ("ko", "README_KR.md")):
        path = ROOT / filename
        expected = update_readme(path, render(language, rows, config, environment))
        if args.check:
            if path.read_text(encoding="utf-8") != expected:
                stale.append(filename)
        else:
            path.write_text(expected, encoding="utf-8")
            print(f"updated {filename}")
    if stale:
        raise SystemExit(f"Stale benchmark tables: {', '.join(stale)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
