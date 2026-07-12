"""Matplotlib visualizations designed for honest HPO comparison."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from hpo_lab.advanced import MultifidelityResult, MultiObjectiveResult
from hpo_lab.search import BenchmarkResult


def plot_convergence(result: BenchmarkResult):
    """Plot mean best-so-far loss with a 95% interval across seeds."""

    history = result.history
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for method in result.config.methods:
        method_frame = history[history["method"] == method]
        if method_frame.empty:
            continue
        pivot = method_frame.pivot(
            index="trial",
            columns="seed",
            values="best_cv_loss",
        )
        mean = pivot.mean(axis=1)
        axis.plot(mean.index, mean, label=method, linewidth=2)
        if pivot.shape[1] > 1:
            critical_value = t.ppf(0.975, df=pivot.shape[1] - 1)
            ci = (
                critical_value
                * pivot.std(axis=1, ddof=1)
                / np.sqrt(pivot.shape[1])
            )
            axis.fill_between(mean.index, mean - ci, mean + ci, alpha=0.12)

    baseline = result.runs_frame.query("method == 'Baseline'")["best_cv_loss"].mean()
    axis.axhline(baseline, color="black", linestyle="--", label="Baseline")
    axis.set(
        xlabel="Completed trials",
        ylabel=f"Best CV {result.dataset.loss_name} (lower is better)",
        title="Convergence under an equal trial budget",
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    figure.tight_layout()
    return figure, axis


def plot_quality_vs_time(result: BenchmarkResult):
    """Show the quality/cost trade-off instead of reporting a winner alone."""

    summary = result.summary
    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.scatter(
        summary["search_seconds_mean"],
        summary["holdout_loss_mean"],
        s=90,
    )
    for row in summary.itertuples():
        axis.annotate(
            row.method,
            (row.search_seconds_mean, row.holdout_loss_mean),
            xytext=(6, 5),
            textcoords="offset points",
        )
    axis.set(
        xlabel="Mean search time (seconds)",
        ylabel=f"Mean holdout {result.dataset.loss_name} (lower is better)",
        title="Optimization quality versus search cost",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure, axis


def plot_seed_stability(result: BenchmarkResult):
    """Expose sensitivity to data splits and random seeds."""

    frame = result.runs_frame
    methods = frame["method"].drop_duplicates().tolist()
    values = [
        frame.loc[frame["method"] == method, "holdout_loss"].to_numpy()
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.boxplot(values, tick_labels=methods, showmeans=True)
    axis.set(
        ylabel=f"Holdout {result.dataset.loss_name} (lower is better)",
        title="Stability across data splits and random seeds",
    )
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return figure, axis


def plot_pruning_budget(result: MultifidelityResult):
    """Compare resource consumption and completed/pruned trials."""

    summary = result.summary
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(summary["strategy"], summary["resource_units"])
    axes[0].set(
        ylabel="Tree-fit resource units",
        title="Compute consumed",
    )
    completed = summary["completed_trials"].to_numpy()
    pruned = summary["pruned_trials"].to_numpy()
    axes[1].bar(summary["strategy"], completed, label="Completed")
    axes[1].bar(
        summary["strategy"],
        pruned,
        bottom=completed,
        label="Pruned",
    )
    axes[1].set(ylabel="Trials", title="Trial outcomes")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    return figure, axes


def plot_pareto_front(result: MultiObjectiveResult):
    """Plot all trials and highlight the non-dominated frontier."""

    figure, axis = plt.subplots(figsize=(8, 5.5))
    axis.scatter(
        result.trials["complexity"],
        result.trials["cv_loss"],
        alpha=0.45,
        label="Dominated",
    )
    axis.scatter(
        result.pareto["complexity"],
        result.pareto["cv_loss"],
        s=90,
        label="Pareto frontier",
    )
    selected = result.trials[result.trials["trial"] == result.selected_trial].iloc[0]
    axis.scatter(
        [selected["complexity"]],
        [selected["cv_loss"]],
        marker="*",
        s=220,
        label="Selected knee",
    )
    axis.set_xscale("log")
    axis.set(
        xlabel="Model complexity proxy (estimators x leaves, log scale)",
        ylabel="CV loss (lower is better)",
        title="Accuracy/complexity Pareto frontier",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    return figure, axis
