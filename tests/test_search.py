from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import optuna
import pytest

from hpo_lab import (
    ExperimentConfig,
    load_dataset,
    run_benchmark,
    run_search,
    split_dataset,
)
from hpo_lab.plots import plot_convergence


def test_all_core_optimizers_share_trial_and_fit_budget(
    smoke_config: ExperimentConfig,
) -> None:
    methods = ("Grid", "Random", "TPE", "GP", "CMA-ES")
    result = run_benchmark(smoke_config, methods=methods)

    assert set(result.runs_frame["method"]) == {"Baseline", *methods}
    for run in result.runs:
        if run.method == "Baseline":
            continue
        assert len(run.history) == smoke_config.n_trials
        assert run.fit_count == smoke_config.n_trials * smoke_config.cv_folds + 1
        assert run.best_cv_loss == run.history["cv_loss"].min()
        assert run.holdout_loss >= 0

    assert result.summary["holdout_loss_mean"].notna().all()
    assert result.summary["holdout_loss_ci95"].isna().all()
    assert result.history.groupby("method")["trial"].nunique().eq(
        smoke_config.n_trials
    ).all()


def test_benchmark_exports_machine_readable_artifacts(
    smoke_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    result = run_benchmark(smoke_config, methods=("Random",))
    output = result.save(tmp_path)

    expected = {
        "best_params.json",
        "config.json",
        "history.csv",
        "runs.csv",
        "summary.csv",
    }
    assert expected == {path.name for path in output.iterdir()}
    assert not pd.read_csv(output / "history.csv").empty
    manifest = json.loads((output / "config.json").read_text())
    assert manifest["dataset"]["name"] == "Diabetes regression"
    assert manifest["experiment"]["methods"] == ["Random"]
    assert result.config.methods == ("Random",)
    figure, _ = plot_convergence(result)
    figure.clear()


def test_persistent_study_resumes_to_target_total(
    smoke_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    dataset = load_dataset("diabetes")
    split = split_dataset(dataset, seed=42)
    storage = f"sqlite:///{tmp_path / 'study.db'}"

    first = run_search(
        "TPE",
        dataset,
        split,
        smoke_config,
        seed=42,
        storage=storage,
        study_name="resume-test",
    )
    second = run_search(
        "TPE",
        dataset,
        split,
        smoke_config,
        seed=42,
        storage=storage,
        study_name="resume-test",
    )

    assert len(first.study.trials) == smoke_config.n_trials
    assert len(second.study.trials) == smoke_config.n_trials
    assert second.best_params == first.best_params


def test_random_resume_does_not_replay_seeded_trials(
    smoke_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    dataset = load_dataset("diabetes")
    split = split_dataset(dataset, seed=42)
    storage = f"sqlite:///{tmp_path / 'random.db'}"
    first_config = replace(smoke_config, n_trials=4)

    first = run_search(
        "Random",
        dataset,
        split,
        first_config,
        seed=42,
        storage=storage,
        study_name="random-resume",
    )
    first_params = {
        tuple(sorted(trial.params.items()))
        for trial in first.study.trials
    }
    resumed = run_search(
        "Random",
        dataset,
        split,
        smoke_config,
        seed=42,
        storage=storage,
        study_name="random-resume",
    )

    added_params = {
        tuple(sorted(trial.params.items()))
        for trial in resumed.study.trials[4:]
    }
    assert first_params.isdisjoint(added_params)
    assert len(resumed.study.trials) == smoke_config.n_trials


def test_resume_rejects_incompatible_objective(
    smoke_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    dataset = load_dataset("diabetes")
    split = split_dataset(dataset, seed=42)
    storage = f"sqlite:///{tmp_path / 'fingerprint.db'}"
    run_search(
        "TPE",
        dataset,
        split,
        smoke_config,
        seed=42,
        storage=storage,
        study_name="fingerprint-test",
    )

    with pytest.raises(ValueError, match="objective does not match"):
        run_search(
            "TPE",
            dataset,
            split,
            replace(smoke_config, cv_folds=3),
            seed=42,
            storage=storage,
            study_name="fingerprint-test",
        )


def test_resume_rejects_unaccounted_failed_trials(
    smoke_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    dataset = load_dataset("diabetes")
    split = split_dataset(dataset, seed=42)
    storage = f"sqlite:///{tmp_path / 'failed.db'}"
    partial = replace(smoke_config, n_trials=4)
    run = run_search(
        "TPE",
        dataset,
        split,
        partial,
        seed=42,
        storage=storage,
        study_name="failed-trial-test",
    )
    failed_trial = run.study.ask()
    run.study.tell(failed_trial, state=optuna.trial.TrialState.FAIL)

    with pytest.raises(ValueError, match="cannot resume"):
        run_search(
            "TPE",
            dataset,
            split,
            smoke_config,
            seed=42,
            storage=storage,
            study_name="failed-trial-test",
        )


def test_concurrent_sqlite_resume_is_serialized(
    smoke_config: ExperimentConfig,
    tmp_path: Path,
) -> None:
    dataset = load_dataset("diabetes")
    split = split_dataset(dataset, seed=42)
    storage = f"sqlite:///{tmp_path / 'concurrent.db'}"
    target = replace(smoke_config, n_trials=4)

    def execute():
        return run_search(
            "TPE",
            dataset,
            split,
            target,
            seed=42,
            storage=storage,
            study_name="concurrent-test",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: execute(), range(2)))

    assert all(len(result.study.trials) == target.n_trials for result in results)
    assert all(len(result.history) == target.n_trials for result in results)


@pytest.mark.parametrize("method", ("Random", "GP"))
def test_timeout_never_returns_an_incomplete_equal_budget_result(
    smoke_config: ExperimentConfig,
    method: str,
) -> None:
    dataset = load_dataset("diabetes")
    split = split_dataset(dataset, seed=42)
    timed = replace(smoke_config, timeout_seconds=1e-12)

    with pytest.raises(RuntimeError, match="completed"):
        run_search(method, dataset, split, timed, seed=42)


def test_duplicate_method_selection_is_rejected(
    smoke_config: ExperimentConfig,
) -> None:
    with pytest.raises(ValueError, match="Duplicate methods"):
        run_benchmark(smoke_config, methods=("Random", "Random"))


def test_classification_uses_auc_without_changing_engine(
    smoke_config: ExperimentConfig,
) -> None:
    result = run_benchmark(
        smoke_config,
        dataset_name="breast_cancer",
        methods=("TPE",),
    )

    assert result.dataset.metric_name == "ROC AUC"
    assert result.runs_frame["holdout_metric"].between(0, 1).all()
    assert result.runs_frame["holdout_loss"].between(0, 1).all()
