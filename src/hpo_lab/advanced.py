"""Advanced HPO workflows used in the second half of the tutorial."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Iterator

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from hpo_lab.config import ExperimentConfig
from hpo_lab.data import (
    DatasetBundle,
    DatasetSplit,
    constrained_num_leaves,
    load_dataset,
    make_model,
    predict_for_metric,
    split_dataset,
)
from hpo_lab.search import evaluate_cv, run_baseline, run_search


@dataclass
class StudyResult:
    study: optuna.Study
    trials: pd.DataFrame
    best_params: dict[str, Any]
    best_cv_loss: float
    holdout_metric: float
    holdout_loss: float


@dataclass
class MultifidelityResult:
    summary: pd.DataFrame
    trials: pd.DataFrame
    studies: dict[str, optuna.Study]


@dataclass
class MultiObjectiveResult:
    study: optuna.Study
    trials: pd.DataFrame
    pareto: pd.DataFrame
    selected_trial: int
    selected_params: dict[str, Any]
    holdout_metric: float
    holdout_loss: float
    predict_milliseconds: float


def _suggest_core_without_resource(
    trial: optuna.Trial,
    config: ExperimentConfig,
) -> dict[str, float | int]:
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            config.min_learning_rate,
            config.max_learning_rate,
            log=True,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            config.min_max_depth,
            config.max_max_depth,
        ),
    }


def run_multifidelity_demo(
    config: ExperimentConfig,
    *,
    dataset_name: str = "diabetes",
    seed: int | None = None,
) -> MultifidelityResult:
    """Compare full-budget trials with real Optuna Hyperband pruning."""

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dataset = load_dataset(dataset_name)
    selected_seed = config.seeds[0] if seed is None else seed
    split = split_dataset(dataset, seed=selected_seed)
    studies: dict[str, optuna.Study] = {}
    summaries: list[dict[str, object]] = []
    trial_rows: list[dict[str, object]] = []

    pruners = {
        "No pruning": optuna.pruners.NopPruner(),
        "Hyperband": optuna.pruners.HyperbandPruner(
            min_resource=config.min_resource,
            max_resource=config.max_resource,
            reduction_factor=config.reduction_factor,
        ),
    }

    for label, pruner in pruners.items():
        sampler = optuna.samplers.TPESampler(
            seed=selected_seed,
            n_startup_trials=max(2, config.advanced_trials // 4),
        )
        study = optuna.create_study(
            study_name=f"multifidelity-{label.lower().replace(' ', '-')}",
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

        def objective(trial: optuna.Trial) -> float:
            params = _suggest_core_without_resource(trial, config)
            steps = (
                (config.max_resource,)
                if label == "No pruning"
                else config.resource_steps
            )
            total_seconds = 0.0
            resource_units = 0
            fit_count = 0
            last_loss = float("inf")
            for (
                resource,
                last_loss,
                std_loss,
                elapsed,
                step_resource_units,
                step_fit_count,
            ) in _incremental_cv_evaluations(
                dataset,
                split,
                params,
                resources=steps,
                seed=selected_seed,
                cv_folds=config.cv_folds,
            ):
                total_seconds += elapsed
                resource_units += step_resource_units
                fit_count += step_fit_count
                trial.set_user_attr("cv_std", std_loss)
                trial.set_user_attr("eval_seconds", total_seconds)
                trial.set_user_attr("fit_count", fit_count)
                trial.set_user_attr("resource_units", resource_units)
                trial.set_user_attr("last_resource", resource)
                trial.report(last_loss, step=resource)
                if resource < config.max_resource and trial.should_prune():
                    raise optuna.TrialPruned()
            return last_loss

        study.optimize(
            objective,
            n_trials=config.advanced_trials,
            timeout=config.timeout_seconds,
            n_jobs=1,
            show_progress_bar=False,
        )
        studies[label] = study
        complete = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if not complete:
            raise RuntimeError(f"{label} produced no completed trials.")

        best_params = {
            **study.best_trial.params,
            "n_estimators": config.max_resource,
        }
        final_model = make_model(dataset, best_params, seed=selected_seed)
        final_model.fit(split.train_X, split.train_y)
        prediction = predict_for_metric(dataset, final_model, split.test_X)
        holdout_metric = dataset.holdout_metric(split.test_y, prediction)

        summaries.append(
            {
                "strategy": label,
                "completed_trials": len(complete),
                "pruned_trials": sum(
                    trial.state == optuna.trial.TrialState.PRUNED
                    for trial in study.trials
                ),
                "best_cv_loss": study.best_value,
                "holdout_metric": holdout_metric,
                "holdout_loss": dataset.metric_to_loss(holdout_metric),
                "resource_units": sum(
                    int(trial.user_attrs.get("resource_units", 0))
                    for trial in study.trials
                ),
                "search_seconds": sum(
                    float(trial.user_attrs.get("eval_seconds", 0.0))
                    for trial in study.trials
                ),
            }
        )
        for trial in study.trials:
            trial_rows.append(
                {
                    "strategy": label,
                    "trial": trial.number + 1,
                    "state": trial.state.name,
                    "last_loss": trial.value,
                    "last_resource": trial.user_attrs.get("last_resource", 0),
                    "resource_units": trial.user_attrs.get("resource_units", 0),
                    "eval_seconds": trial.user_attrs.get("eval_seconds", 0.0),
                    **trial.params,
                }
            )

    return MultifidelityResult(
        summary=pd.DataFrame(summaries).sort_values(
            "holdout_loss",
            ignore_index=True,
        ),
        trials=pd.DataFrame(trial_rows),
        studies=studies,
    )


def run_conditional_search(
    config: ExperimentConfig,
    *,
    dataset_name: str = "diabetes",
    seed: int | None = None,
) -> StudyResult:
    """Optimize a genuinely conditional LightGBM search space."""

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dataset = load_dataset(dataset_name)
    selected_seed = config.seeds[0] if seed is None else seed
    split = split_dataset(dataset, seed=selected_seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=selected_seed,
            n_startup_trials=max(2, config.advanced_trials // 4),
        ),
    )
    study.enqueue_trial({"boosting_type": "gbdt"})
    study.enqueue_trial({"boosting_type": "dart"})

    def objective(trial: optuna.Trial) -> float:
        boosting_type = trial.suggest_categorical(
            "boosting_type",
            ("gbdt", "dart"),
        )
        max_depth = trial.suggest_int(
            "max_depth",
            config.min_max_depth,
            config.max_max_depth,
        )
        params: dict[str, Any] = {
            "boosting_type": boosting_type,
            "max_depth": max_depth,
            "num_leaves": trial.suggest_int(
                "num_leaves",
                2,
                min(128, 2**max_depth),
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                config.min_learning_rate,
                config.max_learning_rate,
                log=True,
            ),
            "n_estimators": trial.suggest_int(
                "n_estimators",
                config.min_n_estimators,
                config.max_n_estimators,
                step=10,
            ),
        }
        if boosting_type == "gbdt":
            params["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
            params["subsample_freq"] = 1
        else:
            params["drop_rate"] = trial.suggest_float("drop_rate", 0.05, 0.3)

        mean_loss, std_loss, elapsed = evaluate_cv(
            dataset,
            split,
            params,
            seed=selected_seed,
            cv_folds=config.cv_folds,
            n_jobs=config.n_jobs,
        )
        trial.set_user_attr("cv_std", std_loss)
        trial.set_user_attr("eval_seconds", elapsed)
        return mean_loss

    study.optimize(
        objective,
        n_trials=config.advanced_trials,
        timeout=config.timeout_seconds,
        n_jobs=1,
        show_progress_bar=False,
    )
    best_params = dict(study.best_params)
    if best_params["boosting_type"] == "gbdt":
        best_params["subsample_freq"] = 1
    model = make_model(dataset, best_params, seed=selected_seed)
    model.fit(split.train_X, split.train_y)
    prediction = predict_for_metric(dataset, model, split.test_X)
    metric = dataset.holdout_metric(split.test_y, prediction)
    trials = pd.DataFrame(
        [
            {
                "trial": trial.number + 1,
                "cv_loss": trial.value,
                "state": trial.state.name,
                **trial.params,
            }
            for trial in study.trials
        ]
    )
    return StudyResult(
        study=study,
        trials=trials,
        best_params=best_params,
        best_cv_loss=float(study.best_value),
        holdout_metric=metric,
        holdout_loss=dataset.metric_to_loss(metric),
    )


def run_multiobjective_search(
    config: ExperimentConfig,
    *,
    dataset_name: str = "diabetes",
    seed: int | None = None,
) -> MultiObjectiveResult:
    """Find a Pareto frontier for predictive loss and model complexity."""

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dataset = load_dataset(dataset_name)
    selected_seed = config.seeds[0] if seed is None else seed
    split = split_dataset(dataset, seed=selected_seed)
    study = optuna.create_study(
        directions=("minimize", "minimize"),
        sampler=optuna.samplers.TPESampler(seed=selected_seed),
    )

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        max_depth = trial.suggest_int(
            "max_depth",
            config.min_max_depth,
            config.max_max_depth,
        )
        num_leaves = trial.suggest_int(
            "num_leaves",
            2,
            constrained_num_leaves(max_depth),
        )
        params = {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                config.min_learning_rate,
                config.max_learning_rate,
                log=True,
            ),
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "n_estimators": trial.suggest_int(
                "n_estimators",
                config.min_n_estimators,
                config.max_n_estimators,
                step=10,
            ),
        }
        mean_loss, std_loss, elapsed = evaluate_cv(
            dataset,
            split,
            params,
            seed=selected_seed,
            cv_folds=config.cv_folds,
            n_jobs=config.n_jobs,
        )
        complexity = float(params["n_estimators"] * num_leaves)
        trial.set_user_attr("cv_std", std_loss)
        trial.set_user_attr("eval_seconds", elapsed)
        return mean_loss, complexity

    study.optimize(
        objective,
        n_trials=config.advanced_trials,
        timeout=config.timeout_seconds,
        n_jobs=1,
        show_progress_bar=False,
    )
    pareto_numbers = {trial.number for trial in study.best_trials}
    rows = [
        {
            "trial": trial.number + 1,
            "cv_loss": trial.values[0],
            "complexity": trial.values[1],
            "is_pareto": trial.number in pareto_numbers,
            **trial.params,
        }
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.values is not None
    ]
    trials = pd.DataFrame(rows)
    pareto = trials[trials["is_pareto"]].copy()
    normalized_loss = _min_max(pareto["cv_loss"].to_numpy())
    normalized_complexity = _min_max(np.log1p(pareto["complexity"].to_numpy()))
    knee_index = int(np.argmin(normalized_loss + normalized_complexity))
    selected = pareto.iloc[knee_index]
    selected_trial = int(selected["trial"])
    selected_frozen_trial = next(
        trial
        for trial in study.trials
        if trial.number == selected_trial - 1
    )
    selected_params = dict(selected_frozen_trial.params)

    model = make_model(dataset, selected_params, seed=selected_seed)
    model.fit(split.train_X, split.train_y)
    started = perf_counter()
    prediction = predict_for_metric(dataset, model, split.test_X)
    predict_milliseconds = (perf_counter() - started) * 1_000
    metric = dataset.holdout_metric(split.test_y, prediction)
    return MultiObjectiveResult(
        study=study,
        trials=trials,
        pareto=pareto.reset_index(drop=True),
        selected_trial=selected_trial,
        selected_params=selected_params,
        holdout_metric=metric,
        holdout_loss=dataset.metric_to_loss(metric),
        predict_milliseconds=predict_milliseconds,
    )


def run_nested_validation(
    config: ExperimentConfig,
    *,
    dataset_name: str = "diabetes",
    seed: int | None = None,
) -> pd.DataFrame:
    """Estimate tuning benefit without reusing an outer validation fold."""

    dataset = load_dataset(dataset_name)
    selected_seed = config.seeds[0] if seed is None else seed
    if dataset.task == "classification":
        outer_cv = StratifiedKFold(
            n_splits=config.nested_outer_folds,
            shuffle=True,
            random_state=selected_seed,
        )
    else:
        outer_cv = KFold(
            n_splits=config.nested_outer_folds,
            shuffle=True,
            random_state=selected_seed,
        )
    nested_config = replace(
        config,
        n_trials=config.nested_trials,
        cv_folds=config.nested_inner_folds,
        methods=("TPE",),
    )
    rows: list[dict[str, object]] = []
    for fold, (train_index, test_index) in enumerate(
        outer_cv.split(dataset.X, dataset.y),
        start=1,
    ):
        fold_seed = selected_seed + fold
        split = DatasetSplit(
            train_X=dataset.X.iloc[train_index],
            test_X=dataset.X.iloc[test_index],
            train_y=dataset.y.iloc[train_index],
            test_y=dataset.y.iloc[test_index],
        )
        baseline = run_baseline(dataset, split, nested_config, seed=fold_seed)
        tuned = run_search("TPE", dataset, split, nested_config, seed=fold_seed)
        rows.append(
            {
                "outer_fold": fold,
                "baseline_holdout_metric": baseline.holdout_metric,
                "baseline_holdout_loss": baseline.holdout_loss,
                "tuned_holdout_metric": tuned.holdout_metric,
                "tuned_holdout_loss": tuned.holdout_loss,
                "selected_inner_cv_loss": tuned.best_cv_loss,
                "best_params": tuned.best_params,
            }
        )
    return pd.DataFrame(rows)


def _min_max(values: np.ndarray) -> np.ndarray:
    spread = values.max() - values.min()
    if spread == 0:
        return np.zeros_like(values, dtype=float)
    return (values - values.min()) / spread


def _incremental_cv_evaluations(
    dataset: DatasetBundle,
    split: DatasetSplit,
    params: dict[str, float | int],
    *,
    resources: tuple[int, ...],
    seed: int,
    cv_folds: int,
) -> Iterator[tuple[int, float, float, float, int, int]]:
    """Grow each fold's LightGBM booster so later rungs reuse earlier work."""

    if dataset.task == "classification":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    folds = list(cv.split(split.train_X, split.train_y))
    boosters: list[Any | None] = [None] * len(folds)
    previous_resource = 0

    for resource in resources:
        additional_trees = resource - previous_resource
        if additional_trees <= 0:
            raise ValueError("Resource rungs must be strictly increasing.")
        losses: list[float] = []
        started = perf_counter()
        for fold_index, (train_index, valid_index) in enumerate(folds):
            model = make_model(
                dataset,
                {**params, "n_estimators": additional_trees},
                seed=seed,
            )
            fit_kwargs = {}
            if boosters[fold_index] is not None:
                fit_kwargs["init_model"] = boosters[fold_index]
            model.fit(
                split.train_X.iloc[train_index],
                split.train_y.iloc[train_index],
                **fit_kwargs,
            )
            boosters[fold_index] = model.booster_
            prediction = predict_for_metric(
                dataset,
                model,
                split.train_X.iloc[valid_index],
            )
            metric = dataset.holdout_metric(
                split.train_y.iloc[valid_index],
                prediction,
            )
            losses.append(dataset.metric_to_loss(metric))
        elapsed = perf_counter() - started
        previous_resource = resource
        yield (
            resource,
            float(np.mean(losses)),
            float(np.std(losses, ddof=0)),
            elapsed,
            additional_trees * cv_folds,
            cv_folds,
        )
