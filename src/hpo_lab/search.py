"""Fair black-box HPO benchmarks using a shared objective and compute budget."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field, replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from filelock import FileLock
import numpy as np
import optuna
import pandas as pd
from scipy.stats import norm, t
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from hpo_lab.config import ExperimentConfig
from hpo_lab.data import (
    DatasetBundle,
    DatasetSplit,
    load_dataset,
    make_model,
    predict_for_metric,
    split_dataset,
)

METHODS = ("Grid", "Random", "TPE", "GP", "CMA-ES")


@dataclass
class SearchRun:
    method: str
    seed: int
    best_params: dict[str, float | int]
    best_cv_loss: float
    best_cv_std: float
    holdout_metric: float
    holdout_loss: float
    search_seconds: float
    evaluation_seconds: float
    optimizer_overhead_seconds: float
    train_seconds: float
    predict_milliseconds: float
    fit_count: int
    history: pd.DataFrame
    study: optuna.Study | None = field(repr=False, default=None)

    def as_record(self) -> dict[str, object]:
        return {
            "method": self.method,
            "seed": self.seed,
            "best_cv_loss": self.best_cv_loss,
            "best_cv_std": self.best_cv_std,
            "holdout_metric": self.holdout_metric,
            "holdout_loss": self.holdout_loss,
            "search_seconds": self.search_seconds,
            "evaluation_seconds": self.evaluation_seconds,
            "optimizer_overhead_seconds": self.optimizer_overhead_seconds,
            "train_seconds": self.train_seconds,
            "predict_milliseconds": self.predict_milliseconds,
            "fit_count": self.fit_count,
            "best_params": json.dumps(self.best_params, sort_keys=True),
        }


@dataclass
class BenchmarkResult:
    dataset: DatasetBundle
    config: ExperimentConfig
    runs: list[SearchRun]

    @property
    def runs_frame(self) -> pd.DataFrame:
        return pd.DataFrame([run.as_record() for run in self.runs])

    @property
    def history(self) -> pd.DataFrame:
        frames = [run.history for run in self.runs if not run.history.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @property
    def summary(self) -> pd.DataFrame:
        frame = self.runs_frame
        rows: list[dict[str, object]] = []
        for method, group in frame.groupby("method", sort=False):
            count = len(group)
            cv_ci = _ci95(group["best_cv_loss"])
            holdout_ci = _ci95(group["holdout_loss"])
            rows.append(
                {
                    "method": method,
                    "runs": count,
                    "cv_loss_mean": group["best_cv_loss"].mean(),
                    "cv_loss_ci95": cv_ci,
                    "holdout_metric_mean": group["holdout_metric"].mean(),
                    "holdout_loss_mean": group["holdout_loss"].mean(),
                    "holdout_loss_ci95": holdout_ci,
                    "search_seconds_mean": group["search_seconds"].mean(),
                    "optimizer_overhead_seconds_mean": group[
                        "optimizer_overhead_seconds"
                    ].mean(),
                    "fit_count_mean": group["fit_count"].mean(),
                }
            )
        return pd.DataFrame(rows).sort_values(
            ["holdout_loss_mean", "search_seconds_mean"],
            ignore_index=True,
        )

    @property
    def studies(self) -> dict[tuple[str, int], optuna.Study]:
        return {
            (run.method, run.seed): run.study
            for run in self.runs
            if run.study is not None
        }

    def save(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.runs_frame.to_csv(output_path / "runs.csv", index=False)
        self.history.to_csv(output_path / "history.csv", index=False)
        self.summary.to_csv(output_path / "summary.csv", index=False)
        manifest = {
            "dataset": {
                "name": self.dataset.name,
                "task": self.dataset.task,
                "scoring": self.dataset.scoring,
                "metric_name": self.dataset.metric_name,
                "loss_name": self.dataset.loss_name,
            },
            "experiment": self.config.as_dict(),
        }
        (output_path / "config.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        best_params = {
            f"{run.method}/seed-{run.seed}": run.best_params
            for run in self.runs
        }
        (output_path / "best_params.json").write_text(
            json.dumps(best_params, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return output_path


def available_methods() -> tuple[str, ...]:
    return METHODS


def _ci95(values: pd.Series) -> float:
    if len(values) < 2:
        return float("nan")
    critical_value = t.ppf(0.975, df=len(values) - 1)
    return float(critical_value * values.std(ddof=1) / np.sqrt(len(values)))


def _cv_splitter(dataset: DatasetBundle, folds: int, seed: int):
    if dataset.task == "classification":
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return KFold(n_splits=folds, shuffle=True, random_state=seed)


def evaluate_cv(
    dataset: DatasetBundle,
    split: DatasetSplit,
    params: dict[str, float | int] | None,
    *,
    seed: int,
    cv_folds: int,
    n_jobs: int,
) -> tuple[float, float, float]:
    """Evaluate one configuration on the same deterministic CV folds."""

    model = make_model(dataset, params, seed=seed)
    cv = _cv_splitter(dataset, cv_folds, seed)
    started = perf_counter()
    scores = cross_val_score(
        model,
        split.train_X,
        split.train_y,
        scoring=dataset.scoring,
        cv=cv,
        n_jobs=n_jobs,
        error_score="raise",
    )
    elapsed = perf_counter() - started
    losses = dataset.scores_to_losses(np.asarray(scores))
    return float(losses.mean()), float(losses.std(ddof=0)), elapsed


def _suggest_params(
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
        "n_estimators": trial.suggest_int(
            "n_estimators",
            config.min_n_estimators,
            config.max_n_estimators,
            step=10,
        ),
    }


def _make_sampler(
    method: str,
    config: ExperimentConfig,
    seed: int,
) -> optuna.samplers.BaseSampler:
    startup = max(2, min(10, config.n_trials // 3))
    if method == "Grid":
        return optuna.samplers.GridSampler(config.grid_space, seed=seed)
    if method == "Random":
        return optuna.samplers.RandomSampler(seed=seed)
    if method == "TPE":
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=startup,
        )
    if method == "CMA-ES":
        return optuna.samplers.CmaEsSampler(
            seed=seed,
            n_startup_trials=startup,
            warn_independent_sampling=False,
        )
    raise ValueError(f"Unknown method {method!r}; choose one of {METHODS}.")


def _completed_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]


def run_search(
    method: str,
    dataset: DatasetBundle,
    split: DatasetSplit,
    config: ExperimentConfig,
    *,
    seed: int,
    storage: str | None = None,
    study_name: str | None = None,
) -> SearchRun:
    """Run one optimizer without touching the holdout split during selection."""

    with _study_lock(storage):
        return _run_search(
            method,
            dataset,
            split,
            config,
            seed=seed,
            storage=storage,
            study_name=study_name,
        )


def _run_search(
    method: str,
    dataset: DatasetBundle,
    split: DatasetSplit,
    config: ExperimentConfig,
    *,
    seed: int,
    storage: str | None,
    study_name: str | None,
) -> SearchRun:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose one of {METHODS}.")
    if method == "GP":
        if storage is not None:
            raise ValueError("Persistent storage is currently supported for Optuna samplers only.")
        return _run_gp_search(dataset, split, config, seed=seed)

    sampler = _make_sampler(method, config, seed)
    name = study_name or f"{dataset.name}-{method}-seed-{seed}"
    study = optuna.create_study(
        study_name=name,
        direction="minimize",
        sampler=sampler,
        storage=storage,
        load_if_exists=storage is not None,
    )
    fingerprint, manifest = _objective_fingerprint(
        method,
        dataset,
        split,
        config,
        seed=seed,
    )
    _validate_study_fingerprint(study, fingerprint, manifest)
    _reject_noncomplete_trials(study)
    existing_trials = len(study.trials)
    existing = len(_completed_trials(study))
    if existing > config.n_trials:
        raise ValueError(
            f"Study already has {existing} completed trials, which exceeds "
            f"the target total of {config.n_trials}."
        )
    if storage is not None and existing_trials and method != "Grid":
        study = optuna.load_study(
            study_name=name,
            storage=storage,
            sampler=_make_sampler(
                method,
                config,
                _resume_seed(seed, existing_trials),
            ),
        )

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, config)
        mean_loss, std_loss, elapsed = evaluate_cv(
            dataset,
            split,
            params,
            seed=seed,
            cv_folds=config.cv_folds,
            n_jobs=config.n_jobs,
        )
        trial.set_user_attr("cv_std", std_loss)
        trial.set_user_attr("eval_seconds", elapsed)
        trial.set_user_attr("fit_count", config.cv_folds)
        return mean_loss

    remaining = max(0, config.n_trials - existing)
    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=config.timeout_seconds,
            n_jobs=1,
            show_progress_bar=False,
        )

    complete = _completed_trials(study)
    if len(complete) != config.n_trials:
        raise RuntimeError(
            f"{method} completed {len(complete)}/{config.n_trials} trials. "
            "The timeout ended the fixed-budget search; resume the study or "
            "increase timeout_seconds before comparing methods."
        )

    best_trial = study.best_trial
    best_params = {
        key: _python_scalar(value)
        for key, value in best_trial.params.items()
    }
    final_model = make_model(dataset, best_params, seed=seed)
    train_started = perf_counter()
    final_model.fit(split.train_X, split.train_y)
    train_seconds = perf_counter() - train_started
    predict_started = perf_counter()
    prediction = predict_for_metric(dataset, final_model, split.test_X)
    predict_milliseconds = (perf_counter() - predict_started) * 1_000
    metric = dataset.holdout_metric(split.test_y, prediction)

    history_rows: list[dict[str, object]] = []
    cumulative = 0.0
    best_so_far = float("inf")
    for completed_number, trial in enumerate(complete, start=1):
        eval_seconds = float(trial.user_attrs.get("eval_seconds", 0.0))
        cumulative += eval_seconds
        value = float(trial.value)
        best_so_far = min(best_so_far, value)
        history_rows.append(
            {
                "method": method,
                "seed": seed,
                "trial": completed_number,
                "optuna_trial_id": trial.number,
                "cv_loss": value,
                "best_cv_loss": best_so_far,
                "cv_std": float(trial.user_attrs.get("cv_std", np.nan)),
                "eval_seconds": eval_seconds,
                "cumulative_seconds": cumulative,
                "fit_count": int(trial.user_attrs.get("fit_count", config.cv_folds)),
                **trial.params,
            }
        )

    evaluation_seconds = sum(
        float(trial.user_attrs.get("eval_seconds", 0.0))
        for trial in complete
    )
    search_seconds = sum(
        trial.duration.total_seconds()
        for trial in complete
        if trial.duration is not None
    )
    return SearchRun(
        method=method,
        seed=seed,
        best_params=best_params,
        best_cv_loss=float(best_trial.value),
        best_cv_std=float(best_trial.user_attrs.get("cv_std", np.nan)),
        holdout_metric=metric,
        holdout_loss=dataset.metric_to_loss(metric),
        search_seconds=search_seconds,
        evaluation_seconds=evaluation_seconds,
        optimizer_overhead_seconds=max(0.0, search_seconds - evaluation_seconds),
        train_seconds=train_seconds,
        predict_milliseconds=predict_milliseconds,
        fit_count=sum(
            int(trial.user_attrs.get("fit_count", config.cv_folds))
            for trial in complete
        )
        + 1,
        history=pd.DataFrame(history_rows),
        study=study,
    )


def run_baseline(
    dataset: DatasetBundle,
    split: DatasetSplit,
    config: ExperimentConfig,
    *,
    seed: int,
) -> SearchRun:
    cv_loss, cv_std, search_seconds = evaluate_cv(
        dataset,
        split,
        None,
        seed=seed,
        cv_folds=config.cv_folds,
        n_jobs=config.n_jobs,
    )
    model = make_model(dataset, None, seed=seed)
    train_started = perf_counter()
    model.fit(split.train_X, split.train_y)
    train_seconds = perf_counter() - train_started
    predict_started = perf_counter()
    prediction = predict_for_metric(dataset, model, split.test_X)
    predict_milliseconds = (perf_counter() - predict_started) * 1_000
    metric = dataset.holdout_metric(split.test_y, prediction)
    return SearchRun(
        method="Baseline",
        seed=seed,
        best_params={},
        best_cv_loss=cv_loss,
        best_cv_std=cv_std,
        holdout_metric=metric,
        holdout_loss=dataset.metric_to_loss(metric),
        search_seconds=search_seconds,
        evaluation_seconds=search_seconds,
        optimizer_overhead_seconds=0.0,
        train_seconds=train_seconds,
        predict_milliseconds=predict_milliseconds,
        fit_count=config.cv_folds + 1,
        history=pd.DataFrame(),
    )


def run_benchmark(
    config: ExperimentConfig,
    *,
    dataset_name: str = "diabetes",
    methods: Iterable[str] | None = None,
) -> BenchmarkResult:
    """Run baseline and optimizers over identical dataset seeds and budgets."""

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dataset = load_dataset(dataset_name)
    selected_methods = config.methods if methods is None else tuple(methods)
    if len(selected_methods) != len(set(selected_methods)):
        raise ValueError("Duplicate methods are not allowed.")
    unknown = set(selected_methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    effective_config = replace(config, methods=selected_methods)
    runs: list[SearchRun] = []
    for seed in effective_config.seeds:
        split = split_dataset(dataset, seed=seed)
        runs.append(run_baseline(dataset, split, effective_config, seed=seed))
        for method in selected_methods:
            runs.append(
                run_search(
                    method,
                    dataset,
                    split,
                    effective_config,
                    seed=seed,
                )
            )
    return BenchmarkResult(dataset=dataset, config=effective_config, runs=runs)


def _run_gp_search(
    dataset: DatasetBundle,
    split: DatasetSplit,
    config: ExperimentConfig,
    *,
    seed: int,
) -> SearchRun:
    """Sequential Gaussian-process search with Expected Improvement."""

    rng = np.random.default_rng(seed)
    startup_trials = max(2, min(8, config.n_trials // 3))
    normalized_points: list[np.ndarray] = []
    losses: list[float] = []
    history_rows: list[dict[str, object]] = []
    cumulative_seconds = 0.0
    best_so_far = float("inf")
    started = perf_counter()

    for trial_number in range(1, config.n_trials + 1):
        if (
            trial_number > 1
            and config.timeout_seconds is not None
            and perf_counter() - started >= config.timeout_seconds
        ):
            break
        if trial_number <= startup_trials:
            point = rng.random(3)
        else:
            model = GaussianProcessRegressor(
                kernel=Matern(length_scale=np.ones(3), nu=2.5),
                alpha=1e-5,
                normalize_y=True,
                optimizer=None,
                random_state=seed,
            )
            model.fit(np.vstack(normalized_points), np.asarray(losses))
            candidates = rng.random((2_048, 3))
            mean, std = model.predict(candidates, return_std=True)
            improvement = min(losses) - mean - 0.01
            safe_std = np.maximum(std, 1e-12)
            z_score = improvement / safe_std
            expected_improvement = (
                improvement * norm.cdf(z_score)
                + safe_std * norm.pdf(z_score)
            )
            point = candidates[int(np.argmax(expected_improvement))]

        params = _decode_gp_point(point, config)
        mean_loss, std_loss, eval_seconds = evaluate_cv(
            dataset,
            split,
            params,
            seed=seed,
            cv_folds=config.cv_folds,
            n_jobs=config.n_jobs,
        )
        normalized_points.append(point)
        losses.append(mean_loss)
        cumulative_seconds += eval_seconds
        best_so_far = min(best_so_far, mean_loss)
        history_rows.append(
            {
                "method": "GP",
                "seed": seed,
                "trial": trial_number,
                "cv_loss": mean_loss,
                "best_cv_loss": best_so_far,
                "cv_std": std_loss,
                "eval_seconds": eval_seconds,
                "cumulative_seconds": cumulative_seconds,
                "fit_count": config.cv_folds,
                **params,
            }
        )

    if len(losses) < config.n_trials:
        raise RuntimeError(
            f"GP completed {len(losses)}/{config.n_trials} trials. "
            "The timeout ended the fixed-budget search; increase "
            "timeout_seconds before comparing methods."
        )
    search_seconds = perf_counter() - started
    best_index = int(np.argmin(losses))
    best_row = history_rows[best_index]
    best_params = {
        key: best_row[key]
        for key in ("learning_rate", "max_depth", "n_estimators")
    }
    final_model = make_model(dataset, best_params, seed=seed)
    train_started = perf_counter()
    final_model.fit(split.train_X, split.train_y)
    train_seconds = perf_counter() - train_started
    predict_started = perf_counter()
    prediction = predict_for_metric(dataset, final_model, split.test_X)
    predict_milliseconds = (perf_counter() - predict_started) * 1_000
    metric = dataset.holdout_metric(split.test_y, prediction)
    return SearchRun(
        method="GP",
        seed=seed,
        best_params=best_params,
        best_cv_loss=losses[best_index],
        best_cv_std=float(best_row["cv_std"]),
        holdout_metric=metric,
        holdout_loss=dataset.metric_to_loss(metric),
        search_seconds=search_seconds,
        evaluation_seconds=cumulative_seconds,
        optimizer_overhead_seconds=max(0.0, search_seconds - cumulative_seconds),
        train_seconds=train_seconds,
        predict_milliseconds=predict_milliseconds,
        fit_count=config.n_trials * config.cv_folds + 1,
        history=pd.DataFrame(history_rows),
    )


def _decode_gp_point(
    point: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, float | int]:
    log_min = np.log(config.min_learning_rate)
    log_max = np.log(config.max_learning_rate)
    learning_rate = float(np.exp(log_min + point[0] * (log_max - log_min)))
    max_depth = int(
        np.rint(
            config.min_max_depth
            + point[1] * (config.max_max_depth - config.min_max_depth)
        )
    )
    estimator_value = (
        config.min_n_estimators
        + point[2] * (config.max_n_estimators - config.min_n_estimators)
    )
    n_estimators = int(np.rint(estimator_value / 10) * 10)
    return {
        "learning_rate": learning_rate,
        "max_depth": int(
            np.clip(max_depth, config.min_max_depth, config.max_max_depth)
        ),
        "n_estimators": int(
            np.clip(
                n_estimators,
                config.min_n_estimators,
                config.max_n_estimators,
            )
        ),
    }


def _python_scalar(value: Any) -> float | int:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    raise TypeError(f"Unsupported hyperparameter type: {type(value).__name__}")


def _resume_seed(seed: int, existing_trials: int) -> int:
    return int((seed + existing_trials * 1_000_003) % (2**32 - 1))


def _study_lock(storage: str | None):
    if storage is None:
        return nullcontext()
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        raise ValueError(
            "Persistent fixed-budget runs currently support SQLite URLs only."
        )
    database = storage[len(prefix):]
    if database == ":memory:":
        raise ValueError("Use a file-backed SQLite database for persistent runs.")
    database_path = Path(database).expanduser().resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(f"{database_path}.lock")


def _reject_noncomplete_trials(study: optuna.Study) -> None:
    unexpected = [
        trial
        for trial in study.trials
        if trial.state != optuna.trial.TrialState.COMPLETE
    ]
    if unexpected:
        states = ", ".join(
            f"{trial.number}:{trial.state.name}"
            for trial in unexpected
        )
        raise ValueError(
            "Persistent fixed-budget studies cannot resume with failed, "
            f"pruned, waiting, or running trials ({states}). Use a new study "
            "name so cost and convergence accounting remain complete."
        )


def _objective_fingerprint(
    method: str,
    dataset: DatasetBundle,
    split: DatasetSplit,
    config: ExperimentConfig,
    *,
    seed: int,
) -> tuple[str, dict[str, object]]:
    manifest: dict[str, object] = {
        "schema": 1,
        "method": method,
        "dataset": dataset.name,
        "task": dataset.task,
        "scoring": dataset.scoring,
        "seed": seed,
        "cv_folds": config.cv_folds,
        "search_bounds": {
            "learning_rate": [
                config.min_learning_rate,
                config.max_learning_rate,
            ],
            "max_depth": [config.min_max_depth, config.max_max_depth],
            "n_estimators": [
                config.min_n_estimators,
                config.max_n_estimators,
                10,
            ],
        },
        "grid_space": config.grid_space if method == "Grid" else None,
        "model_stack": {
            "lightgbm": version("lightgbm"),
            "scikit-learn": version("scikit-learn"),
        },
        "split_digest": _split_digest(split),
    }
    serialized = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), manifest


def _split_digest(split: DatasetSplit) -> str:
    digest = hashlib.sha256()
    for value in (
        split.train_X,
        split.train_y,
        split.test_X,
        split.test_y,
    ):
        digest.update(
            pd.util.hash_pandas_object(value, index=True)
            .to_numpy()
            .tobytes()
        )
        if isinstance(value, pd.DataFrame):
            digest.update(
                json.dumps(
                    [str(column) for column in value.columns],
                    separators=(",", ":"),
                ).encode("utf-8")
            )
    return digest.hexdigest()


def _validate_study_fingerprint(
    study: optuna.Study,
    fingerprint: str,
    manifest: dict[str, object],
) -> None:
    stored = study.user_attrs.get("objective_fingerprint")
    if stored is None:
        if study.trials:
            raise ValueError(
                "Existing study has no objective fingerprint and cannot be "
                "resumed safely. Use a new study name or storage database."
            )
        study.set_user_attr("objective_fingerprint", fingerprint)
        study.set_user_attr("objective_manifest", manifest)
        return
    if stored != fingerprint:
        raise ValueError(
            "Stored study objective does not match the current dataset split, "
            "metric, CV folds, seed, method, or search bounds. Use a new study "
            "name instead of mixing incomparable trials."
        )
