"""Dataset loading, splitting, metrics, and deterministic model creation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

Task = Literal["regression", "classification"]


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    task: Task
    X: pd.DataFrame
    y: pd.Series
    scoring: str
    metric_name: str
    loss_name: str

    def scores_to_losses(self, scores: np.ndarray) -> np.ndarray:
        if self.task == "regression":
            return -scores
        return 1.0 - scores

    def holdout_metric(self, y_true: np.ndarray, prediction: np.ndarray) -> float:
        if self.task == "regression":
            return float(mean_squared_error(y_true, prediction))
        return float(roc_auc_score(y_true, prediction))

    def metric_to_loss(self, metric: float) -> float:
        return metric if self.task == "regression" else 1.0 - metric


@dataclass(frozen=True)
class DatasetSplit:
    train_X: pd.DataFrame
    test_X: pd.DataFrame
    train_y: pd.Series
    test_y: pd.Series


def load_dataset(name: str = "diabetes") -> DatasetBundle:
    """Load the regression tutorial data or an optional classification challenge."""

    if name == "diabetes":
        dataset = load_diabetes(as_frame=True)
        return DatasetBundle(
            name="Diabetes regression",
            task="regression",
            X=dataset.data,
            y=dataset.target,
            scoring="neg_mean_squared_error",
            metric_name="MSE",
            loss_name="MSE",
        )
    if name == "breast_cancer":
        dataset = load_breast_cancer(as_frame=True)
        return DatasetBundle(
            name="Breast cancer classification",
            task="classification",
            X=dataset.data,
            y=dataset.target,
            scoring="roc_auc",
            metric_name="ROC AUC",
            loss_name="1 - ROC AUC",
        )
    raise ValueError(f"Unknown dataset {name!r}; choose diabetes or breast_cancer.")


def split_dataset(
    dataset: DatasetBundle,
    *,
    seed: int,
    test_size: float = 0.2,
) -> DatasetSplit:
    stratify = dataset.y if dataset.task == "classification" else None
    train_X, test_X, train_y, test_y = train_test_split(
        dataset.X,
        dataset.y,
        test_size=test_size,
        shuffle=True,
        stratify=stratify,
        random_state=seed,
    )
    return DatasetSplit(train_X, test_X, train_y, test_y)


def constrained_num_leaves(max_depth: int) -> int:
    """Keep LightGBM leaves consistent with the selected tree depth."""

    return min(31, 2**max_depth)


def make_model(
    dataset: DatasetBundle,
    params: dict[str, float | int] | None,
    *,
    seed: int,
):
    model_params: dict[str, object] = {
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    if params:
        model_params.update(params)
        max_depth = int(model_params.get("max_depth", -1))
        if max_depth > 0 and "num_leaves" not in model_params:
            model_params["num_leaves"] = constrained_num_leaves(max_depth)
    if dataset.task == "regression":
        return LGBMRegressor(**model_params)
    return LGBMClassifier(**model_params)


def predict_for_metric(
    dataset: DatasetBundle,
    model,
    X: pd.DataFrame,
) -> np.ndarray:
    if dataset.task == "classification":
        return np.asarray(model.predict_proba(X)[:, 1])
    return np.asarray(model.predict(X))
