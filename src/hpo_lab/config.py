"""Experiment presets with explicit and comparable compute budgets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Literal

Mode = Literal["smoke", "quick", "full"]


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration shared by every optimizer in a benchmark."""

    mode: Mode
    n_trials: int
    cv_folds: int
    seeds: tuple[int, ...]
    methods: tuple[str, ...]
    n_jobs: int
    timeout_seconds: float | None
    grid_learning_rates: tuple[float, ...]
    grid_max_depths: tuple[int, ...]
    grid_n_estimators: tuple[int, ...]
    min_learning_rate: float = 0.002
    max_learning_rate: float = 0.3
    min_max_depth: int = 2
    max_max_depth: int = 10
    min_n_estimators: int = 100
    max_n_estimators: int = 800
    min_resource: int = 30
    max_resource: int = 270
    reduction_factor: int = 3
    advanced_trials: int = 8
    nested_outer_folds: int = 2
    nested_inner_folds: int = 2
    nested_trials: int = 3

    @classmethod
    def for_mode(cls, mode: Mode = "quick", *, n_jobs: int = 1) -> "ExperimentConfig":
        """Return a preset sized for CI, interactive learning, or deeper study."""

        presets: dict[Mode, dict[str, object]] = {
            "smoke": {
                "n_trials": 8,
                "cv_folds": 2,
                "seeds": (42,),
                "methods": ("Grid", "Random", "TPE"),
                "timeout_seconds": 90.0,
                "grid_learning_rates": (0.002, 0.3),
                "grid_max_depths": (2, 10),
                "grid_n_estimators": (100, 800),
                "min_resource": 20,
                "max_resource": 60,
                "advanced_trials": 4,
                "nested_outer_folds": 2,
                "nested_inner_folds": 2,
                "nested_trials": 2,
            },
            "quick": {
                "n_trials": 12,
                "cv_folds": 3,
                "seeds": (17, 42),
                "methods": ("Grid", "Random", "TPE", "GP", "CMA-ES"),
                "timeout_seconds": 300.0,
                "grid_learning_rates": (0.002, 0.025, 0.3),
                "grid_max_depths": (2, 10),
                "grid_n_estimators": (100, 800),
                "min_resource": 30,
                "max_resource": 270,
                "advanced_trials": 12,
                "nested_outer_folds": 3,
                "nested_inner_folds": 3,
                "nested_trials": 6,
            },
            "full": {
                "n_trials": 60,
                "cv_folds": 5,
                "seeds": (17, 42, 91),
                "methods": ("Grid", "Random", "TPE", "GP", "CMA-ES"),
                "timeout_seconds": None,
                "grid_learning_rates": (0.002, 0.01, 0.04, 0.12, 0.3),
                "grid_max_depths": (2, 4, 7, 10),
                "grid_n_estimators": (100, 350, 800),
                "min_resource": 30,
                "max_resource": 810,
                "advanced_trials": 40,
                "nested_outer_folds": 5,
                "nested_inner_folds": 4,
                "nested_trials": 20,
            },
        }
        if mode not in presets:
            raise ValueError(f"Unknown mode {mode!r}; choose smoke, quick, or full.")
        config = cls(mode=mode, n_jobs=n_jobs, **presets[mode])
        if config.grid_size != config.n_trials:
            raise ValueError(
                f"{mode} grid has {config.grid_size} candidates, "
                f"but the shared trial budget is {config.n_trials}."
            )
        return config

    @classmethod
    def from_env(cls, default: Mode = "quick") -> "ExperimentConfig":
        """Read HPO_MODE and HPO_N_JOBS for notebook and CI execution."""

        mode = os.getenv("HPO_MODE", default).lower()
        if mode not in {"smoke", "quick", "full"}:
            raise ValueError("HPO_MODE must be smoke, quick, or full.")
        n_jobs = int(os.getenv("HPO_N_JOBS", "1"))
        return cls.for_mode(mode, n_jobs=n_jobs)  # type: ignore[arg-type]

    @property
    def grid_space(self) -> dict[str, tuple[float | int, ...]]:
        return {
            "learning_rate": self.grid_learning_rates,
            "max_depth": self.grid_max_depths,
            "n_estimators": self.grid_n_estimators,
        }

    @property
    def grid_size(self) -> int:
        return (
            len(self.grid_learning_rates)
            * len(self.grid_max_depths)
            * len(self.grid_n_estimators)
        )

    @property
    def resource_steps(self) -> tuple[int, ...]:
        steps: list[int] = []
        resource = self.min_resource
        while resource < self.max_resource:
            steps.append(resource)
            resource *= self.reduction_factor
        steps.append(self.max_resource)
        return tuple(dict.fromkeys(steps))

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
