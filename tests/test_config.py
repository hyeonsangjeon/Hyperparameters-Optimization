from __future__ import annotations

import pytest

from hpo_lab import ExperimentConfig


@pytest.mark.parametrize("mode", ("smoke", "quick", "full"))
def test_grid_matches_shared_trial_budget(mode: str) -> None:
    config = ExperimentConfig.for_mode(mode)

    assert config.grid_size == config.n_trials
    assert min(config.grid_learning_rates) == config.min_learning_rate
    assert max(config.grid_learning_rates) == config.max_learning_rate
    assert min(config.grid_max_depths) == config.min_max_depth
    assert max(config.grid_max_depths) == config.max_max_depth
    assert min(config.grid_n_estimators) == config.min_n_estimators
    assert max(config.grid_n_estimators) == config.max_n_estimators
    assert config.resource_steps[-1] == config.max_resource
    assert list(config.resource_steps) == sorted(set(config.resource_steps))


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        ExperimentConfig.for_mode("overnight")  # type: ignore[arg-type]
