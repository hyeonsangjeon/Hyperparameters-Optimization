from __future__ import annotations

import pytest

from hpo_lab import ExperimentConfig


@pytest.fixture
def smoke_config() -> ExperimentConfig:
    return ExperimentConfig.for_mode("smoke", n_jobs=1)
