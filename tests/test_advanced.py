from __future__ import annotations

import pandas as pd

from hpo_lab import (
    ExperimentConfig,
    run_conditional_search,
    run_multifidelity_demo,
    run_multiobjective_search,
    run_nested_validation,
)


def test_hyperband_reports_and_prunes_real_intermediate_work(
) -> None:
    config = ExperimentConfig.for_mode("quick", n_jobs=1)
    result = run_multifidelity_demo(config)
    summary = result.summary.set_index("strategy")
    pruned = result.trials[result.trials["state"] == "PRUNED"]

    assert summary.loc["Hyperband", "pruned_trials"] >= 1
    assert (
        summary.loc["Hyperband", "resource_units"]
        <= summary.loc["No pruning", "resource_units"]
    )
    assert {"COMPLETE", "PRUNED"}.issubset(set(result.trials["state"]))
    assert result.trials["last_resource"].gt(0).all()
    assert pruned["last_resource"].lt(config.max_resource).all()


def test_conditional_space_respects_tree_constraints(
    smoke_config: ExperimentConfig,
) -> None:
    result = run_conditional_search(smoke_config)

    assert not result.trials.empty
    assert (
        result.trials["num_leaves"]
        <= 2 ** result.trials["max_depth"]
    ).all()
    gbdt = result.trials["boosting_type"] == "gbdt"
    assert result.trials.loc[gbdt, "subsample"].notna().all()
    assert result.trials.loc[~gbdt, "drop_rate"].notna().all()


def test_multiobjective_returns_non_dominated_frontier(
    smoke_config: ExperimentConfig,
) -> None:
    result = run_multiobjective_search(smoke_config)

    assert not result.pareto.empty
    assert result.pareto["is_pareto"].all()
    assert result.selected_trial in set(result.pareto["trial"])
    assert result.holdout_loss >= 0


def test_nested_validation_keeps_outer_folds_separate(
    smoke_config: ExperimentConfig,
) -> None:
    result = run_nested_validation(smoke_config)

    assert len(result) == smoke_config.nested_outer_folds
    assert result["outer_fold"].is_unique
    assert result[
        [
            "baseline_holdout_loss",
            "tuned_holdout_loss",
            "selected_inner_cv_loss",
        ]
    ].apply(pd.to_numeric).notna().all().all()
