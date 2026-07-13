#!/usr/bin/env python
"""Build the synchronized Korean and English tutorial notebooks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = {
    "ko": ROOT / "HyperParameterInspect.ipynb",
    "en": ROOT / "HyperParameterInspect_EN.ipynb",
}


@dataclass(frozen=True)
class CellSpec:
    kind: str
    ko: str
    en: str
    tags: tuple[str, ...] = ()


def md(ko: str, en: str) -> CellSpec:
    return CellSpec("markdown", dedent(ko).strip(), dedent(en).strip())


def code(source: str, *tags: str) -> CellSpec:
    normalized = dedent(source).strip()
    return CellSpec("code", normalized, normalized, tags)


CELLS = [
    md(
        """
        <div align="center">
          <img src="pic/hyperparameteroptimization.png" alt="Hyperparameter Optimization" width="360"/>
          <h1>하이퍼파라미터 최적화: 신뢰할 수 있는 실험 설계</h1>
          <p><strong>Grid · Random · TPE · Gaussian Process · CMA-ES · Hyperband · Multi-objective</strong></p>
        </div>

        > 이 노트북은 알고리즘의 순위를 선언하지 않습니다. 같은 데이터, 같은 탐색 범위,
        > 같은 검증 분할과 계산 예산에서 각 방법의 **품질·속도·안정성**을 직접 측정합니다.

        [English notebook](HyperParameterInspect_EN.ipynb) ·
        [한국어 README](README_KR.md) · [Project README](README.md)
        """,
        """
        <div align="center">
          <img src="pic/hyperparameteroptimization.png" alt="Hyperparameter Optimization" width="360"/>
          <h1>Hyperparameter Optimization: Designing Trustworthy Experiments</h1>
          <p><strong>Grid · Random · TPE · Gaussian Process · CMA-ES · Hyperband · Multi-objective</strong></p>
        </div>

        > This notebook does not declare a universal winner. It measures **quality, cost,
        > and stability** under the same data, search bounds, validation splits, and
        > compute budget.

        [한국어 노트북](HyperParameterInspect.ipynb) ·
        [한국어 README](README_KR.md) · [Project README](README.md)
        """,
    ),
    md(
        """
        ## 이 노트북에서 배우는 것

        1. **알고리즘과 프레임워크를 구분**합니다.
           - Grid, Random, TPE, GP, CMA-ES는 탐색 전략입니다.
           - Optuna와 Hyperopt는 전략을 실행하는 프레임워크입니다.
           - Hyperband와 Pruning은 제한된 자원을 배분하는 전략입니다.
        2. 후보 수뿐 아니라 **model fit 수와 실제 시간**까지 함께 비교합니다.
        3. 테스트 세트를 튜닝에 재사용하지 않고, 여러 seed와 95% 신뢰구간으로 변동성을 봅니다.
        4. 조건부 탐색공간, 실제 조기 종료, Nested CV, Pareto 최적화를 실행합니다.
        5. SQLite 재개, CSV/JSON 내보내기, CLI와 CI까지 실무 흐름을 연결합니다.

        ### 실행 순서

        **설정 → 공정한 기본 비교 → 탐색공간 설계 → Multi-fidelity →
        Nested CV → Multi-objective → 운영·재현성**
        """,
        """
        ## What you will learn

        1. **Separate algorithms from frameworks.**
           - Grid, Random, TPE, GP, and CMA-ES are search strategies.
           - Optuna and Hyperopt are frameworks that execute strategies.
           - Hyperband and pruning allocate limited resources.
        2. Compare candidate count, **model-fit count, and wall-clock time** together.
        3. Keep the test split out of tuning and expose variance with multiple seeds
           and 95% confidence intervals.
        4. Run conditional spaces, real early stopping, nested CV, and Pareto optimization.
        5. Connect the lesson to SQLite resume, CSV/JSON export, CLI, and CI.

        ### Learning path

        **Setup → fair benchmark → search-space design → multi-fidelity →
        nested CV → multi-objective → operations and reproducibility**
        """,
    ),
    code(
        """
        from dataclasses import replace
        import importlib.metadata as metadata
        import os
        from pathlib import Path
        import platform

        from IPython import get_ipython
        from IPython.display import Markdown, display
        import matplotlib.pyplot as plt
        import numpy as np
        import optuna
        import pandas as pd

        from hpo_lab import (
            ExperimentConfig,
            load_dataset,
            run_benchmark,
            run_conditional_search,
            run_multifidelity_demo,
            run_multiobjective_search,
            run_nested_validation,
            run_search,
            split_dataset,
        )
        from hpo_lab.plots import (
            plot_convergence,
            plot_pareto_front,
            plot_pruning_budget,
            plot_quality_vs_time,
            plot_seed_stability,
        )

        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic("matplotlib", "inline")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        plt.style.use("seaborn-v0_8-whitegrid")
        pd.set_option("display.max_colwidth", 120)

        config = ExperimentConfig.from_env(default="quick")
        ARTIFACT_DIR = Path("artifacts") / f"notebook-{config.mode}"
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"mode={config.mode} | trials={config.n_trials} | "
              f"folds={config.cv_folds} | seeds={config.seeds}")
        print(f"artifacts={ARTIFACT_DIR.resolve()}")
        """,
        "setup",
    ),
    md(
        """
        ## 1. 실행 모드와 실험 계약

        | 모드 | 용도 | 기본 특성 |
        |---|---|---|
        | `smoke` | CI와 설치 확인 | 최소 trial, 가장 빠름 |
        | `quick` | 기본 학습 모드 | 모든 핵심 탐색기와 2개 seed |
        | `full` | 심층 분석 | 더 많은 trial·fold·seed |

        셸에서 `HPO_MODE=full jupyter lab`처럼 지정할 수 있습니다. 어떤 모드든
        Grid의 조합 수는 다른 탐색기의 trial 수와 정확히 같습니다.

        ### 실험 계약

        - 모든 탐색기는 동일한 train/test split과 CV fold를 사용합니다.
        - `learning_rate`, `max_depth`, `n_estimators`의 바깥 경계가 같습니다.
        - 최적 파라미터는 **CV로만** 선택하고 test는 마지막에 한 번만 평가합니다.
        - Grid의 유한 격자와 연속형 탐색기는 완전히 같은 점을 볼 수 없으므로,
          후보 수·fit 수·시간을 함께 공개합니다.
        """,
        """
        ## 1. Execution modes and the experiment contract

        | Mode | Purpose | Default behavior |
        |---|---|---|
        | `smoke` | CI and installation check | Minimum trials, fastest run |
        | `quick` | Default learning mode | All core optimizers and two seeds |
        | `full` | Deeper analysis | More trials, folds, and seeds |

        Set a mode before starting Jupyter, for example
        `HPO_MODE=full jupyter lab`. In every mode, the Grid candidate count
        exactly matches the trial budget of the other optimizers.

        ### Experiment contract

        - Every optimizer sees the same train/test split and CV folds.
        - The outer bounds of `learning_rate`, `max_depth`, and `n_estimators` match.
        - Parameters are selected **only by CV**; the test split is evaluated once at the end.
        - A finite Grid and continuous optimizers cannot inspect identical points, so the
          notebook reports candidates, model fits, and time together.
        """,
    ),
    code(
        """
        mode_table = pd.DataFrame(
            [
                {
                    "mode": mode,
                    "trials per method": preset.n_trials,
                    "CV folds": preset.cv_folds,
                    "seeds": len(preset.seeds),
                    "grid candidates": preset.grid_size,
                    "methods": ", ".join(preset.methods),
                }
                for mode in ("smoke", "quick", "full")
                for preset in [ExperimentConfig.for_mode(mode)]
            ]
        )
        display(mode_table)
        assert config.grid_size == config.n_trials
        """,
    ),
    md(
        """
        ## 2. 먼저 개념 지도를 바로잡기

        | 층위 | 예시 | 답하는 질문 |
        |---|---|---|
        | 탐색 전략 | Grid, Random, TPE, GP, CMA-ES | 다음 후보를 어디서 고를까? |
        | 자원 배분 | Successive Halving, Hyperband, Pruning | 나쁜 후보를 언제 중단할까? |
        | 실행 프레임워크 | Optuna, Hyperopt, Ray Tune | 저장·병렬화·재개를 어떻게 할까? |
        | 평가 설계 | Holdout, CV, Nested CV, multi-seed | 결과를 얼마나 믿을 수 있을까? |

        기존 튜토리얼에서 흔한 혼동은 “Optuna 대 Hyperopt”를 “TPE 대 TPE”처럼
        비교하는 것입니다. 여기서는 Grid/Random/TPE/CMA-ES를 같은 Optuna 실행 틀로
        비교하고, GP는 가벼운 Gaussian Process + Expected Improvement 구현을 사용합니다.

        ![HPO concept map](pic/mindmap_kr.png)
        """,
        """
        ## 2. Fix the conceptual map first

        | Layer | Examples | Question answered |
        |---|---|---|
        | Search strategy | Grid, Random, TPE, GP, CMA-ES | Where should the next candidate come from? |
        | Resource allocation | Successive Halving, Hyperband, pruning | When should a weak candidate stop? |
        | Execution framework | Optuna, Hyperopt, Ray Tune | How do we store, parallelize, and resume? |
        | Evaluation design | Holdout, CV, nested CV, multiple seeds | How much can we trust the result? |

        A common tutorial mistake is comparing “Optuna versus Hyperopt” as if it were
        “TPE versus TPE.” Here Grid, Random, TPE, and CMA-ES share the same Optuna
        execution layer, while GP uses a lightweight Gaussian Process + Expected
        Improvement implementation.

        ![HPO concept map](pic/mindmap_en.png)
        """,
    ),
    md(
        """
        ## 3. 데이터와 누수 없는 평가

        기본 문제는 sklearn Diabetes 회귀 데이터입니다.

        - 442개 샘플, 10개 특성
        - LightGBM 회귀
        - 선택 지표: CV MSE
        - 최종 보고: 보지 않은 holdout MSE

        **중요:** Baseline의 CV MSE와 테스트 MSE는 서로 다른 값입니다. 이 노트북은
        둘을 같은 값처럼 비교하지 않습니다. 모든 방법은 CV에서 선택된 뒤 동일한
        holdout split에서 딱 한 번 평가됩니다.
        """,
        """
        ## 3. Data and leakage-free evaluation

        The default task is sklearn's Diabetes regression dataset.

        - 442 samples and 10 features
        - LightGBM regression
        - Selection metric: CV MSE
        - Final report: unseen holdout MSE

        **Important:** baseline CV MSE and test MSE are different quantities. This
        notebook never substitutes one for the other. Each method is selected by CV
        and evaluated exactly once on the same holdout split.
        """,
    ),
    code(
        """
        dataset = load_dataset("diabetes")
        preview_split = split_dataset(dataset, seed=config.seeds[0])

        dataset_overview = pd.DataFrame(
            {
                "item": ["dataset", "task", "samples", "features", "train", "holdout", "metric"],
                "value": [
                    dataset.name,
                    dataset.task,
                    dataset.X.shape[0],
                    dataset.X.shape[1],
                    preview_split.train_X.shape[0],
                    preview_split.test_X.shape[0],
                    dataset.metric_name,
                ],
            }
        )
        display(dataset_overview)
        display(dataset.X.head(3))
        """,
    ),
    md(
        """
        ## 4. 공정한 Black-box 탐색 비교

        | 방법 | 다음 후보 선택 | 강점 | 주의점 |
        |---|---|---|---|
        | Grid | 정해진 모든 격자점 | 단순·재현 가능 | 차원이 늘면 조합 폭발 |
        | Random | 분포에서 독립 표본 | 강한 baseline | 이전 결과를 학습하지 않음 |
        | TPE | 좋은/나쁜 영역의 밀도 비율 | 혼합·조건부 공간에 강함 | startup trial 필요 |
        | GP + EI | 확률적 대리 모델과 획득함수 | 저차원·비싼 목적함수 | trial 수 증가 시 대리 모델 비용 증가 |
        | CMA-ES | 진화 전략 | 연속 공간의 상호작용 탐색 | 범주형·조건부 공간에 부적합 |

        아래 실행은 방법마다 동일한 후보 수와 CV fold 수를 사용합니다. `full` 모드가
        아니어도 모든 핵심 흐름을 볼 수 있습니다.
        """,
        """
        ## 4. Fair black-box search benchmark

        | Method | How the next candidate is chosen | Strength | Caveat |
        |---|---|---|---|
        | Grid | Every predefined grid point | Simple and reproducible | Combinatorial growth |
        | Random | Independent samples from distributions | Strong baseline | Does not learn from history |
        | TPE | Density ratio of good and bad regions | Strong on mixed/conditional spaces | Needs startup trials |
        | GP + EI | Probabilistic surrogate and acquisition | Low-dimensional expensive objectives | Surrogate cost grows with trials |
        | CMA-ES | Evolution strategy | Interactions in continuous spaces | Poor fit for categorical/conditional spaces |

        The run below gives every method the same candidate count and CV fold count.
        All core workflows are available without selecting `full` mode.
        """,
    ),
    code(
        """
        benchmark = run_benchmark(config, dataset_name="diabetes")
        benchmark_path = benchmark.save(ARTIFACT_DIR / "benchmark")
        print(f"Saved benchmark tables to {benchmark_path}")
        """,
        "benchmark",
    ),
    code(
        """
        summary_columns = [
            "method",
            "runs",
            "cv_loss_mean",
            "cv_loss_ci95",
            "holdout_metric_mean",
            "holdout_loss_ci95",
            "search_seconds_mean",
            "fit_count_mean",
        ]
        display(benchmark.summary[summary_columns].round(4))

        budget_check = (
            benchmark.runs_frame.query("method != 'Baseline'")
            .groupby("method", as_index=False)["fit_count"]
            .agg(["min", "max"])
        )
        display(budget_check)
        """,
    ),
    md(
        """
        ### 결과를 읽는 순서

        1. `cv_loss_mean`: 탐색기가 실제로 최적화한 값
        2. `holdout_metric_mean`: 선택 후 한 번 평가한 일반화 성능
        3. `*_ci95`: seed가 바뀔 때의 불확실성 (`smoke`의 단일 seed에서는 `NaN`)
        4. `search_seconds_mean`: 목적함수와 탐색기 오버헤드를 포함한 시간
        5. `fit_count_mean`: trial 수가 달라도 계산량을 점검할 수 있는 지표

        한 번의 실행에서 1등인 방법보다, 여러 seed에서 안정적이고 비용 대비 개선이
        충분한 방법이 실무에서는 더 좋은 선택일 수 있습니다.
        """,
        """
        ### Read results in this order

        1. `cv_loss_mean`: the quantity actually optimized
        2. `holdout_metric_mean`: generalization after selection
        3. `*_ci95`: uncertainty across seeds (`NaN` for the single-seed `smoke` mode)
        4. `search_seconds_mean`: objective plus optimizer overhead
        5. `fit_count_mean`: a compute check even when trial counts differ

        The method that wins one run is often less useful than a stable method whose
        improvement justifies its cost across seeds.
        """,
    ),
    code(
        """
        figure, _ = plot_convergence(benchmark)
        figure.savefig(ARTIFACT_DIR / "convergence.png", dpi=160, bbox_inches="tight")
        plt.show()
        """,
    ),
    code(
        """
        figure, _ = plot_quality_vs_time(benchmark)
        figure.savefig(ARTIFACT_DIR / "quality-vs-time.png", dpi=160, bbox_inches="tight")
        plt.show()

        figure, _ = plot_seed_stability(benchmark)
        figure.savefig(ARTIFACT_DIR / "seed-stability.png", dpi=160, bbox_inches="tight")
        plt.show()
        """,
    ),
    md(
        """
        ### 파라미터와 탐색 이력 확인

        수렴곡선만으로는 탐색기가 왜 달라졌는지 알 수 없습니다. 선택된 파라미터와
        TPE study의 importance를 함께 봅니다. Importance는 **현재 데이터·범위·trial에
        국한된 진단값**이지, 파라미터의 보편적 중요도를 뜻하지 않습니다.
        """,
        """
        ### Inspect parameters and search history

        A convergence curve does not explain why optimizers differ. Inspect selected
        parameters and the TPE study's importance together. Importance is a diagnostic
        **specific to this dataset, range, and trial history**, not a universal ranking.
        """,
    ),
    code(
        """
        selected_runs = (
            benchmark.runs_frame.query("method != 'Baseline'")
            [["method", "seed", "best_cv_loss", "holdout_metric", "best_params"]]
            .sort_values(["seed", "best_cv_loss"])
        )
        display(selected_runs)

        tpe_study = benchmark.studies[("TPE", config.seeds[0])]
        importance = optuna.importance.get_param_importances(tpe_study)
        importance_frame = pd.DataFrame(
            {"parameter": importance.keys(), "importance": importance.values()}
        )
        display(importance_frame)
        """,
    ),
    md(
        """
        ## 5. 탐색공간 설계가 알고리즘보다 먼저다

        잘못된 공간에서는 좋은 탐색기도 좋은 답을 찾을 수 없습니다.

        - `learning_rate`처럼 자릿수가 중요한 값은 **log 분포**를 사용합니다.
        - 정수와 실수를 구분하고 의미 없는 정밀도를 만들지 않습니다.
        - `num_leaves ≤ 2^max_depth`처럼 도메인 제약을 코드로 표현합니다.
        - `boosting_type='dart'`일 때만 `drop_rate`를 제안하는 식으로 조건부 공간을 만듭니다.

        다음 실습은 GBDT와 DART가 서로 다른 하위 공간을 갖는 실제 조건부 TPE 탐색입니다.
        """,
        """
        ## 5. Search-space design comes before the optimizer

        A good optimizer cannot rescue a bad search space.

        - Use a **log distribution** for scale-sensitive values such as `learning_rate`.
        - Distinguish integers from reals and avoid meaningless precision.
        - Encode domain constraints such as `num_leaves ≤ 2^max_depth`.
        - Create conditional branches, for example suggesting `drop_rate` only when
          `boosting_type='dart'`.

        The next exercise runs a real conditional TPE space where GBDT and DART have
        different child parameters.
        """,
    ),
    code(
        """
        conditional = run_conditional_search(config, dataset_name="diabetes")
        display(
            pd.DataFrame(
                {
                    "best CV loss": [conditional.best_cv_loss],
                    "holdout metric": [conditional.holdout_metric],
                    "best params": [conditional.best_params],
                }
            )
        )
        display(conditional.trials.sort_values("cv_loss").head())
        """,
        "advanced",
    ),
    md(
        """
        ## 6. Multi-fidelity: Hyperband가 실제로 중단하게 만들기

        TPE는 “어디를 탐색할지”, Hyperband는 “각 후보에 자원을 얼마나 줄지”를 결정합니다.
        둘은 대체 관계가 아니라 결합할 수 있는 서로 다른 층입니다.

        여기서는 boosting tree 수를 자원으로 봅니다.

        1. 작은 tree budget으로 시작
        2. `trial.report()`로 중간 CV loss 보고
        3. `trial.should_prune()`으로 약한 후보 중단
        4. 살아남은 fold의 LightGBM booster는 다음 rung에서 이어서 학습

        따라서 아래 코드는 “Pruning이라고 설명하지만 실제로는 끝까지 학습”하는 예제가 아닙니다.
        `resource_units`는 fold별로 추가 학습한 tree 수를 합산합니다.
        """,
        """
        ## 6. Multi-fidelity: make Hyperband actually stop work

        TPE decides **where to search**; Hyperband decides **how much resource each
        candidate receives**. They belong to different layers and can be combined.

        Here the number of boosting trees is the resource.

        1. Start with a small tree budget.
        2. Report intermediate CV loss with `trial.report()`.
        3. Stop weak candidates with `trial.should_prune()`.
        4. Continue each surviving fold's LightGBM booster at the next rung.

        This is not an example that talks about pruning but trains every candidate to
        completion. `resource_units` sums the additional trees trained across folds.
        """,
    ),
    code(
        """
        multifidelity = run_multifidelity_demo(config, dataset_name="diabetes")
        display(multifidelity.summary.round(4))
        display(
            multifidelity.trials.groupby(["strategy", "state"], as_index=False)
            .agg(trials=("trial", "count"), resource_units=("resource_units", "sum"))
        )
        """,
        "advanced",
    ),
    code(
        """
        figure, _ = plot_pruning_budget(multifidelity)
        figure.savefig(ARTIFACT_DIR / "hyperband-budget.png", dpi=160, bbox_inches="tight")
        plt.show()
        """,
    ),
    md(
        """
        ## 7. Nested CV: 튜닝 자체의 과적합 측정

        같은 CV 결과를 보고 파라미터를 고른 뒤 그 CV 점수를 최종 성능처럼 보고하면
        낙관적 편향이 생깁니다. Nested CV는 역할을 분리합니다.

        - **Inner CV:** 하이퍼파라미터 선택
        - **Outer CV:** 선택 절차 전체의 일반화 성능 평가

        비용이 크기 때문에 기본 모드에서는 작은 예산을 사용합니다. 논문·모델 선택처럼
        신뢰성이 중요한 작업에서는 단일 holdout보다 강한 평가 방식입니다.
        """,
        """
        ## 7. Nested CV: measure overfitting by the tuning process

        Selecting parameters from CV and then reporting that same CV score as final
        performance creates optimistic bias. Nested CV separates the roles.

        - **Inner CV:** select hyperparameters
        - **Outer CV:** evaluate the entire selection procedure

        The default modes use a small budget because nested CV is expensive. It is
        stronger than a single holdout when trustworthy model selection matters.
        """,
    ),
    code(
        """
        nested = run_nested_validation(config, dataset_name="diabetes")
        display(nested)
        nested_summary = pd.DataFrame(
            {
                "baseline outer loss mean": [nested["baseline_holdout_loss"].mean()],
                "tuned outer loss mean": [nested["tuned_holdout_loss"].mean()],
                "tuned outer loss std": [nested["tuned_holdout_loss"].std(ddof=1)],
            }
        )
        display(nested_summary.round(4))
        """,
        "advanced",
    ),
    md(
        """
        ## 8. Multi-objective: 정확도 하나만 최적화하지 않기

        프로덕션 모델은 정확도뿐 아니라 지연시간·메모리·모델 크기 제약을 받습니다.
        여기서는 두 목적을 동시에 최소화합니다.

        1. CV loss
        2. 모델 복잡도 proxy = `n_estimators × num_leaves`

        실행시간은 환경 잡음이 크므로 목적함수에는 결정적인 복잡도 proxy를 사용하고,
        선택된 Pareto knee 모델의 실제 예측시간은 별도로 측정합니다. Pareto frontier의
        어떤 점도 다른 점보다 두 목적 모두에서 나쁘지 않습니다.
        """,
        """
        ## 8. Multi-objective: optimize more than accuracy

        Production models face latency, memory, and size constraints in addition to
        predictive quality. This exercise minimizes two objectives together.

        1. CV loss
        2. Model-complexity proxy = `n_estimators × num_leaves`

        Runtime is noisy across machines, so the objective uses a deterministic
        complexity proxy and measures actual prediction time separately for the selected
        Pareto-knee model. No point on the Pareto frontier is worse than another point
        on both objectives.
        """,
    ),
    code(
        """
        multiobjective = run_multiobjective_search(config, dataset_name="diabetes")
        display(multiobjective.pareto.sort_values("complexity"))
        display(
            pd.DataFrame(
                {
                    "selected trial": [multiobjective.selected_trial],
                    "holdout metric": [multiobjective.holdout_metric],
                    "prediction ms": [multiobjective.predict_milliseconds],
                    "selected params": [multiobjective.selected_params],
                }
            )
        )
        """,
        "advanced",
    ),
    code(
        """
        figure, _ = plot_pareto_front(multiobjective)
        figure.savefig(ARTIFACT_DIR / "pareto-front.png", dpi=160, bbox_inches="tight")
        plt.show()
        """,
    ),
    md(
        """
        ## 9. 실무 운영: 저장·재개·내보내기

        긴 HPO는 프로세스가 종료되어도 이어져야 합니다. `run_search`는 Optuna sampler에
        SQLite storage를 전달하면 기존 study를 불러오고 **목표 총 trial 수까지만** 추가합니다.

        - 같은 셀을 다시 실행해도 trial을 무한히 추가하지 않습니다.
        - SQLite writer는 파일 lock으로 직렬화되어 동시 재개 시 목표 예산을 넘지 않습니다.
        - 실패·중단 상태가 남은 study는 비용 누락을 막기 위해 새 이름으로 다시 시작합니다.
        - 각 trial의 상태·파라미터·중간 속성이 SQLite에 남습니다.
        - 기본 벤치마크는 이미 `runs.csv`, `history.csv`, `summary.csv`,
          `best_params.json`, `config.json`을 내보냈습니다.
        - CLI에서는 `hpo-lab benchmark --mode quick`으로 같은 엔진을 실행합니다.

        병렬 trial은 목적함수 내부 CV 병렬화와 중첩하지 않는 것이 좋습니다. 이 튜토리얼은
        oversubscription을 피하기 위해 trial은 직렬, CV의 `n_jobs`만 설정합니다.
        """,
        """
        ## 9. Operations: persistence, resume, and export

        Long HPO runs must survive process restarts. When `run_search` receives SQLite
        storage for an Optuna sampler, it reloads the study and adds trials **only until
        the target total is reached**.

        - Re-running the cell does not append trials forever.
        - SQLite writers are file-locked so concurrent resumes cannot exceed the target.
        - Studies containing failed or interrupted trials must restart under a new name
          to keep cost accounting complete.
        - Trial state, parameters, and intermediate attributes remain in SQLite.
        - The benchmark already exported `runs.csv`, `history.csv`, `summary.csv`,
          `best_params.json`, and `config.json`.
        - The CLI runs the same engine with `hpo-lab benchmark --mode quick`.

        Avoid nesting parallel trials and parallel CV. This tutorial executes trials
        sequentially and controls only CV `n_jobs` to prevent oversubscription.
        """,
    ),
    code(
        """
        resume_config = replace(
            config,
            n_trials=min(6, config.n_trials),
            cv_folds=min(3, config.cv_folds),
            methods=("TPE",),
        )
        resume_split = split_dataset(dataset, seed=config.seeds[0])
        database_path = (ARTIFACT_DIR / "resume-study.db").resolve()
        resume_run = run_search(
            "TPE",
            dataset,
            resume_split,
            resume_config,
            seed=config.seeds[0],
            storage=f"sqlite:///{database_path}",
            study_name=f"notebook-resume-{config.mode}",
        )
        print(
            f"study trials={len(resume_run.study.trials)} | "
            f"target={resume_config.n_trials} | database={database_path}"
        )
        """,
        "operations",
    ),
    md(
        """
        ## 10. 회귀에서 분류로 확장

        HPO 엔진은 Breast Cancer 분류 데이터와 ROC AUC도 지원합니다. `full` 모드에서는
        Random과 TPE를 분류 문제에서도 실행합니다. 빠른 모드는 전체 실행시간을 지키기 위해
        이 도전 과제를 건너뜁니다.

        회귀에서 잘 작동한 탐색 범위가 분류에서도 적절하다는 보장은 없습니다. 데이터셋과
        metric이 바뀌면 탐색공간과 validation protocol도 다시 검토해야 합니다.
        """,
        """
        ## 10. Extend from regression to classification

        The same engine supports the Breast Cancer classification dataset and ROC AUC.
        In `full` mode, Random and TPE are also evaluated on classification. Faster modes
        skip this challenge to preserve execution time.

        A search space that works for regression is not automatically appropriate for
        classification. Revisit both the space and validation protocol when the dataset
        or metric changes.
        """,
    ),
    code(
        """
        if config.mode == "full":
            classification = run_benchmark(
                config,
                dataset_name="breast_cancer",
                methods=("Random", "TPE"),
            )
            classification.save(ARTIFACT_DIR / "classification")
            display(classification.summary.round(4))
        else:
            display(Markdown(
                "**Classification challenge skipped.** "
                "Set `HPO_MODE=full` before launching Jupyter to run it."
            ))
        """,
        "full-only",
    ),
    md(
        """
        ## 11. 해석 체크리스트와 흔한 실패

        - [ ] 테스트 세트를 파라미터 선택에 사용하지 않았는가?
        - [ ] 비교 방법의 범위·fold·seed·후보 예산이 일치하는가?
        - [ ] `loguniform`의 밑을 혼동하지 않았는가? (`exp`와 `10**x`는 다름)
        - [ ] “Optuna”를 하나의 알고리즘처럼 부르지 않았는가?
        - [ ] pruning 코드가 실제로 중간값을 보고하고 중단하는가?
        - [ ] 한 번의 최고 점수 대신 분산과 비용을 함께 봤는가?
        - [ ] 측정 시간의 하드웨어·병렬성·캐시 영향을 기록했는가?
        - [ ] 파라미터 중요도를 현재 study 밖으로 일반화하지 않았는가?

        ### 이 실습의 한계

        Diabetes는 작고 빠른 교육용 데이터입니다. 알고리즘 순위는 데이터, 공간, budget,
        noise, seed에 따라 바뀝니다. 이 결과를 보편적 순위표로 사용하지 마세요.
        """,
        """
        ## 11. Interpretation checklist and common failures

        - [ ] Was the test split excluded from parameter selection?
        - [ ] Do methods share bounds, folds, seeds, and candidate budgets?
        - [ ] Did you distinguish `exp` from `10**x` in log-uniform spaces?
        - [ ] Did you avoid calling “Optuna” a single algorithm?
        - [ ] Does pruning actually report intermediate values and stop work?
        - [ ] Did you inspect variance and cost instead of one best score?
        - [ ] Did you record hardware, parallelism, and cache effects on time?
        - [ ] Did you avoid generalizing importance beyond the current study?

        ### Limitations

        Diabetes is a small, fast teaching dataset. Rankings change with the data,
        search space, budget, noise, and seed. Do not treat these results as a universal
        leaderboard.
        """,
    ),
    code(
        """
        packages = ["hpo-learning-lab", "numpy", "pandas", "scikit-learn", "lightgbm", "optuna"]
        environment = pd.DataFrame(
            {
                "component": ["Python", "platform", *packages],
                "version": [
                    platform.python_version(),
                    platform.platform(),
                    *[metadata.version(package) for package in packages],
                ],
            }
        )
        display(environment)
        """,
    ),
    md(
        """
        ## 마무리

        좋은 HPO는 “가장 화려한 optimizer”를 고르는 일이 아니라 다음을 명시하는 일입니다.

        1. 무엇을 최적화하는가?
        2. 어떤 공간을 허용하는가?
        3. 계산 예산을 어떻게 셀 것인가?
        4. 선택과 최종 평가를 어떻게 분리할 것인가?
        5. 결과를 어떻게 재현하고 재개할 것인가?

        ### 핵심 참고문헌

        - Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization* (2012)
        - Bergstra et al., *Algorithms for Hyper-Parameter Optimization* (2011)
        - Snoek et al., *Practical Bayesian Optimization of Machine Learning Algorithms* (2012)
        - Li et al., *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization* (2018)
        - Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization Framework* (2019)

        실행 결과와 그림은 `artifacts/notebook-<mode>/`에 저장됩니다.
        """,
        """
        ## Closing perspective

        Good HPO is not about choosing the fanciest optimizer. It is about stating:

        1. What is being optimized?
        2. Which space is allowed?
        3. How is compute budget counted?
        4. How are selection and final evaluation separated?
        5. How can the run be reproduced and resumed?

        ### Essential references

        - Bergstra & Bengio, *Random Search for Hyper-Parameter Optimization* (2012)
        - Bergstra et al., *Algorithms for Hyper-Parameter Optimization* (2011)
        - Snoek et al., *Practical Bayesian Optimization of Machine Learning Algorithms* (2012)
        - Li et al., *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization* (2018)
        - Akiba et al., *Optuna: A Next-generation Hyperparameter Optimization Framework* (2019)

        Tables and figures are saved under `artifacts/notebook-<mode>/`.
        """,
    ),
]


def build_notebook(language: str) -> nbformat.NotebookNode:
    cells = []
    for index, spec in enumerate(CELLS):
        source = spec.ko if language == "ko" else spec.en
        if spec.kind == "markdown":
            cell = nbformat.v4.new_markdown_cell(source)
            cell["id"] = f"{language}-markdown-{index:02d}"
        else:
            cell = nbformat.v4.new_code_cell(
                source,
                metadata={"tags": list(spec.tags)} if spec.tags else {},
                execution_count=None,
                outputs=[],
            )
            cell["id"] = f"shared-code-{index:02d}"
        cells.append(cell)

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
            "hpo_lab": {
                "language": language,
                "code_cells_synchronized": True,
                "generated_by": "tools/build_notebooks.py",
            },
        },
    )
    nbformat.validate(notebook)
    return notebook


def render(language: str) -> str:
    return nbformat.writes(build_notebook(language), version=4) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if committed notebooks differ from generated content.",
    )
    args = parser.parse_args()

    stale: list[Path] = []
    for language, path in OUTPUTS.items():
        expected = render(language)
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != expected:
                stale.append(path)
        else:
            path.write_text(expected, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")

    if stale:
        names = ", ".join(str(path.relative_to(ROOT)) for path in stale)
        raise SystemExit(f"Generated notebooks are stale: {names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
