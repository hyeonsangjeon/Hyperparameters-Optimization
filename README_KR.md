[English](README.md) | [**한국어**](README_KR.md)

# 하이퍼파라미터 최적화 러닝 랩

[![CI](https://github.com/hyeonsangjeon/Hyperparameters-Optimization/actions/workflows/ci.yml/badge.svg)](https://github.com/hyeonsangjeon/Hyperparameters-Optimization/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-ready-orange.svg)](HyperParameterInspect.ipynb)

<div align="center">
  <img src="pic/hyperparameteroptimization.png" alt="Hyperparameter Optimization" width="320"/>
</div>

최적화 라이브러리 호출법만 나열하지 않고, **신뢰할 수 있는 HPO 실험을 설계하고
평가하는 방법**을 다루는 재현 가능한 한·영 튜토리얼입니다.

학습 흐름은 언어별 하나의 종합 노트북으로 유지합니다.

- [한국어 노트북](HyperParameterInspect.ipynb)
- [English notebook](HyperParameterInspect_EN.ipynb)

두 노트북의 코드 셀은 바이트 단위로 동일하며 설명 언어만 다릅니다.

## 이 튜토리얼의 특징

- **탐색 알고리즘**, **자원 배분**, **실행 프레임워크**, **평가 설계**를 구분합니다.
- Grid, Random, TPE, Gaussian Process, CMA-ES를 같은 trial·fold·분할·범위
  예산에서 비교합니다.
- CV 선택 loss, 미사용 holdout 성능, model fit 수, 실행시간, 다중 seed,
  95% 신뢰구간을 함께 보고합니다.
- 중간값 보고와 LightGBM 증분 학습을 사용해 Hyperband가 실제로 trial을 중단합니다.
- 조건부 공간, Nested CV, 다목적 Pareto 최적화, 분류 문제, SQLite 재개,
  CLI 내보내기, CI 노트북 실행까지 포함합니다.
- 특정 optimizer가 항상 최고라는 고정 순위나 과장된 결론을 사용하지 않습니다.

## 빠른 시작

### uv 사용

```bash
git clone https://github.com/hyeonsangjeon/Hyperparameters-Optimization.git
cd Hyperparameters-Optimization
uv sync --extra notebook
uv run jupyter lab HyperParameterInspect.ipynb
```

### pip 사용

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab HyperParameterInspect.ipynb
```

기본 실행은 `quick` 모드입니다. Jupyter 실행 전에 모드를 바꿀 수 있습니다.

```bash
HPO_MODE=smoke uv run jupyter lab  # 가장 빠른 설치 확인
HPO_MODE=full uv run jupyter lab   # 심층 실험
```

| 모드 | 용도 | 핵심 동작 |
|---|---|---|
| `smoke` | CI와 환경 확인 | 최소 trial, 단일 seed |
| `quick` | 대화형 튜토리얼 | 모든 핵심 탐색기, 2개 seed |
| `full` | 심층 분석 | 더 많은 trial·fold·seed와 분류 실험 |

## 학습 흐름

| 구간 | 내용 |
|---|---|
| 실험 계약 | 동일 예산, 공통 fold, holdout 격리 |
| Black-box 탐색 | Grid, Random, TPE, GP + Expected Improvement, CMA-ES |
| 탐색공간 설계 | Log scale, 정수 영역, 제약식, 조건부 분기 |
| Multi-fidelity | Successive Halving 개념, Hyperband, 실제 pruning |
| 신뢰성 평가 | 다중 seed, 신뢰구간, Nested CV |
| 다목적 HPO | 정확도·복잡도 Pareto frontier와 knee 선택 |
| 운영 | SQLite 재개, CSV/JSON, CLI, 재현 가능한 환경 |
| 확장 과제 | 회귀와 선택적 분류 벤치마크 |

## 재현 가능한 CLI 벤치마크

```bash
uv run hpo-lab benchmark --mode smoke
uv run hpo-lab benchmark --mode quick --method Random --method TPE
uv run hpo-lab benchmark --mode full --dataset breast_cancer
```

기존 명령도 호환됩니다.

```bash
uv run python benchmark_hpo_algorithms.py --mode quick
```

실행 결과는 `artifacts/`에 저장됩니다.

```text
best_params.json
config.json
convergence.png
history.csv
quality-vs-time.png
runs.csv
seed-stability.png
summary.csv
```

## 실험 설계

기본 벤치마크는 sklearn Diabetes 회귀 데이터와 LightGBM을 사용합니다. 모든 탐색기는
다음 조건을 공유합니다.

1. seed별 동일 train/holdout 분할
2. 동일한 결정적 CV fold
3. 동일한 파라미터 바깥 경계
4. 동일한 후보 수와 fold 수
5. CV가 최적 구성을 선택한 뒤에만 holdout에 접근

Grid는 유한 격자를 사용하고 다른 탐색기는 연속 영역을 표본화하므로 후보 수만으로
공정성을 증명할 수 없습니다. 따라서 model fit 수, 자원 단위, optimizer overhead,
실행시간, seed 민감도, Nested CV 추정값도 함께 공개합니다.

## 프로젝트 구조

```text
.
├── HyperParameterInspect.ipynb       # 한국어 종합 튜토리얼
├── HyperParameterInspect_EN.ipynb    # 영어 번역본, 동일 코드
├── src/hpo_lab/                      # 테스트된 실험 엔진과 시각화
├── tools/build_notebooks.py          # 결정적 양언어 노트북 생성기
├── tests/                            # 단위·동기화 테스트
├── benchmark_hpo_algorithms.py       # 기존 CLI 호환 진입점
├── pyproject.toml                    # 패키지와 의존성 정의
├── uv.lock                           # 재현 가능한 의존성 잠금
├── README.md
├── README_KR.md
└── pic/                              # 튜토리얼 그림
```

노트북 생성과 동기화 확인:

```bash
uv run python tools/build_notebooks.py
uv run python tools/build_notebooks.py --check
```

## 프로젝트 배경

기존 HPO 발표·교육 자료를 실행 가능한 러닝 랩으로 전면 개편했습니다.

- 전현상, “AI 모델링에서의 하이퍼파라미터 최적화,” *ITDAILY*, 2022 —
  [기사](http://www.itdaily.kr/news/articleView.html?idxno=210339)
- 전현상, “딥러닝 플랫폼에서 HPO를 활용한 AutoDL,” *AI Innovation 2020* —
  [영상](https://youtu.be/QMorERxb1YY)
- 기존 발표 PDF는 저장소 루트에 보존되어 있습니다.

## 라이선스

MIT License. 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.

**작성자:** [전현상 (Hyeonsang Jeon)](https://github.com/hyeonsangjeon)
