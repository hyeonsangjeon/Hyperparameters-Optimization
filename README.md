**🇬🇧 English** | [🇰🇷 한국어](README_KR.md)

# 🚀 Hyperparameter Optimization Tutorial

<div align="center">
  <img src="pic/hyperparameteroptimization.png" alt="Hyperparameter Optimization" width="300"/>
</div>

> **Practical comparison of 5 hyperparameter optimization algorithms for machine learning**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-brightgreen)](https://github.com/microsoft/LightGBM)

[🇰🇷 한국어](HyperParameterInspect.ipynb) | [🇬🇧 English](HyperParameterInspect_EN.ipynb) | [🎯 Quick Start](#-quick-start) | [📊 Results](#-key-results)

---

## ⚡ Key Results

**Typical Performance Pattern (Diabetes Dataset: 442 samples, 10 features, 50 iterations)**

| Method | Typical Improvement | Speed | Best For |
|--------|-------------------|-------|----------|
| **TPE (Hyperopt)** | ~27% ⭐⭐ | **Fastest** ⚡ | **Best overall performance** |
| **Random Search** | ~26% ⭐ | Fast | Quick prototyping, reliable |
| **Optuna (TPE+Pruning)** | ~26% ⭐ | Fast | Production systems |
| **Bayesian Optimization** | ~26% ⭐ | Moderate | Critical performance needs |
| **Grid Search** | ~22% | Slow | Small search spaces |
| *Baseline (default)* | *0%* | *-* | *Reference point* |

> 💡 **Important Note**: Actual results vary based on random_state, data split, and environment. All methods typically improve baseline by 20-27%. Run the notebook to see results on your machine.

> ⚡ **Key Insight**: TPE (Hyperopt) achieved the highest improvement (+27.12%), closely followed by Random Search (+26.33%) and Optuna (+26.02%). Modern Bayesian methods consistently outperform Grid Search with better efficiency.

---

## 🎓 What You'll Learn

### 📚 Five Optimization Algorithms

1. **Grid Search** - Exhaustive search through all parameter combinations
2. **Random Search** - Random sampling from parameter distributions  
3. **Optuna** - Modern TPE with pruning (replaces deprecated HyperBand)
4. **Bayesian Optimization** - Probabilistic model-based optimization
5. **TPE (Hyperopt)** - Tree-structured Parzen Estimator

### 🎯 Learning Outcomes

- Understand strengths and weaknesses of each algorithm
- Know which method to choose for different scenarios
- Implement optimization in real projects with working code
- Compare results with statistical rigor
- Reduce hyperparameter tuning time significantly

### 🗺️ Concept Mind Map

<details>
<summary><strong>📌 Click to view Hyperparameter Optimization Concept Mind Map</strong></summary>

<div align="center">
  <img src="pic/mindmap_en.png" alt="Hyperparameter Optimization Concept Mind Map"/>
</div>

</details>

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/hyeonsangjeon/Hyperparameters-Optimization.git
cd Hyperparameters-Optimization
pip install -r requirements.txt
```

### Run Tutorial

**Interactive Notebook** (Recommended)
```bash
jupyter notebook HyperParameterInspect.ipynb        # Korean
jupyter notebook HyperParameterInspect_EN.ipynb     # English
```

**Automated Benchmark**
```bash
python benchmark_hpo_algorithms.py
```

---

## 📊 Algorithm Comparison

### Selection Guide

| Your Scenario | Recommended | Why |
|---------------|-------------|-----|
| **Quick prototyping** | Random Search | Fast setup, decent results |
| **Production deployment** | Optuna | Modern, pruning, actively maintained |
| **Best performance needed** | Bayesian Optimization | Superior results, worth extra time |
| **Limited time budget** | TPE (Hyperopt) | Best speed/quality tradeoff |
| **Small discrete space** | Grid Search | Guarantees finding optimum |
| **Research paper** | Bayesian + TPE | Multiple strong baselines |

### Algorithm Details

| Algorithm | How It Works | Strengths | Limitations |
|-----------|--------------|-----------|-------------|
| **Grid Search** | Exhaustive evaluation of all combinations | Complete coverage, reproducible | Exponential complexity |
| **Random Search** | Random sampling from distributions | Fast, handles continuous params | No learning between trials |
| **Optuna** | TPE with automatic pruning | Modern, efficient, production-ready | Requires setup |
| **Bayesian Optimization** | Gaussian process model of objective | Intelligent search, best results | Slower initial phase |
| **TPE** | Tree-structured Parzen estimators | Fast convergence, proven reliability | Fewer features than Optuna |

---

## 🏆 Benchmark Details

### Experimental Setup

- **Dataset**: Sklearn Diabetes (442 samples, 10 features)
- **Model**: LightGBM Regressor
- **Iterations**: 50 trials per method
- **Validation**: 2-fold cross-validation
- **Metric**: Mean Squared Error (lower is better)

### Performance Characteristics

| Algorithm | Speed | Consistency | Typical Improvement |
|-----------|-------|-------------|---------------------|
| **TPE (Hyperopt)** | ⚡⚡⚡ Fastest | High | 25-35% |
| **Optuna** | ⚡⚡⚡ Very Fast | High | 20-30% |
| **Random Search** | ⚡⚡ Fast | Medium | 20-30% |
| **Bayesian Opt** | ⚡ Moderate | High | 20-30% |
| **Grid Search** | ❌ Slow | Very High | 15-25% |

> ⚠️ **Note**: Values shown are from recent benchmark run. Absolute MSE values vary by environment and random_state, but the ranking and relative performance are consistent across runs.

---

## 📁 Project Structure

```
Hyperparameters-Optimization/
├── HyperParameterInspect.ipynb           # Korean tutorial notebook
├── HyperParameterInspect_EN.ipynb        # English tutorial notebook
├── benchmark_hpo_algorithms.py           # Automated benchmark script
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── pic/                                  # Images and plots
└── doc/                                  # Additional documentation
```

---

## 🔧 Requirements

**Core Dependencies**
- Python 3.8+
- numpy, pandas, scikit-learn, lightgbm

**Optimization Libraries**
- optuna >= 3.0.0 (Modern HPO with pruning)
- hyperopt >= 0.2.7 (TPE algorithm)
- scikit-optimize >= 0.9.0 (Bayesian optimization)

**Visualization**
- matplotlib, jupyter

> ⚠️ **Important**: This project uses **Optuna** instead of the deprecated `scikit-hyperband` library due to compatibility issues with modern scikit-learn versions.

---

## 📚 References

### Key Papers

- **Random Search**: [Bergstra & Bengio, JMLR 2012](https://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf)
- **TPE**: [Bergstra et al., NIPS 2011](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
- **Bayesian Optimization**: [Snoek et al., 2012](https://arxiv.org/abs/1206.2944)
- **HyperBand**: [Li et al., ICLR 2018](https://arxiv.org/pdf/1603.06560.pdf)

### Presentations & Media

- 🎤 Hyeonsang Jeon, **"Expert Lecture: Hyperparameter Optimization in AI Modeling"**, *ITDAILY*, 2022. [Article](http://www.itdaily.kr/news/articleView.html?idxno=210339)

- 🎤 Hyeonsang Jeon, **"Case Study: AutoDL with Hyperparameter Optimization in Deep Learning Platforms"**, *AI Innovation 2020*, The Electronic Times, 2020. [Video](https://youtu.be/QMorERxb1YY?si=iN8opTIjZPc2tTzq)

- 📰 Featured in [ComWorld](https://www.comworld.co.kr/news/articleView.html?idxno=50677)

---

## 🤝 Contributing

Contributions welcome! Ways to help:

- 🐛 Report bugs or issues
- 💡 Suggest new features or algorithms
- 📝 Improve documentation
- 🌍 Translate to other languages
- 🔬 Add optimization methods

**Development Setup**
```bash
git clone https://github.com/YOUR_USERNAME/Hyperparameters-Optimization.git
cd Hyperparameters-Optimization
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python benchmark_hpo_algorithms.py
```

---

##  License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Hyeonsang Jeon**  
GitHub: [@hyeonsangjeon](https://github.com/hyeonsangjeon)

---

## 🙏 Acknowledgments

Special thanks to:
- [Optuna](https://github.com/optuna/optuna) - Modern HPO framework
- [Hyperopt](https://github.com/hyperopt/hyperopt) - TPE implementation
- [scikit-optimize](https://github.com/scikit-optimize/scikit-optimize) - Bayesian optimization
- [LightGBM](https://github.com/microsoft/LightGBM) - Fast gradient boosting

---

## 🔗 Related Projects

- **[Optuna](https://github.com/optuna/optuna)** - Next-generation HPO framework
- **[Hyperopt](https://github.com/hyperopt/hyperopt)** - Distributed HPO library
- **[scikit-optimize](https://github.com/scikit-optimize/scikit-optimize)** - Bayesian optimization
- **[Ray Tune](https://github.com/ray-project/ray)** - Scalable distributed tuning

---

<div align="center">

## ⭐ Found this helpful?

**Star this repository** to support the project and help others discover it!

### 🚀 Share with your team

This tutorial is actively maintained and regularly updated with new techniques.

**Made with ❤️ for the ML community**

[⬆ Back to Top](#-hyperparameter-optimization-tutorial)

</div>
