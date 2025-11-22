#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to validate all sections of HyperParameterInspect.ipynb
"""

import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import warnings
import time
warnings.filterwarnings('ignore')

print("=" * 80)
print("HYPERPARAMETER OPTIMIZATION TUTORIAL - TEST SCRIPT")
print("=" * 80)

print("\n" + "=" * 80)
print("STEP 1: Setup and Data Loading")
print("=" * 80)

# Load data
diabetes = load_diabetes()
n = diabetes.data.shape[0]
data = diabetes.data
targets = diabetes.target

print(f"Dataset shape: {data.shape}")
print(f"Number of samples: {n}")

# Configuration
random_state = 42
n_iter = 50

# Train-test split
train_data, test_data, train_targets, test_targets = train_test_split(
    data, targets, test_size=0.20, shuffle=True, random_state=random_state
)

num_folds = 2
kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)

print(f"Train data: {train_data.shape}")
print(f"Test data: {test_data.shape}")

# Baseline model
model = LGBMRegressor(random_state=random_state, verbose=-1)
baseline_score = -cross_val_score(model, train_data, train_targets,
                                   cv=kf, scoring="neg_mean_squared_error",
                                   n_jobs=-1).mean()
print(f"\nBaseline MSE: {baseline_score:.3f}")

# Storage for results
results = {
    'Method': [],
    'CV MSE': [],
    'Test MSE': [],
    'Time (s)': [],
    'Best Params': []
}

print("\n" + "=" * 80)
print("STEP 2: Grid Search")
print("=" * 80)

from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': np.linspace(5, 12, 8, dtype=int),
    'n_estimators': np.linspace(800, 1200, 5, dtype=int),
    'learning_rate': np.logspace(-3, -1, 3),
    'random_state': [random_state]
}

start_time = time.time()
gs = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error',
                  n_jobs=-1, cv=kf, verbose=False)
gs.fit(train_data, train_targets)
gs_time = time.time() - start_time
gs_test_score = mean_squared_error(test_targets, gs.predict(test_data))

print(f"Best CV MSE: {-gs.best_score_:.3f}")
print(f"Test MSE: {gs_test_score:.3f}")
print(f"Time: {gs_time:.2f}s")
print(f"Best params: {gs.best_params_}")

results['Method'].append('Grid Search')
results['CV MSE'].append(-gs.best_score_)
results['Test MSE'].append(gs_test_score)
results['Time (s)'].append(gs_time)
results['Best Params'].append(str(gs.best_params_))

print("\n" + "=" * 80)
print("STEP 3: Random Search")
print("=" * 80)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_grid_rand = {
    'learning_rate': np.logspace(-5, 0, 100),
    'max_depth': randint(2, 20),
    'n_estimators': randint(100, 2000),
    'random_state': [random_state]
}

start_time = time.time()
rs = RandomizedSearchCV(model, param_grid_rand, n_iter=n_iter,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1, cv=kf, verbose=False,
                        random_state=random_state)
rs.fit(train_data, train_targets)
rs_time = time.time() - start_time
rs_test_score = mean_squared_error(test_targets, rs.predict(test_data))

print(f"Best CV MSE: {-rs.best_score_:.3f}")
print(f"Test MSE: {rs_test_score:.3f}")
print(f"Time: {rs_time:.2f}s")
print(f"Best params: {rs.best_params_}")

results['Method'].append('Random Search')
results['CV MSE'].append(-rs.best_score_)
results['Test MSE'].append(rs_test_score)
results['Time (s)'].append(rs_time)
results['Best Params'].append(str(rs.best_params_))

print("\n" + "=" * 80)
print("STEP 4: Optuna (Modern TPE + Pruning)")
print("=" * 80)

import optuna
from optuna.samplers import TPESampler

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optuna_objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'random_state': random_state,
        'verbose': -1
    }
    model_optuna = LGBMRegressor(**params)
    score = -cross_val_score(model_optuna, train_data, train_targets,
                             cv=kf, scoring="neg_mean_squared_error",
                             n_jobs=-1).mean()
    return score

start_time = time.time()
study = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=random_state)
)
study.optimize(optuna_objective, n_trials=n_iter, show_progress_bar=False)
optuna_time = time.time() - start_time

best_params_optuna = study.best_params.copy()
best_params_optuna['random_state'] = random_state
best_params_optuna['verbose'] = -1

optuna_model = LGBMRegressor(**best_params_optuna)
optuna_model.fit(train_data, train_targets)
optuna_test_score = mean_squared_error(test_targets, optuna_model.predict(test_data))

print(f"Best CV MSE: {study.best_value:.3f}")
print(f"Test MSE: {optuna_test_score:.3f}")
print(f"Time: {optuna_time:.2f}s")
print(f"Best params: {study.best_params}")

results['Method'].append('Optuna')
results['CV MSE'].append(study.best_value)
results['Test MSE'].append(optuna_test_score)
results['Time (s)'].append(optuna_time)
results['Best Params'].append(str(study.best_params))

print("\n" + "=" * 80)
print("STEP 5: Bayesian Optimization")
print("=" * 80)

from skopt import BayesSearchCV
from skopt.space import Integer

search_space = {
    'learning_rate': np.logspace(-5, 0, 100),
    "max_depth": Integer(2, 20),
    'n_estimators': Integer(100, 2000),
    'random_state': [random_state]
}

start_time = time.time()
bayes_search = BayesSearchCV(model, search_space, n_iter=n_iter,
                              scoring='neg_mean_squared_error',
                              n_jobs=-1, cv=kf, verbose=0)
bayes_search.fit(train_data, train_targets)
bayes_time = time.time() - start_time
bayes_test_score = mean_squared_error(test_targets, bayes_search.predict(test_data))

print(f"Best CV MSE: {-bayes_search.best_score_:.3f}")
print(f"Test MSE: {bayes_test_score:.3f}")
print(f"Time: {bayes_time:.2f}s")
print(f"Best params: {bayes_search.best_params_}")

results['Method'].append('Bayesian Opt')
results['CV MSE'].append(-bayes_search.best_score_)
results['Test MSE'].append(bayes_test_score)
results['Time (s)'].append(bayes_time)
results['Best Params'].append(str(bayes_search.best_params_))

print("\n" + "=" * 80)
print("STEP 6: Hyperopt (TPE)")
print("=" * 80)

from hyperopt import fmin, tpe, hp, Trials

def gb_mse_cv(params, random_state=random_state, cv=kf, X=train_data, y=train_targets):
    params = {
        'n_estimators': int(params['n_estimators']),
        'max_depth': int(params['max_depth']),
        'learning_rate': params['learning_rate']
    }
    model = LGBMRegressor(random_state=random_state, verbose=-1, **params)
    score = -cross_val_score(model, X, y, cv=cv,
                             scoring="neg_mean_squared_error",
                             n_jobs=-1).mean()
    return score

space = {
    'n_estimators': hp.quniform('n_estimators', 100, 2000, 1),
    'max_depth': hp.quniform('max_depth', 2, 20, 1),
    'learning_rate': hp.loguniform('learning_rate', -5, 0)
}

# Set numpy random seed (instead of rstate parameter for compatibility)
np.random.seed(random_state)

trials = Trials()
start_time = time.time()
best = fmin(fn=gb_mse_cv,
            space=space,
            algo=tpe.suggest,
            max_evals=n_iter,
            trials=trials,
            verbose=0)
tpe_time = time.time() - start_time

model_tpe = LGBMRegressor(
    random_state=random_state,
    n_estimators=int(best['n_estimators']),
    max_depth=int(best['max_depth']),
    learning_rate=best['learning_rate'],
    verbose=-1
)
model_tpe.fit(train_data, train_targets)
tpe_test_score = mean_squared_error(test_targets, model_tpe.predict(test_data))

print(f"Best CV MSE: {gb_mse_cv(best):.3f}")
print(f"Test MSE: {tpe_test_score:.3f}")
print(f"Time: {tpe_time:.2f}s")
print(f"Best params: {best}")

results['Method'].append('TPE (Hyperopt)')
results['CV MSE'].append(gb_mse_cv(best))
results['Test MSE'].append(tpe_test_score)
results['Time (s)'].append(tpe_time)
results['Best Params'].append(str(best))

print("\n" + "=" * 80)
print("FINAL RESULTS - COMPREHENSIVE COMPARISON")
print("=" * 80)

# Create results dataframe
results_df = pd.DataFrame(results)

# Add baseline
baseline_model = LGBMRegressor(random_state=random_state, verbose=-1)
baseline_model.fit(train_data, train_targets)
baseline_test_score = mean_squared_error(test_targets, baseline_model.predict(test_data))

baseline_row = pd.DataFrame({
    'Method': ['Baseline'],
    'CV MSE': [baseline_score],
    'Test MSE': [baseline_test_score],
    'Time (s)': [0.0],
    'Best Params': ['Default']
})

results_df = pd.concat([baseline_row, results_df], ignore_index=True)

# Sort by Test MSE (lower is better)
results_df_sorted = results_df.sort_values('Test MSE').reset_index(drop=True)

print("\n📊 RESULTS TABLE (Sorted by Test MSE - Lower is Better):")
print("=" * 120)
# Display without Best Params for cleaner view
display_df = results_df_sorted[['Method', 'CV MSE', 'Test MSE', 'Time (s)']].copy()
display_df.index = display_df.index + 1  # Start index from 1
print(display_df.to_string())

print("\n" + "=" * 120)
print("📈 STATISTICAL SUMMARY:")
print("=" * 120)

# Calculate improvements over baseline
results_df_sorted['CV Improvement (%)'] = ((baseline_score - results_df_sorted['CV MSE']) / baseline_score * 100).round(2)
results_df_sorted['Test Improvement (%)'] = ((baseline_test_score - results_df_sorted['Test MSE']) / baseline_test_score * 100).round(2)

stats = {
    'Metric': [
        'Best Test MSE',
        'Worst Test MSE',
        'Best Method',
        'Fastest Method',
        'Avg Improvement over Baseline',
        'Best CV-Test Gap'
    ],
    'Value': [
        f"{results_df_sorted['Test MSE'].min():.3f}",
        f"{results_df_sorted['Test MSE'].max():.3f}",
        results_df_sorted.iloc[0]['Method'],
        results_df_sorted.loc[results_df_sorted['Method'] != 'Baseline'].sort_values('Time (s)').iloc[0]['Method'],
        f"{results_df_sorted.loc[results_df_sorted['Method'] != 'Baseline', 'Test Improvement (%)'].mean():.2f}%",
        f"{abs(results_df_sorted['CV MSE'] - results_df_sorted['Test MSE']).min():.3f}"
    ]
}

stats_df = pd.DataFrame(stats)
print(stats_df.to_string(index=False))

print("\n" + "=" * 120)
print("🏆 RANKING BY DIFFERENT CRITERIA:")
print("=" * 120)

print("\n1️⃣  By Test MSE (Lower is Better):")
rank_test = results_df_sorted[results_df_sorted['Method'] != 'Baseline'][['Method', 'Test MSE']].copy()
rank_test.index = range(1, len(rank_test) + 1)
print(rank_test.to_string())

print("\n2️⃣  By Speed (Faster is Better):")
rank_speed = results_df[results_df['Method'] != 'Baseline'][['Method', 'Time (s)']].sort_values('Time (s)').copy()
rank_speed.index = range(1, len(rank_speed) + 1)
print(rank_speed.to_string())

print("\n3️⃣  By CV MSE (Lower is Better):")
rank_cv = results_df_sorted[results_df_sorted['Method'] != 'Baseline'][['Method', 'CV MSE']].sort_values('CV MSE').copy()
rank_cv.index = range(1, len(rank_cv) + 1)
print(rank_cv.to_string())

print("\n" + "=" * 120)
print("💡 KEY INSIGHTS:")
print("=" * 120)

best_method = results_df_sorted.iloc[1]['Method']  # Skip baseline
fastest_method = results_df.loc[results_df['Method'] != 'Baseline'].sort_values('Time (s)').iloc[0]['Method']
best_improvement = results_df_sorted.loc[results_df_sorted['Method'] != 'Baseline', 'Test Improvement (%)'].max()

print(f"• Best performing method: {best_method}")
print(f"• Fastest method: {fastest_method}")
print(f"• Maximum improvement over baseline: {best_improvement:.2f}%")
print(f"• All methods improved over baseline: {(results_df_sorted['Test Improvement (%)'] > 0).sum() - 1}/{len(results_df_sorted) - 1}")

print("\n✅ All tests completed successfully!")
print("=" * 120)
