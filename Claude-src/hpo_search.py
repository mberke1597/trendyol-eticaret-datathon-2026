"""
Optional hyperparameter search for the 3 GBDTs, run standalone AFTER
04_build_features.py and BEFORE (or instead of an initial) 05_train.py.

Why this exists: 05_train.py's train_lgb/train_xgb/train_cat hyperparameters
(n_estimators=3000, learning_rate=0.05, depth=8, etc.) were ported unchanged
from ../src/04_train.py -- reasonable, proven starting points, but never
actually searched for THIS competition's specific feature set (49 features,
18.7% positive rate, hard-candidate negatives). A real search over
learning_rate/depth/regularization is one of the more reliable ways to close a
few points of macro-F1 without touching the data or features at all.

Design (kept deliberately simple/fast, not a from-scratch AutoML system):
  - ONE GroupShuffleSplit(term_id) train/val split per algorithm (not full
    GroupKFold) -- a hyperparameter search runs many trials, so each trial
    needs to be cheap; the final chosen hyperparameters get properly
    evaluated via 05_train.py's real 5-fold CV regardless, this script only
    picks a configuration to try there, it doesn't certify performance itself.
  - GLOBAL (non-fold-safe) popularity features for the search's internal
    train/val split -- a documented, deliberate simplification: 05_train.py's
    fold-safe popularity recompute matters for an HONEST final macro-F1
    estimate, but for comparing hyperparameter configurations against each
    other (relative ranking, not an absolute score), the small leakage this
    introduces should affect all configurations roughly equally and is a
    reasonable trade for a much faster search loop.
  - Writes models/best_hyperparams.json: {"lgb": {...}, "xgb": {...}, "cat": {...}}.
    05_train.py's train_lgb/xgb/cat load this file IF PRESENT and override
    their hardcoded defaults with it; if this script was never run, 05_train.py
    is completely unaffected (falls back to the proven hardcoded values).

Usage:
    pip install optuna  # not in the default requirements.txt -- opt-in
    python hpo_search.py                      # searches all 3 algorithms
    python hpo_search.py --algo lgb           # just one, e.g. to fit a time budget
    python 05_train.py                        # picks up best_hyperparams.json automatically
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    CACHE_DIR,
    HPO_N_TRIALS,
    HPO_TIMEOUT_SECONDS,
    MODEL_DIR,
    RANDOM_SEED,
)
from training_utils import best_threshold_for_macro_f1, get_feature_cols  # noqa: E402

try:
    import torch
    USE_GPU = torch.cuda.is_available()
except ImportError:
    USE_GPU = False


def _split(X, y, groups, seed=RANDOM_SEED):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, val_idx = next(gss.split(X, y, groups=groups))
    gss_es = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed + 1)
    inner_tr, inner_es = next(gss_es.split(tr_idx, groups=groups[tr_idx]))
    return tr_idx[inner_tr], tr_idx[inner_es], val_idx


def _macro_f1_for_proba(y_val, proba):
    _, f1 = best_threshold_for_macro_f1(y_val, proba)
    return f1


def search_lgb(X, y, groups, n_trials, timeout):
    import lightgbm as lgb
    import optuna

    tr_i, es_i, val_i = _split(X, y, groups)
    spw = (y[tr_i] == 0).sum() / max((y[tr_i] == 1).sum(), 1)

    def objective(trial):
        params = dict(
            n_estimators=3000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            scale_pos_weight=spw, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(X[tr_i], y[tr_i], eval_set=[(X[es_i], y[es_i])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        proba = model.predict_proba(X[val_i])[:, 1]
        return _macro_f1_for_proba(y[val_i], proba)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    print(f"[lgb] best macro-F1={study.best_value:.4f} params={study.best_params}")
    return study.best_params


def search_xgb(X, y, groups, n_trials, timeout):
    import optuna
    import xgboost as xgb

    tr_i, es_i, val_i = _split(X, y, groups)
    spw = (y[tr_i] == 0).sum() / max((y[tr_i] == 1).sum(), 1)

    def objective(trial):
        params = dict(
            n_estimators=3000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            scale_pos_weight=spw, tree_method="hist", device="cuda" if USE_GPU else "cpu",
            random_state=RANDOM_SEED, n_jobs=-1, early_stopping_rounds=50, eval_metric="logloss",
        )
        model = xgb.XGBClassifier(**params)
        model.fit(X[tr_i], y[tr_i], eval_set=[(X[es_i], y[es_i])], verbose=False)
        proba = model.predict_proba(X[val_i])[:, 1]
        return _macro_f1_for_proba(y[val_i], proba)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    print(f"[xgb] best macro-F1={study.best_value:.4f} params={study.best_params}")
    return study.best_params


def search_cat(X, y, groups, n_trials, timeout):
    import optuna
    from catboost import CatBoostClassifier, Pool

    tr_i, es_i, val_i = _split(X, y, groups)
    spw = (y[tr_i] == 0).sum() / max((y[tr_i] == 1).sum(), 1)

    def objective(trial):
        params = dict(
            iterations=3000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            class_weights=[1.0, spw], random_seed=RANDOM_SEED, verbose=False,
            early_stopping_rounds=50, eval_metric="Logloss",
            task_type="GPU" if USE_GPU else "CPU", devices="0" if USE_GPU else None,
        )
        model = CatBoostClassifier(**params)
        model.fit(Pool(X[tr_i], y[tr_i]), eval_set=Pool(X[es_i], y[es_i]), use_best_model=True)
        proba = model.predict_proba(X[val_i])[:, 1]
        return _macro_f1_for_proba(y[val_i], proba)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    print(f"[cat] best macro-F1={study.best_value:.4f} params={study.best_params}")
    return study.best_params


SEARCHERS = {"lgb": search_lgb, "xgb": search_xgb, "cat": search_cat}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(SEARCHERS.keys()), default=None,
                         help="Search only this algorithm (default: all 3, sequentially)")
    parser.add_argument("--n-trials", type=int, default=HPO_N_TRIALS)
    parser.add_argument("--timeout", type=int, default=HPO_TIMEOUT_SECONDS,
                         help="Per-algorithm wall-clock budget in seconds")
    args = parser.parse_args()

    t_start = time.time()
    train = pd.read_parquet(f"{CACHE_DIR}/train_features.parquet")
    feature_cols = get_feature_cols(train)
    X = train[feature_cols].values.astype(np.float32)
    y = train["label"].values.astype(np.int8)
    groups = train["term_id"].values
    print(f"[main] {len(train):,} rows, {len(feature_cols)} features, "
          f"n_trials={args.n_trials}, timeout={args.timeout}s/algo")

    out_path = MODEL_DIR / "best_hyperparams.json"
    best = {}
    if out_path.exists():
        with open(out_path) as fh:
            best = json.load(fh)
        print(f"[main] {out_path} already has entries for {list(best.keys())} -- will overwrite searched algos only")

    algos = [args.algo] if args.algo else list(SEARCHERS.keys())
    for algo in algos:
        print(f"[main] searching {algo}...")
        t0 = time.time()
        best[algo] = SEARCHERS[algo](X, y, groups, args.n_trials, args.timeout)
        print(f"[main] {algo} search done in {time.time()-t0:.1f}s")

    with open(out_path, "w") as fh:
        json.dump(best, fh, indent=2)
    print(f"[main] wrote {out_path}")
    print(f"[main] total time {time.time()-t_start:.1f}s")
    print("[main] next: python 05_train.py  (automatically loads best_hyperparams.json if present)")


if __name__ == "__main__":
    main()
