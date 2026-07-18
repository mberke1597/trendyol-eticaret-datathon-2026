"""
GroupKFold (by term_id) CV training of a 3-model GBDT ensemble (LightGBM,
XGBoost, CatBoost), fold-safe popularity recomputation, an honest comparison
between simple blend-weight search and logistic-regression meta-stacking, and
exact macro-F1 threshold calibration.

Zero query leakage: term sets in submission_pairs.csv are 100% disjoint from
training_pairs.csv, so CV uses GroupKFold(term_id) -- ported unchanged (proven,
already unit-tested in ../src/) along with the fold-safe popularity recompute
(POP_COLS / build_fold_popularity / fold_X_with_fresh_popularity) that closes
the cross-fold popularity leak documented in ../DESIGN.md (2026-07-03 fix).

NEW vs. ../src/04_train.py -- stacking, not just blend-weight grid search:
A teammate's train_stacking.py used a logistic-regression meta-model on OOF
predictions (a genuinely more principled combiner than a 0.1-step blend-weight
grid), but scored it by re-predicting on the exact same OOF rows it was fit on
-- mildly optimistic. Here, BOTH combiners (grid-searched blend weights AND
logistic-regression stacking) are compared on a held-out slice of the OOF data
that neither combiner was fit on (nested_compare_ensemble_methods, split via a
further GroupShuffleSplit so term_id grouping is preserved even at the
meta-level), and whichever wins is refit on the FULL OOF set for the final
persisted combiner -- meta.json records both the honest selection score and
final full-data metrics so it's clear which number is which.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    CACHE_DIR,
    DATA_DIR,
    LLM_REL_AS_ENSEMBLE_MEMBER,
    META_NESTED_HOLDOUT_FRAC,
    MODEL_DIR,
    N_FOLDS,
    RANDOM_SEED,
    TARGET_POS_RATE,
)
from training_utils import (  # noqa: E402
    POP_COLS,
    best_threshold_for_macro_f1,
    build_fold_popularity,
    fold_X_with_fresh_popularity,
    get_feature_cols,
    load_item_catalog_for_popularity,
    nested_compare_ensemble_methods,
    search_blend_weights,
)

try:
    import torch
    USE_GPU = torch.cuda.is_available()
except ImportError:
    USE_GPU = False

# ---- optional hyperparameter overrides (see hpo_search.py) ----
# If hpo_search.py was run, models/best_hyperparams.json has searched values
# for learning_rate/depth/regularization/etc per algorithm; load them here and
# override the hardcoded defaults below. If the file doesn't exist (hpo_search.py
# was never run -- the common case), BEST_HYPERPARAMS is just {} and every
# train_* function below behaves EXACTLY as before this change.
_best_hp_path = MODEL_DIR / "best_hyperparams.json"
if _best_hp_path.exists():
    with open(_best_hp_path) as _fh:
        BEST_HYPERPARAMS = json.load(_fh)
    print(f"[main] loaded hyperparameter overrides from {_best_hp_path}: "
          f"{list(BEST_HYPERPARAMS.keys())}")
else:
    BEST_HYPERPARAMS = {}


def train_lgb(X_tr, y_tr, X_es, y_es, spw):
    params = dict(
        n_estimators=3000, learning_rate=0.05, num_leaves=63, max_depth=-1,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
        scale_pos_weight=spw, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1,
    )
    params.update(BEST_HYPERPARAMS.get("lgb", {}))
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr, eval_set=[(X_es, y_es)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def train_xgb(X_tr, y_tr, X_es, y_es, spw):
    params = dict(
        n_estimators=3000, learning_rate=0.05, max_depth=8, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=spw,
        tree_method="hist", device="cuda" if USE_GPU else "cpu",
        random_state=RANDOM_SEED, n_jobs=-1,
        early_stopping_rounds=100, eval_metric="logloss",
    )
    params.update(BEST_HYPERPARAMS.get("xgb", {}))
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
    return model


def train_cat(X_tr, y_tr, X_es, y_es, spw):
    params = dict(
        iterations=3000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
        class_weights=[1.0, spw], random_seed=RANDOM_SEED, verbose=False,
        early_stopping_rounds=100, eval_metric="Logloss",
        task_type="GPU" if USE_GPU else "CPU", devices="0" if USE_GPU else None,
    )
    params.update(BEST_HYPERPARAMS.get("cat", {}))
    model = CatBoostClassifier(**params)
    model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_es, y_es), use_best_model=True)
    return model


def main():
    t_start = time.time()
    print(f"[main] USE_GPU={USE_GPU} (xgb device={'cuda' if USE_GPU else 'cpu'}, "
          f"catboost task_type={'GPU' if USE_GPU else 'CPU'}, lgb=cpu always)")
    train = pd.read_parquet(f"{CACHE_DIR}/train_features.parquet")
    feature_cols = get_feature_cols(train)
    print(f"[main] {len(train):,} rows, {len(feature_cols)} features, "
          f"{train['label'].mean()*100:.1f}% positive")

    X = train[feature_cols].values.astype(np.float32)
    y = train["label"].values.astype(np.int8)
    groups = train["term_id"].values

    have_pop_cols = any(c in feature_cols for c in POP_COLS)
    if have_pop_cols:
        item_pos, category, brand = load_item_catalog_for_popularity(DATA_DIR)
        n_items = len(category)
        training_pairs_raw = pd.read_csv(f"{DATA_DIR}/training_pairs.csv", usecols=["term_id", "item_id"])
        item_idx_rows = item_pos.reindex(train["item_id"].values).values
        assert not np.isnan(item_idx_rows).any(), "item_id in train_features.parquet not found in items.csv"
        item_idx_rows = item_idx_rows.astype(np.int64)
    else:
        print("[main] no popularity columns present in features -- skipping fold-aware recompute")

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof = {"lgb": np.zeros(len(train)), "xgb": np.zeros(len(train)), "cat": np.zeros(len(train))}
    fold_models = {"lgb": [], "xgb": [], "cat": []}
    fold_reports = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        t_fold = time.time()
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_SEED + fold)
        inner_tr, inner_es = next(gss.split(tr_idx, groups=groups[tr_idx]))
        tr_i, es_i = tr_idx[inner_tr], tr_idx[inner_es]

        if have_pop_cols:
            fold_pop_stats = build_fold_popularity(
                groups[tr_idx], training_pairs_raw, item_pos, category, brand, n_items
            )
            X_tr = fold_X_with_fresh_popularity(X, feature_cols, tr_i, item_idx_rows, y, fold_pop_stats)
            X_es = fold_X_with_fresh_popularity(X, feature_cols, es_i, item_idx_rows, y, fold_pop_stats)
            X_val = fold_X_with_fresh_popularity(X, feature_cols, val_idx, item_idx_rows, y, fold_pop_stats)
        else:
            X_tr, X_es, X_val = X[tr_i], X[es_i], X[val_idx]

        n_pos, n_neg = (y[tr_i] == 1).sum(), (y[tr_i] == 0).sum()
        spw = n_neg / max(n_pos, 1)

        m_lgb = train_lgb(X_tr, y[tr_i], X_es, y[es_i], spw)
        m_xgb = train_xgb(X_tr, y[tr_i], X_es, y[es_i], spw)
        m_cat = train_cat(X_tr, y[tr_i], X_es, y[es_i], spw)

        p_lgb = m_lgb.predict_proba(X_val)[:, 1]
        p_xgb = m_xgb.predict_proba(X_val)[:, 1]
        p_cat = m_cat.predict_proba(X_val)[:, 1]
        oof["lgb"][val_idx], oof["xgb"][val_idx], oof["cat"][val_idx] = p_lgb, p_xgb, p_cat

        fold_models["lgb"].append(m_lgb)
        fold_models["xgb"].append(m_xgb)
        fold_models["cat"].append(m_cat)

        auc_blend = roc_auc_score(y[val_idx], (p_lgb + p_xgb + p_cat) / 3)
        print(f"[fold {fold}] n_train={len(tr_i):,} n_val={len(val_idx):,} "
              f"spw={spw:.2f} auc(avg)={auc_blend:.4f} ({time.time()-t_fold:.1f}s)")
        fold_reports.append({"fold": fold, "n_val": int(len(val_idx)), "auc_avg": float(auc_blend)})

    # ---- optional 4th ensemble member: LLM pairwise relevance score ----
    # Added 2026-07-11. The LLM score (10_llm_relevance.py, joined as the
    # llm_rel_score column by 04_build_features.py) is a fixed EXTERNAL predictor
    # with no access to labels and no fold dependence -- its "OOF" prediction is
    # simply its own score on every row, identical at train and inference. So it
    # can be dropped straight into the blend/stacking combiner as a 4th member
    # ("llm") with no leakage: unlike a GBDT, it was never trained on any fold,
    # so there is no train-fold row it has an unfair view of. It is ALSO already
    # a GBDT feature (get_feature_cols picked up llm_rel_score) -- using it both
    # ways is exactly the "both" combination requested. combine_predictions in
    # 07_predict.py mirrors this by feeding the submission llm_rel_score as the
    # "llm" member at inference.
    if LLM_REL_AS_ENSEMBLE_MEMBER and "llm_rel_score" in train.columns:
        oof["llm"] = train["llm_rel_score"].values.astype(np.float64)
        llm_auc = roc_auc_score(y, oof["llm"])
        print(f"[main] LLM relevance added as 4th ensemble member "
              f"(standalone OOF AUC={llm_auc:.4f}); combiner will weight it vs. lgb/xgb/cat")
    elif LLM_REL_AS_ENSEMBLE_MEMBER:
        print("[main] TY_LLM_REL_ENSEMBLE=1 but no llm_rel_score column in train_features "
              "-- run 10_llm_relevance.py + 04_build_features.py (TY_USE_LLM_RELEVANCE=1) first. "
              "Falling back to the 3-model lgb/xgb/cat ensemble.")

    print(f"[main] comparing blend-weight vs. stacking combiners on a nested holdout "
          f"({len(oof)} members: {', '.join(oof.keys())})...")
    ensemble = nested_compare_ensemble_methods(oof, y, groups, META_NESTED_HOLDOUT_FRAC, seed=RANDOM_SEED)
    oof_blend = ensemble.pop("final_oof_pred")
    best_thr, best_f1 = ensemble["threshold"], ensemble["oof_macro_f1_all_data"]

    # Save OOF blend + labels so NewPipeline/80_macrof1_scorer.py can measure
    # macro-F1 / tune the threshold locally (added 2026-07-15).
    np.save(MODEL_DIR / "oof_blend.npy", oof_blend.astype(np.float32))
    np.save(MODEL_DIR / "oof_y.npy", y.astype(np.int8))
    print(f"[main] saved OOF -> {MODEL_DIR}/oof_blend.npy , oof_y.npy "
          f"(measure: 80_macrof1_scorer.py --oof-proba oof_blend.npy --oof-labels oof_y.npy)")

    sorted_p = np.sort(oof_blend)[::-1]
    k = int(len(sorted_p) * TARGET_POS_RATE)
    density_thr = sorted_p[k] if k < len(sorted_p) else sorted_p[-1]
    density_f1 = f1_score(y, (oof_blend >= density_thr).astype(np.int8), average="macro")

    print(f"[main] chosen ensemble method: {ensemble['method']}")
    print(f"[main] OOF macro-F1 @ F1-optimal threshold {best_thr:.4f} = {best_f1:.4f}")
    print(f"[main] OOF macro-F1 @ density-matching ({TARGET_POS_RATE*100:.0f}%) threshold {density_thr:.4f} = {density_f1:.4f}")
    print(f"[main] chosen predicted-positive rate at F1-optimal thr = {(oof_blend >= best_thr).mean()*100:.1f}%")

    for algo in ("lgb", "xgb", "cat"):
        for i, m in enumerate(fold_models[algo]):
            if algo == "lgb":
                m.booster_.save_model(str(MODEL_DIR / f"lgb_fold{i}.txt"))
            elif algo == "xgb":
                m.save_model(str(MODEL_DIR / f"xgb_fold{i}.json"))
            else:
                m.save_model(str(MODEL_DIR / f"cat_fold{i}.cbm"))

    meta = {
        "feature_cols": feature_cols,
        "n_folds": N_FOLDS,
        "ensemble": ensemble,
        "threshold": float(best_thr),
        "density_threshold": float(density_thr),
        "oof_macro_f1": float(best_f1),
        "oof_macro_f1_density": float(density_f1),
        "fold_reports": fold_reports,
        "target_pos_rate": TARGET_POS_RATE,
        "hyperparameter_overrides_used": BEST_HYPERPARAMS,  # {} if hpo_search.py was never run
    }
    with open(MODEL_DIR / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[main] saved models + meta.json -> {MODEL_DIR}")
    print(f"[main] total time {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
