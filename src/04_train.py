"""
GroupKFold (by term_id) CV training of a 3-model GBDT ensemble (LightGBM,
XGBoost, CatBoost), OOF blend-weight search, and macro-F1 threshold calibration.

Zero query leakage: term sets in submission_pairs.csv are 100% disjoint from
training_pairs.csv (see Trendyol_EDA_Raporu.docx section 6.1), so a random
row split would let a term's other rows leak into validation and overstate
CV score. GroupKFold on term_id reproduces the cold-start term regime the
model actually faces at test time.

Within each outer fold we carve a further inner GroupShuffleSplit purely for
early stopping, so the outer (OOF) fold is never touched during training --
the OOF macro-F1 / threshold are an honest, leakage-free estimate.
"""
import json
import sys
import time
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CACHE_DIR, DATA_DIR, MODEL_DIR, N_FOLDS, RANDOM_SEED, TARGET_POS_RATE  # noqa: E402

# XGBoost/CatBoost pip wheels ship prebuilt GPU support (no special install needed) --
# used when available (e.g. Kaggle GPU T4 x2) to cut the 5-fold x 3-model training time
# down from CPU-hist. LightGBM's PyPI wheel does *not* include GPU support without a
# custom build, so it stays on CPU (it's the fastest of the three there anyway).
try:
    import torch
    USE_GPU = torch.cuda.is_available()
except ImportError:
    USE_GPU = False

NON_FEATURE_COLS = {"term_id", "item_id", "label", "neg_source", "id"}

# Popularity columns baked into train_features.parquet by 03_build_features.py are
# computed from the FULL training_pairs.csv (see build_popularity() there) -- fine for
# the submission set (no labels to leak), but a real problem for CV: GroupKFold only
# splits on term_id, so an item that is positive for term A (train fold) and term B
# (val fold) leaks A's click into B's validation row through item/brand/category
# aggregates, even after the existing row-level leave-one-out correction (which only
# removes a row's *own* contribution, not other folds' contributions). Concretely,
# ~7.8% of items are positive for 2+ terms (Trendyol_EDA_Raporu.docx section 5.1), and
# item_cat_mean_log had *no* LOO correction at all (fully global category means), so
# category-level dilution reaches further than just those items.
# Fix: recompute these 3 columns per outer fold, using only the training fold's own
# term_id set, and overwrite them in the per-fold X slices below -- the values baked
# into train_features.parquet are effectively placeholders once this runs.
POP_COLS = ("item_click_log", "item_click_cat_rel", "brand_click_log")


def get_feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]


def load_item_catalog_for_popularity():
    """category/brand per item, plus a positional item_id -> row-index map, both
    aligned the same way 03_build_features.py aligns things (by item_id string)."""
    items = pd.read_csv(f"{DATA_DIR}/items.csv", usecols=["item_id", "category", "brand"])
    item_pos = pd.Series(np.arange(len(items)), index=items["item_id"].values)
    category = items["category"].fillna("").values
    brand = items["brand"].fillna("").values
    return item_pos, category, brand


def build_fold_popularity(fold_term_ids, training_pairs, item_pos, category, brand, n_items):
    """Popularity stats computed ONLY from positive clicks whose term_id belongs to the
    current fold's training term set -- val-fold terms' clicks never enter these stats,
    which is what actually closes the cross-fold leak (row-level LOO alone does not)."""
    fold_term_set = set(fold_term_ids)
    sub = training_pairs[training_pairs["term_id"].isin(fold_term_set)]
    sub_item_pos = item_pos.reindex(sub["item_id"].values).values
    click_counts = np.zeros(n_items, dtype=np.int64)
    valid = ~np.isnan(sub_item_pos)
    if valid.any():
        vc = pd.Series(sub_item_pos[valid].astype(int)).value_counts()
        click_counts[vc.index.values] = vc.values
    cat_mean_log = pd.Series(np.log1p(click_counts)).groupby(category).transform("mean").values.astype(np.float32)
    brand_click_count = pd.Series(click_counts).groupby(brand).transform("sum").values.astype(np.int64)
    return {
        "item_click_count": click_counts,
        "item_cat_mean_log": cat_mean_log,
        "brand_click_count": brand_click_count,
    }


def fold_X_with_fresh_popularity(X, feature_cols, row_idx, item_idx_rows, labels, pop_stats):
    """Copy of X[row_idx] with POP_COLS recomputed from this fold's leakage-free
    pop_stats (row-level LOO still applied on top, for the row's own label)."""
    Xc = X[row_idx].copy()
    sel_item_idx = item_idx_rows[row_idx]
    loo = labels[row_idx].astype(np.float32)
    item_click_count = np.clip(pop_stats["item_click_count"][sel_item_idx].astype(np.float32) - loo, 0, None)
    brand_click_count = np.clip(pop_stats["brand_click_count"][sel_item_idx].astype(np.float32) - loo, 0, None)
    fresh = {
        "item_click_log": np.log1p(item_click_count),
        "brand_click_log": np.log1p(brand_click_count),
    }
    fresh["item_click_cat_rel"] = fresh["item_click_log"] - pop_stats["item_cat_mean_log"][sel_item_idx]
    for col in POP_COLS:
        if col in feature_cols:
            Xc[:, feature_cols.index(col)] = fresh[col]
    return Xc


def train_lgb(X_tr, y_tr, X_es, y_es, spw):
    model = lgb.LGBMClassifier(
        n_estimators=3000, learning_rate=0.05, num_leaves=63, max_depth=-1,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=30,
        scale_pos_weight=spw, random_state=RANDOM_SEED, n_jobs=-1, verbosity=-1,
    )
    model.fit(
        X_tr, y_tr, eval_set=[(X_es, y_es)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def train_xgb(X_tr, y_tr, X_es, y_es, spw):
    model = xgb.XGBClassifier(
        n_estimators=3000, learning_rate=0.05, max_depth=8, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=spw,
        tree_method="hist", device="cuda" if USE_GPU else "cpu",
        random_state=RANDOM_SEED, n_jobs=-1,
        early_stopping_rounds=100, eval_metric="logloss",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
    return model


def train_cat(X_tr, y_tr, X_es, y_es, spw):
    model = CatBoostClassifier(
        iterations=3000, learning_rate=0.05, depth=8, l2_leaf_reg=3.0,
        class_weights=[1.0, spw], random_seed=RANDOM_SEED, verbose=False,
        early_stopping_rounds=100, eval_metric="Logloss",
        task_type="GPU" if USE_GPU else "CPU", devices="0" if USE_GPU else None,
    )
    model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_es, y_es), use_best_model=True)
    return model


def best_threshold_for_macro_f1(y_true, proba):
    """O(n log n) exact scan over every possible cut point (sort once, cumsum TP/FP/FN/TN
    for both classes), instead of calling sklearn.f1_score at ~1000 fixed thresholds --
    at ~1.8M rows and dozens of blend-weight combos the naive version would be far too slow."""
    n = len(y_true)
    order = np.argsort(-proba, kind="mergesort")
    y_sorted = y_true[order].astype(np.float64)
    proba_sorted = proba[order]

    total_pos = y_sorted.sum()
    total_neg = n - total_pos
    k = np.arange(1, n + 1, dtype=np.float64)
    TP = np.cumsum(y_sorted)
    FP = k - TP
    FN = total_pos - TP
    TN = total_neg - FP

    with np.errstate(divide="ignore", invalid="ignore"):
        prec_pos = np.where((TP + FP) > 0, TP / (TP + FP), 0.0)
        rec_pos = np.where((TP + FN) > 0, TP / (TP + FN), 0.0)
        f1_pos = np.where((prec_pos + rec_pos) > 0, 2 * prec_pos * rec_pos / (prec_pos + rec_pos), 0.0)
        pred_neg = n - k
        prec_neg = np.where(pred_neg > 0, TN / pred_neg, 0.0)
        rec_neg = np.where(total_neg > 0, TN / total_neg, 0.0)
        f1_neg = np.where((prec_neg + rec_neg) > 0, 2 * prec_neg * rec_neg / (prec_neg + rec_neg), 0.0)
    macro_f1 = (f1_pos + f1_neg) / 2
    best_i = int(np.argmax(macro_f1))
    return float(proba_sorted[best_i]), float(macro_f1[best_i])


def search_blend_weights(oof_dict, y_true, step=0.1):
    names = list(oof_dict.keys())
    best = None
    grid = np.arange(0, 1.0 + step / 2, step)
    for combo in product(grid, repeat=len(names)):
        if abs(sum(combo) - 1.0) > 1e-6:
            continue
        blend = sum(w * oof_dict[n] for w, n in zip(combo, names))
        t, f1 = best_threshold_for_macro_f1(y_true, blend)
        if best is None or f1 > best[0]:
            best = (f1, dict(zip(names, combo)), t)
    return best  # (macro_f1, weights, threshold)


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

    # ---- fold-aware popularity setup (see POP_COLS comment above) ----
    have_pop_cols = any(c in feature_cols for c in POP_COLS)
    if have_pop_cols:
        item_pos, category, brand = load_item_catalog_for_popularity()
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
        # inner split of the outer-train portion only, for early stopping
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_SEED + fold)
        inner_tr, inner_es = next(gss.split(tr_idx, groups=groups[tr_idx]))
        tr_i, es_i = tr_idx[inner_tr], tr_idx[inner_es]

        if have_pop_cols:
            # Recompute item/brand/category popularity from *this fold's* training
            # term set only -- val-fold terms' clicks are excluded by construction,
            # closing the cross-fold leak described above. Applied to tr_i/es_i/val_idx
            # alike so train and validation features come from the same distribution.
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

    print("[main] searching blend weights + threshold on OOF...")
    best_f1, best_weights, best_thr = search_blend_weights(oof, y)
    oof_blend = sum(w * oof[n] for n, w in best_weights.items())
    density_thr, density_f1 = None, None
    # sanity check vs. the ~30% click-density prior mentioned in prompt.md
    sorted_p = np.sort(oof_blend)[::-1]
    k = int(len(sorted_p) * TARGET_POS_RATE)
    density_thr = sorted_p[k] if k < len(sorted_p) else sorted_p[-1]
    density_f1 = f1_score(y, (oof_blend >= density_thr).astype(np.int8), average="macro")

    print(f"[main] best blend weights: {best_weights}")
    print(f"[main] OOF macro-F1 @ F1-optimal threshold {best_thr:.4f} = {best_f1:.4f}")
    print(f"[main] OOF macro-F1 @ density-matching ({TARGET_POS_RATE*100:.0f}%) threshold {density_thr:.4f} = {density_f1:.4f}")
    print(f"[main] chosen predicted-positive rate at F1-optimal thr = {(oof_blend >= best_thr).mean()*100:.1f}%")

    # ---------------- persist ----------------
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
        "blend_weights": best_weights,
        "threshold": float(best_thr),
        "density_threshold": float(density_thr),
        "oof_macro_f1": float(best_f1),
        "oof_macro_f1_density": float(density_f1),
        "fold_reports": fold_reports,
        "target_pos_rate": TARGET_POS_RATE,
    }
    with open(MODEL_DIR / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[main] saved models + meta.json -> {MODEL_DIR}")
    print(f"[main] total time {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
