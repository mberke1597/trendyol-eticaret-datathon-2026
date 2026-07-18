"""
Pure-logic training helpers factored out of 05_train.py so they're testable
without importing lightgbm/xgboost/catboost (heavy binary deps not needed to
verify this logic -- see tests/test_popularity_leakage.py and
tests/test_threshold_scan.py, which import ONLY this module). Only depends on
numpy/pandas/scikit-learn.

Contains:
  - fold-safe popularity recompute (closes the cross-fold click leak, see
    ../DESIGN.md 2026-07-03 fix and this module's own docstrings below)
  - exact O(n log n) macro-F1 threshold scan
  - blend-weight grid search
  - nested (honest, non-optimistic) comparison between blend-weight and
    logistic-regression-stacking combiners
"""
import os
from itertools import product

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

NON_FEATURE_COLS = {"term_id", "item_id", "label", "neg_source", "id"}

# Popularity columns baked into train_features.parquet by 04_build_features.py are
# global (computed from the FULL training_pairs.csv) -- fine for the submission set
# (no labels to leak) but a real leak source for CV: GroupKFold only splits on
# term_id, so an item positive for term A (train fold) and term B (val fold) leaks
# A's click into B's validation row through item/brand/category aggregates, even
# after row-level leave-one-out (which only removes a row's OWN contribution, not
# other folds' contributions). ~7.8% of items are positive for 2+ terms. Fix:
# recompute these 3 columns per outer fold, from only that fold's own training
# term_id set (build_fold_popularity), and overwrite them in the per-fold X slices
# (fold_X_with_fresh_popularity) before training.
POP_COLS = ("item_click_log", "item_click_cat_rel", "brand_click_log")


def get_feature_cols(df):
    """All numeric columns except ids/labels.

    TY_DROP_CLICK_FEATURES=1 (2026-07-15 detective finding): additionally drop
    the click/popularity columns. In training, 100% of positives have click
    history *by construction* (positives ARE the click log) vs 27% of
    negatives — but only 21% of test rows have any click history, so the model
    leans on a near-label-proxy that mostly doesn't exist at test time and
    systematically underscores unclicked-but-relevant items. A/B one real LB
    submission with this flag before deciding to keep or drop them for good.
    05_train.py's fold-popularity recompute is automatically skipped when the
    columns are excluded (it guards on `have_pop_cols`)."""
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    if os.environ.get("TY_DROP_CLICK_FEATURES", "0") == "1":
        dropped = [c for c in cols if c in POP_COLS]
        cols = [c for c in cols if c not in POP_COLS]
        print(f"[get_feature_cols] TY_DROP_CLICK_FEATURES=1 -> dropped {dropped}")
    return cols


def load_item_catalog_for_popularity(data_dir):
    items = pd.read_csv(f"{data_dir}/items.csv", usecols=["item_id", "category", "brand"])
    item_pos = pd.Series(np.arange(len(items)), index=items["item_id"].values)
    category = items["category"].fillna("").values
    brand = items["brand"].fillna("").values
    return item_pos, category, brand


def build_fold_popularity(fold_term_ids, training_pairs, item_pos, category, brand, n_items):
    """Popularity stats computed ONLY from positive clicks whose term_id belongs
    to the current fold's training term set -- val-fold terms' clicks never
    enter these stats, which is what actually closes the cross-fold leak
    (row-level LOO alone does not)."""
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


def combine_ensemble_predictions(preds, ensemble_meta):
    """Member-generic ensemble combiner (moved here 2026-07-11 so it's testable
    without importing lightgbm/xgboost/catboost -- same "pure logic lives in
    training_utils" discipline as the threshold scan and stacking comparison).

    `preds` is a dict of member_name -> probability array, e.g.
    {"lgb":.., "xgb":.., "cat":..} and, when the optional LLM relevance stage is
    on, also "llm":.. . Which members are actually combined is driven purely by
    the keys 05_train.py wrote into blend_weights / meta_coef, so the SAME
    function handles the original 3-model ensemble and the 4-member (incl. LLM)
    one. 07_predict.combine_predictions is a thin alias of this."""
    if ensemble_meta["method"] == "stacking":
        coef = ensemble_meta["meta_coef"]
        z = ensemble_meta["meta_intercept"] + sum(coef[n] * preds[n] for n in coef)
        return 1.0 / (1.0 + np.exp(-z))
    w = ensemble_meta["blend_weights"]
    return sum(w[n] * preds[n] for n in w)


def best_threshold_for_macro_f1(y_true, proba):
    """O(n log n) exact scan over every possible cut point (sort predictions
    once, cumulative TP/FP/FN/TN, closed-form macro-F1 at each point) -- not a
    fixed grid of ~1000 thresholds. A teammate's train_stacking.py used a
    0.05-step grid over [0.10, 0.60] and never found thresholds above 0.60;
    this scan can never miss the true optimum regardless of where it falls."""
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


def search_blend_weights(oof_dict, y_true, idx, step=None):
    names = list(oof_dict.keys())
    # Grid size is (1/step + 1)**len(names); with 3 models a 0.1 step is 1,331
    # combos, but a 4th model (e.g. the LLM relevance score, 2026-07-11) makes
    # that 14,641 -- ~11x the threshold scans, minutes of wall time repeated
    # across the nested + final searches. Default to a coarser 0.2 step once
    # there are 4+ members so the combo count stays comparable (6**4=1,296),
    # while 2-3 members keep the original fine 0.1 grid unchanged.
    if step is None:
        step = 0.2 if len(names) >= 4 else 0.1
    best = None
    grid = np.arange(0, 1.0 + step / 2, step)
    y_sub = y_true[idx]
    for combo in product(grid, repeat=len(names)):
        if abs(sum(combo) - 1.0) > 1e-6:
            continue
        blend = sum(w * oof_dict[n][idx] for w, n in zip(combo, names))
        t, f1 = best_threshold_for_macro_f1(y_sub, blend)
        if best is None or f1 > best[0]:
            best = (f1, dict(zip(names, combo)), t)
    return best


def assign_term_folds(groups, n_folds):
    """Deterministically reproduces 05_train.py's GroupKFold(n_folds).split(...)
    fold assignment as a term_id -> fold_index dict, WITHOUT needing X/y (GroupKFold
    only uses `groups`). Used by 09_hard_negative_mining.py to know which fold's
    model is out-of-fold (never trained on) for each training term_id -- mining
    hard negatives with a model that already saw the term during training would
    leak and understate the term's real difficulty."""
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=n_folds)
    term_fold = {}
    dummy_X = np.zeros(len(groups))
    for fold, (_, val_idx) in enumerate(gkf.split(dummy_X, groups=groups)):
        for term in np.unique(groups[val_idx]):
            term_fold[term] = fold
    return term_fold


def select_hard_negatives(scores, is_positive, item_ids, threshold, max_per_term):
    """Pure selection logic for one term's candidate pool: keep candidates the
    model scores >= threshold that are NOT already a true positive for this term,
    capped at max_per_term (highest-scoring first -- these are the model's most
    confident mistakes, i.e. the most informative negatives to add). Factored out
    from 09_hard_negative_mining.py so it's unit-testable without FAISS/GBDT
    models -- see tests/test_hard_negative_mining.py.

    scores, is_positive: 1D arrays, same length as item_ids (one candidate per row).
    Returns: list of item_ids to add as new label=0 rows, highest-score-first."""
    scores = np.asarray(scores)
    is_positive = np.asarray(is_positive).astype(bool)
    item_ids = np.asarray(item_ids)

    candidate_mask = (~is_positive) & (scores >= threshold)
    cand_idx = np.nonzero(candidate_mask)[0]
    if len(cand_idx) == 0:
        return []
    order = cand_idx[np.argsort(-scores[cand_idx])]
    keep = order[:max_per_term]
    return item_ids[keep].tolist()


def nested_compare_ensemble_methods(oof, y, groups, meta_nested_holdout_frac, seed=42):
    """Honest comparison of blend-weight vs. logistic-regression stacking,
    scored on a GroupShuffleSplit(term_id) holdout neither combiner was fit on
    -- avoids scoring a meta-model on the same OOF rows it was trained on (the
    mild optimism found in a teammate's train_stacking.py review)."""
    names = list(oof.keys())
    gss = GroupShuffleSplit(n_splits=1, test_size=meta_nested_holdout_frac, random_state=seed)
    fit_idx, holdout_idx = next(gss.split(np.zeros(len(y)), y, groups=groups))

    blend_f1_fit, blend_weights, _ = search_blend_weights(oof, y, fit_idx)
    blend_holdout_pred = sum(w * oof[n][holdout_idx] for n, w in blend_weights.items())
    _, blend_holdout_f1 = best_threshold_for_macro_f1(y[holdout_idx], blend_holdout_pred)

    X_fit = np.column_stack([oof[n][fit_idx] for n in names])
    X_holdout = np.column_stack([oof[n][holdout_idx] for n in names])
    meta = LogisticRegression(C=1.0, solver="lbfgs", random_state=seed)
    meta.fit(X_fit, y[fit_idx])
    stack_holdout_pred = meta.predict_proba(X_holdout)[:, 1]
    _, stack_holdout_f1 = best_threshold_for_macro_f1(y[holdout_idx], stack_holdout_pred)

    if stack_holdout_f1 > blend_holdout_f1:
        X_all = np.column_stack([oof[n] for n in names])
        meta_final = LogisticRegression(C=1.0, solver="lbfgs", random_state=seed)
        meta_final.fit(X_all, y)
        final_pred = meta_final.predict_proba(X_all)[:, 1]
        thr, f1_all = best_threshold_for_macro_f1(y, final_pred)
        return {
            "method": "stacking",
            "meta_coef": dict(zip(names, meta_final.coef_[0].tolist())),
            "meta_intercept": float(meta_final.intercept_[0]),
            "threshold": thr,
            "oof_macro_f1_all_data": f1_all,
            "selection_holdout_macro_f1": stack_holdout_f1,
            "selection_holdout_macro_f1_rejected_blend": blend_holdout_f1,
            "final_oof_pred": final_pred,
        }
    else:
        f1_all, weights_all, thr = search_blend_weights(oof, y, np.arange(len(y)))
        return {
            "method": "blend",
            "blend_weights": weights_all,
            "threshold": thr,
            "oof_macro_f1_all_data": f1_all,
            "selection_holdout_macro_f1": blend_holdout_f1,
            "selection_holdout_macro_f1_rejected_stacking": stack_holdout_f1,
            "final_oof_pred": sum(w * oof[n] for n, w in weights_all.items()),
        }
