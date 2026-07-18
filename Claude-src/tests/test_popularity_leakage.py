"""
Synthetic-catalog test for the cross-fold popularity leakage fix (see
../DESIGN.md 2026-07-03 fix and training_utils.py module docstring).

Scenario: item I1 is a true positive for both term A and term B. If A is in
the training fold and B is in the validation fold, a GLOBAL popularity count
would let A's click leak into B's validation row (I1 looks "popular" in B's
row even though, from B's fold-honest point of view, I1's only relevant click
is the one *within this fold's own training term set*). build_fold_popularity
must exclude B's contribution when computing stats for a fold whose training
set is {A}.
"""
import numpy as np
import pandas as pd

import training_utils as tu


def test_fold_popularity_excludes_val_fold_terms_click():
    # 4-item catalog, all same category/brand for simplicity
    category = np.array(["cat1", "cat1", "cat1", "cat1"])
    brand = np.array(["b1", "b1", "b1", "b1"])
    item_pos = pd.Series(np.arange(4), index=["I1", "I2", "I3", "I4"])
    n_items = 4

    # I1 is positive for term A AND term B; I2 positive for term A only; I3 for term B only.
    training_pairs = pd.DataFrame({
        "term_id": ["A", "A", "B", "B"],
        "item_id": ["I1", "I2", "I1", "I3"],
    })

    # Fold where the TRAINING term set is {A} only (B is held out in validation)
    fold_term_ids_train_only_A = ["A"]
    stats = tu.build_fold_popularity(
        fold_term_ids_train_only_A, training_pairs, item_pos, category, brand, n_items
    )
    # I1's count should be 1 (only A's click), NOT 2 (which would include B's leak)
    assert stats["item_click_count"][0] == 1, "term B's click leaked into a fold that only trains on term A"
    assert stats["item_click_count"][1] == 1  # I2, only A
    assert stats["item_click_count"][2] == 0  # I3, only B -- excluded entirely
    assert stats["item_click_count"][3] == 0  # I4, never clicked


def test_fold_popularity_includes_all_terms_when_both_in_training_set():
    category = np.array(["cat1", "cat1", "cat1", "cat1"])
    brand = np.array(["b1", "b1", "b1", "b1"])
    item_pos = pd.Series(np.arange(4), index=["I1", "I2", "I3", "I4"])
    n_items = 4
    training_pairs = pd.DataFrame({
        "term_id": ["A", "A", "B", "B"],
        "item_id": ["I1", "I2", "I1", "I3"],
    })
    stats = tu.build_fold_popularity(["A", "B"], training_pairs, item_pos, category, brand, n_items)
    assert stats["item_click_count"][0] == 2  # I1 clicked by both A and B now legitimately in-fold


def test_fold_x_with_fresh_popularity_applies_row_level_loo():
    feature_cols = ["item_click_log", "item_click_cat_rel", "brand_click_log", "sim_title"]
    X = np.zeros((3, 4), dtype=np.float32)
    X[:, 3] = [0.1, 0.2, 0.3]  # sim_title untouched by popularity recompute
    item_idx_rows = np.array([0, 0, 1])  # rows 0,1 -> item I1 (idx 0); row 2 -> item I2 (idx 1)
    labels = np.array([1, 0, 1], dtype=np.int8)  # row 0 is itself a positive click on I1
    pop_stats = {
        "item_click_count": np.array([3, 1, 0, 0], dtype=np.int64),
        "item_cat_mean_log": np.array([0.5, 0.2, 0.0, 0.0], dtype=np.float32),
        "brand_click_count": np.array([3, 1, 0, 0], dtype=np.int64),
    }
    row_idx = np.array([0, 1, 2])
    Xc = tu.fold_X_with_fresh_popularity(X, feature_cols, row_idx, item_idx_rows, labels, pop_stats)
    # row 0: item I1 has click_count=3, but row 0 IS one of those 3 clicks (label=1) -> LOO subtracts 1 -> 2
    assert np.isclose(Xc[0, 0], np.log1p(2))
    # row 1: item I1 again, but row 1 is itself a negative (label=0) -> no LOO subtraction -> 3
    assert np.isclose(Xc[1, 0], np.log1p(3))
    # sim_title column must be untouched
    assert np.allclose(Xc[:, 3], [0.1, 0.2, 0.3])
