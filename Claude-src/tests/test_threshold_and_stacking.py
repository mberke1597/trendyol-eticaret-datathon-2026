import numpy as np
from sklearn.metrics import f1_score

import training_utils as tu


def _brute_force_best_threshold(y_true, proba):
    """Reference implementation: scan sklearn.f1_score at every distinct proba
    value. Slow (O(n) sklearn calls) but unambiguously correct -- used only to
    cross-check the O(n log n) exact-scan implementation in training_utils.py."""
    best_f1, best_t = -1.0, 0.5
    for t in np.unique(proba):
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred, average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def test_exact_threshold_scan_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.random(n) < 0.3).astype(np.int8)
    proba = np.clip(y * 0.5 + rng.random(n) * 0.6, 0, 1)  # noisy but informative scores
    _, exact_f1 = tu.best_threshold_for_macro_f1(y, proba)
    _, brute_f1 = _brute_force_best_threshold(y, proba)
    assert np.isclose(exact_f1, brute_f1, atol=1e-9)


def test_exact_threshold_scan_handles_all_same_label():
    y = np.zeros(20, dtype=np.int8)
    proba = np.random.default_rng(1).random(20)
    thr, f1 = tu.best_threshold_for_macro_f1(y, proba)
    assert 0.0 <= f1 <= 1.0  # must not crash / NaN on degenerate input


def test_search_blend_weights_returns_valid_weights():
    rng = np.random.default_rng(2)
    n = 300
    y = (rng.random(n) < 0.3).astype(np.int8)
    oof = {
        "lgb": np.clip(y * 0.4 + rng.random(n) * 0.6, 0, 1),
        "xgb": np.clip(y * 0.5 + rng.random(n) * 0.5, 0, 1),
        "cat": np.clip(y * 0.3 + rng.random(n) * 0.7, 0, 1),
    }
    f1, weights, thr = tu.search_blend_weights(oof, y, np.arange(n))
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert 0.0 <= f1 <= 1.0


def test_nested_compare_ensemble_methods_returns_valid_choice():
    rng = np.random.default_rng(3)
    n = 600
    y = (rng.random(n) < 0.3).astype(np.int8)
    groups = np.array([f"term_{i%50}" for i in range(n)])  # 50 groups so GroupShuffleSplit has room
    oof = {
        "lgb": np.clip(y * 0.4 + rng.random(n) * 0.6, 0, 1),
        "xgb": np.clip(y * 0.45 + rng.random(n) * 0.55, 0, 1),
        "cat": np.clip(y * 0.35 + rng.random(n) * 0.65, 0, 1),
    }
    result = tu.nested_compare_ensemble_methods(oof, y, groups, meta_nested_holdout_frac=0.2, seed=3)
    assert result["method"] in ("blend", "stacking")
    assert "threshold" in result
    assert "oof_macro_f1_all_data" in result
    assert "selection_holdout_macro_f1" in result
    assert len(result["final_oof_pred"]) == n
