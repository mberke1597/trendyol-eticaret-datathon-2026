"""
Tests for the pure-logic pieces of 09_hard_negative_mining.py, factored into
training_utils.py so they're testable without FAISS/GBDT models (same rationale
as the rest of training_utils.py -- see its module docstring).
"""
import numpy as np

from training_utils import assign_term_folds, select_hard_negatives


def test_assign_term_folds_matches_sklearn_groupkfold_directly():
    """assign_term_folds must reproduce EXACTLY what GroupKFold(n_folds).split(...)
    would assign for the same groups array -- 09_hard_negative_mining.py relies on
    this to pick the out-of-fold model for each term_id without needing 05_train.py
    to persist anything extra."""
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(0)
    terms = np.array([f"T{i}" for i in range(40)])
    # each term appears a variable number of times, like real training_pairs.csv
    reps = rng.integers(1, 6, size=len(terms))
    groups = np.repeat(terms, reps)
    rng.shuffle(groups)

    term_fold = assign_term_folds(groups, n_folds=5)
    assert set(term_fold.keys()) == set(terms.tolist())

    # cross-check: every row's fold (looked up via term_fold) must match exactly
    # which sklearn split put it in as validation
    gkf = GroupKFold(n_splits=5)
    dummy_X = np.zeros(len(groups))
    sklearn_fold_of_row = np.full(len(groups), -1)
    for fold, (_, val_idx) in enumerate(gkf.split(dummy_X, groups=groups)):
        sklearn_fold_of_row[val_idx] = fold
    expected = np.array([term_fold[g] for g in groups])
    assert (sklearn_fold_of_row == expected).all()


def test_select_hard_negatives_excludes_true_positives():
    scores = np.array([0.9, 0.8, 0.95, 0.1])
    is_positive = np.array([False, True, False, False])  # item at idx 1 is a true positive
    item_ids = np.array(["I1", "I2", "I3", "I4"])
    kept = select_hard_negatives(scores, is_positive, item_ids, threshold=0.5, max_per_term=10)
    assert "I2" not in kept, "true positives must never be mined as hard negatives"
    assert set(kept) == {"I1", "I3"}


def test_select_hard_negatives_respects_threshold():
    scores = np.array([0.9, 0.3, 0.6])
    is_positive = np.array([False, False, False])
    item_ids = np.array(["I1", "I2", "I3"])
    kept = select_hard_negatives(scores, is_positive, item_ids, threshold=0.5, max_per_term=10)
    assert set(kept) == {"I1", "I3"}, "only candidates scoring >= threshold should be kept"


def test_select_hard_negatives_respects_max_per_term_highest_first():
    scores = np.array([0.7, 0.99, 0.6, 0.8])
    is_positive = np.array([False, False, False, False])
    item_ids = np.array(["third", "highest", "lowest", "second"])
    kept = select_hard_negatives(scores, is_positive, item_ids, threshold=0.5, max_per_term=2)
    assert kept == ["highest", "second"], "must keep the highest-scoring candidates first, capped at max_per_term"


def test_select_hard_negatives_empty_when_nothing_qualifies():
    scores = np.array([0.1, 0.2])
    is_positive = np.array([False, False])
    item_ids = np.array(["I1", "I2"])
    kept = select_hard_negatives(scores, is_positive, item_ids, threshold=0.5, max_per_term=10)
    assert kept == []
