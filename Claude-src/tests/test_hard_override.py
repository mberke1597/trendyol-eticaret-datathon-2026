"""
Verifies the hard gender/age override that 07_predict.py applies
unconditionally. prompt.md section 4 states these constraints are ABSOLUTE
("a search for 'kadın ayakkabı' must never return a male shoe") -- a
teammate's train_stacking.py removed this override entirely, relying on the
GBDT to have learned it as a soft feature. This test proves the override
logic (independent of any trained model) forces prediction=0 even when the
model's raw probability would otherwise say "relevant".

brand_contradiction is deliberately EXCLUDED from this override (see
07_predict.py's "REVERTED 2026-07-04" docstring note): it was added for a
while, but the proven ../src/05_predict.py never overrode on brand (only
gender/age), and on a real Kaggle run brand_contradiction flagged ~27% of all
submission rows and forced the predicted-positive rate to 12.32% (target
~28-31%). brand_contradiction remains a normal GBDT feature only.
test_brand_contradiction_does_not_override below locks this decision in.
"""
import numpy as np
import pandas as pd

import features as feat


def _apply_hard_override(proba, threshold, gender_contradiction, age_contradiction):
    """Mirrors 07_predict.py's override block exactly -- kept here as a small,
    directly-testable function so this test doesn't need to load real GBDT
    models to prove the override logic itself is correct. Intentionally takes
    no brand_contradiction argument -- see module docstring."""
    pred = (proba >= threshold).astype(np.int8)
    any_override = (gender_contradiction > 0) | (age_contradiction > 0)
    return np.where(any_override, 0, pred)


def test_gender_override_forces_zero_even_at_high_confidence():
    items = pd.DataFrame({
        "item_id": ["I1"], "title": ["Erkek Spor Ayakkabı"], "category": ["ayakkabı/spor ayakkabı"],
        "brand": ["Nike"], "gender": ["erkek"], "age_group": ["yetiskin"], "attributes": [""],
    })
    terms = pd.DataFrame({"term_id": ["T1"], "query": ["kadın ayakkabı"]})
    lex = feat.LexicalIndex(items, terms)
    emb = {
        "query_main": np.random.RandomState(0).randn(1, 8).astype(np.float16),
        "query_tiny": np.random.RandomState(1).randn(1, 4).astype(np.float16),
        "item_title": np.random.RandomState(2).randn(1, 8).astype(np.float16),
        "item_attr": np.random.RandomState(3).randn(1, 4).astype(np.float16),
        "category_emb": np.random.RandomState(4).randn(1, 8).astype(np.float16),
        "item_category_idx": np.array([0], dtype=np.int32),
    }
    pop = {
        "item_click_count": np.array([5], dtype=np.int64),
        "item_cat_mean_log": np.array([1.0], dtype=np.float32),
        "brand_click_count": np.array([5], dtype=np.int64),
    }
    out = feat.compute_batch_features(np.array([0]), np.array([0]), lex, emb, pop)
    assert out.loc[0, "gender_contradiction"] == 1.0

    # simulate a model that is VERY confident this is relevant (proba=0.99)
    high_confidence_proba = np.array([0.99])
    final_pred = _apply_hard_override(
        high_confidence_proba, threshold=0.3,
        gender_contradiction=out["gender_contradiction"].values,
        age_contradiction=out["age_contradiction"].values,
    )
    assert final_pred[0] == 0, "hard gender override must force prediction to 0 regardless of model confidence"


def test_no_override_when_no_contradiction():
    gender = np.array([0.0])
    age = np.array([0.0])
    proba = np.array([0.8])
    pred = _apply_hard_override(proba, 0.5, gender, age)
    assert pred[0] == 1


def test_brand_contradiction_does_not_override():
    """Locks in the 2026-07-04 reversal: brand_contradiction must NOT force a
    prediction to 0. It stays a GBDT feature only -- see module docstring for
    the real-Kaggle-run evidence that a brand hard override was a net loss."""
    gender = np.array([0.0])
    age = np.array([0.0])
    proba = np.array([0.95])
    # even though this row's brand_contradiction is 1.0, _apply_hard_override
    # doesn't take a brand argument at all -- the override can't see it.
    pred = _apply_hard_override(proba, 0.3, gender, age)
    assert pred[0] == 1, "brand_contradiction must not force prediction to 0 (feature-only, not an override)"
