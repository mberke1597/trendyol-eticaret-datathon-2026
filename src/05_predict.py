"""
Chunked inference over submission_features.parquet (3.36M rows) -> submission.csv.

Ensembles all N_FOLDS x 3-algorithm models saved by 04_train.py (bag folds per
algorithm, then blend algorithms with the OOF-searched weights), applies the
macro-F1-calibrated threshold, and then applies a hard rule override for
gender/age contradiction: prompt.md section 4 states query gender/age
constraints are *absolute* ("a search for 'kadın ayakkabı' must never return a
male shoe"), so even though the GBDT already sees `gender_contradiction` /
`age_contradiction` as features, we don't rely on it learning that perfectly --
we force prediction=0 whenever the (noise-robust, title-fallback) constraint
checker fires, regardless of model score.

Set TY_THRESHOLD env var to override the calibrated threshold (useful for
re-calibrating against public leaderboard feedback without retraining).
"""
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CACHE_DIR, DATA_DIR, MODEL_DIR, OUTPUT_DIR  # noqa: E402

CHUNK_SIZE = 300_000


def load_models(meta):
    models = {"lgb": [], "xgb": [], "cat": []}
    for i in range(meta["n_folds"]):
        m = lgb.Booster(model_file=str(MODEL_DIR / f"lgb_fold{i}.txt"))
        models["lgb"].append(m)
        m = xgb.XGBClassifier()
        m.load_model(str(MODEL_DIR / f"xgb_fold{i}.json"))
        models["xgb"].append(m)
        m = CatBoostClassifier()
        m.load_model(str(MODEL_DIR / f"cat_fold{i}.cbm"))
        models["cat"].append(m)
    return models


def predict_blend(X, models, weights):
    p_lgb = np.mean([m.predict(X) for m in models["lgb"]], axis=0)
    p_xgb = np.mean([m.predict_proba(X)[:, 1] for m in models["xgb"]], axis=0)
    p_cat = np.mean([m.predict_proba(X)[:, 1] for m in models["cat"]], axis=0)
    return weights["lgb"] * p_lgb + weights["xgb"] * p_xgb + weights["cat"] * p_cat


def main():
    t_start = time.time()
    with open(MODEL_DIR / "meta.json") as fh:
        meta = json.load(fh)
    feature_cols = meta["feature_cols"]
    threshold = float(os.environ.get("TY_THRESHOLD", meta["threshold"]))
    print(f"[main] using threshold={threshold:.4f} (meta default={meta['threshold']:.4f})")

    models = load_models(meta)
    sub_feat = pd.read_parquet(f"{CACHE_DIR}/submission_features.parquet")
    n = len(sub_feat)
    print(f"[main] {n:,} rows to predict")

    proba = np.empty(n, dtype=np.float32)
    t0 = time.time()
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        X = sub_feat[feature_cols].iloc[start:end].values.astype(np.float32)
        proba[start:end] = predict_blend(X, models, meta["blend_weights"])
        print(f"  ...predicted {end:,}/{n:,} ({time.time()-t0:.1f}s)")

    pred = (proba >= threshold).astype(np.int8)

    gender_override = sub_feat["gender_contradiction"].values > 0
    age_override = sub_feat["age_contradiction"].values > 0
    n_overridden = int(((gender_override | age_override) & (pred == 1)).sum())
    pred = np.where(gender_override | age_override, 0, pred)
    print(f"[main] hard gender/age constraint override flipped {n_overridden:,} predictions to 0")

    submission = pd.DataFrame({"id": sub_feat["id"].values, "prediction": pred})

    # sanity: exact id-set / row-count match against submission_pairs.csv
    sub_pairs_ids = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv", usecols=["id"])
    assert len(submission) == len(sub_pairs_ids), "row count mismatch vs submission_pairs.csv"
    assert set(submission["id"]) == set(sub_pairs_ids["id"]), "id set mismatch vs submission_pairs.csv"

    out_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"[main] predicted positive rate = {pred.mean()*100:.2f}%")
    print(f"[main] wrote {out_path}")
    print(f"[main] total time {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
