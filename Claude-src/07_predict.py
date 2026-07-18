"""
Chunked inference over submission_features.parquet (3.36M rows) -> submission.csv.

Ensembles all N_FOLDS x 3-algorithm models saved by 05_train.py, combines them
via whichever method (blend weights or logistic-regression stacking) won the
nested comparison in 05_train.py (meta["ensemble"]["method"]), applies the
macro-F1-calibrated threshold, and then applies a HARD RULE OVERRIDE for
gender/age contradiction only.

This override is NOT optional and must never be removed: prompt.md section 4
states query gender/age constraints are *absolute* ("a search for 'kadın
ayakkabı' must never return a male shoe"). A teammate's train_stacking.py
removed this override entirely ("Hard overrides completely removed!") relying
on the GBDT to have learned it as a soft feature -- that is a real regression
against the competition's own stated requirement, not a stylistic choice, and
is why this script asserts the override actually fires (see
tests/test_hard_override.py) rather than just hoping the model learned it.

REVERTED 2026-07-04: brand_contradiction was ALSO force-zeroing predictions
here for a while (an addition beyond what the proven ../src/05_predict.py
does -- that script only ever overrode on gender/age, never brand, and scored
0.83 on the real leaderboard). Unlike gender/age, prompt.md does not state an
equivalent "absolute" rule for brand, and on a real Kaggle run this flagged
~27% of all submission rows and forced the predicted-positive rate down to
12.32% (target ~28-31%, per DESIGN.md's threshold-calibration lessons) --
almost certainly a net loss, not a net win. `brand_contradiction` is still
computed and used as a normal GBDT feature (features.py); it is simply no
longer a hard override here. If you want to re-try brand as a hard override,
A/B it against a real leaderboard submission first -- don't reintroduce it
on assumption alone.

Set TY_THRESHOLD env var to override the calibrated threshold (useful for
re-calibrating against public leaderboard feedback without retraining --
OOF calibration was NOT reliable on its own this competition; see DESIGN.md).

Pass --from-cached-proba to skip the ~30-minute model-inference pass entirely
and re-threshold output/submission_proba.npy from a previous --save-proba run
-- this real run's OOF said 19.0% positive at the F1-optimal threshold, but
the actual submission_pairs.csv predictions came out at only ~12.9% (before
the tiny gender/age override), confirming the same OOF-vs-real gap DESIGN.md
already documents. Re-running full inference just to try a different
threshold wastes ~30 minutes; --from-cached-proba lets you sweep thresholds
in seconds.

FIXED 2026-07-05: a real Kaggle run's kernel got silently restarted (RAM OOM)
during this stage. Root cause: `pd.read_parquet(submission_features.parquet)`
loaded ALL 3.36M rows x every feature column into memory in one shot BEFORE
the "chunked" loop even started -- only the `.predict()` calls were actually
chunked, not the read. Fixed by streaming the parquet file in CHUNK_SIZE-row
batches via `pyarrow.parquet.ParquetFile.iter_batches(columns=feature_cols)`,
so peak memory for the feature matrix is one chunk's worth, not all 3.36M
rows at once. The few small always-needed columns (id, gender_contradiction,
age_contradiction) are still read for all rows up front -- those three
columns for 3.36M rows are cheap (a few tens of MB), unlike the full feature
matrix.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CACHE_DIR, DATA_DIR, MODEL_DIR, OUTPUT_DIR  # noqa: E402
from training_utils import combine_ensemble_predictions  # noqa: E402

CHUNK_SIZE = 300_000

# Thin alias: the member-generic combiner lives in training_utils (pure, no
# lightgbm/xgboost/catboost import) so it's unit-testable there; 09_hard_negative
# _mining.py calls predict_mod.combine_predictions, kept working via this name.
combine_predictions = combine_ensemble_predictions


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


def predict_blend(X, models, ensemble_meta, p_llm=None):
    p_lgb = np.mean([m.predict(X) for m in models["lgb"]], axis=0)
    p_xgb = np.mean([m.predict_proba(X)[:, 1] for m in models["xgb"]], axis=0)
    p_cat = np.mean([m.predict_proba(X)[:, 1] for m in models["cat"]], axis=0)
    preds = {"lgb": p_lgb, "xgb": p_xgb, "cat": p_cat}
    # Only supply the LLM member if the trained combiner actually uses it (its
    # weights/coef will contain the "llm" key). If p_llm wasn't provided but the
    # combiner expects it, neutral-fill so we never KeyError -- but warn-worthy;
    # in the normal flow 04_build_features.py guarantees the column exists.
    combiner_members = (ensemble_meta.get("blend_weights") or ensemble_meta.get("meta_coef") or {}).keys()
    if "llm" in combiner_members:
        preds["llm"] = p_llm if p_llm is not None else np.full(len(p_lgb), 0.5, dtype=np.float32)
    return combine_predictions(preds, ensemble_meta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-proba", action="store_true",
                         help="Also save raw ensemble probabilities to output/submission_proba.npy "
                              "(consumed by 06_rescore_uncertain_band.py)")
    parser.add_argument("--from-cached-proba", action="store_true",
                         help="Skip the ~30-min model-inference pass and re-threshold "
                              "output/submission_proba.npy from a previous --save-proba run. "
                              "Use this to sweep TY_THRESHOLD values quickly.")
    args = parser.parse_args()

    t_start = time.time()
    with open(MODEL_DIR / "meta.json") as fh:
        meta = json.load(fh)
    feature_cols = meta["feature_cols"]
    threshold = float(os.environ.get("TY_THRESHOLD", meta["threshold"]))
    print(f"[main] ensemble method={meta['ensemble']['method']} threshold={threshold:.4f} "
          f"(meta default={meta['threshold']:.4f})")

    feat_path = f"{CACHE_DIR}/submission_features.parquet"
    pf = pq.ParquetFile(feat_path)
    n = pf.metadata.num_rows

    # Only the few small always-needed columns are loaded for ALL rows up
    # front -- id + the two override flags, a few tens of MB for 3.36M rows.
    # The (much larger) feature_cols matrix is streamed in chunks below and
    # never held in full (see "FIXED 2026-07-05" module docstring note).
    small_cols = pd.read_parquet(feat_path, columns=["id", "gender_contradiction", "age_contradiction"])

    proba_path = OUTPUT_DIR / "submission_proba.npy"
    if args.from_cached_proba:
        assert proba_path.exists(), (
            f"--from-cached-proba given but {proba_path} doesn't exist -- "
            "run once with --save-proba first."
        )
        proba = np.load(proba_path)
        assert len(proba) == n, "cached submission_proba.npy length doesn't match submission_features.parquet"
        print(f"[main] loaded cached probabilities from {proba_path} ({n:,} rows) -- skipping model inference")
    else:
        print(f"[main] {n:,} rows to predict (streaming feature parquet in {CHUNK_SIZE:,}-row batches)")
        models = load_models(meta)
        proba = np.empty(n, dtype=np.float32)
        t0 = time.time()
        start = 0
        # If the trained combiner includes the LLM relevance member, pull its
        # per-row score straight out of the streamed chunk (llm_rel_score is a
        # feature column, so it's already in feature_cols / the parquet -- no
        # extra read needed). It's used BOTH as a GBDT feature (inside X) and as
        # the combiner's "llm" member, exactly matching 05_train.py.
        combiner_members = (meta["ensemble"].get("blend_weights")
                            or meta["ensemble"].get("meta_coef") or {}).keys()
        use_llm_member = "llm" in combiner_members
        for batch in pf.iter_batches(batch_size=CHUNK_SIZE, columns=feature_cols):
            bdf = batch.to_pandas()
            X = bdf[feature_cols].values.astype(np.float32)
            p_llm = bdf["llm_rel_score"].values.astype(np.float32) if use_llm_member else None
            end = start + len(X)
            proba[start:end] = predict_blend(X, models, meta["ensemble"], p_llm=p_llm)
            print(f"  ...predicted {end:,}/{n:,} ({time.time()-t0:.1f}s)")
            start = end
            del X, bdf, batch
        assert start == n, f"streamed {start:,} rows but expected {n:,} -- parquet row count mismatch"

        if args.save_proba:
            np.save(proba_path, proba)
            print(f"[main] saved raw probabilities -> {proba_path}")

    pred = (proba >= threshold).astype(np.int8)

    # ---- HARD gender/age override only -- see module docstring, never remove ----
    # brand_contradiction is deliberately NOT part of this override (see the
    # "REVERTED 2026-07-04" note above) -- it remains a normal GBDT feature only.
    gender_override = small_cols["gender_contradiction"].values > 0
    age_override = small_cols["age_contradiction"].values > 0
    any_override = gender_override | age_override
    n_overridden = int((any_override & (pred == 1)).sum())
    pred = np.where(any_override, 0, pred)
    print(f"[main] hard gender/age override flipped {n_overridden:,} predictions to 0 "
          f"(gender={int(gender_override.sum()):,}, age={int(age_override.sum()):,} rows flagged; "
          f"brand_contradiction is a feature only, not an override -- see module docstring)")

    # ---- min-K positives per term (TY_MIN_POS_PER_TERM, default 0 = off) ----
    # Detective finding 2026-07-15: 94.4% of test terms have exactly 100
    # candidate rows (retriever top-K lists), yet 5.5% of terms got ZERO
    # predicted positives in the best 0.894 submission -- implausible for real
    # user queries whose top-100 came from a live retriever, and concentrated
    # F1_relevant recall loss. With TY_MIN_POS_PER_TERM=K (recommended: 3),
    # any term with fewer than K positives gets its top-K highest-probability
    # rows flipped to 1 (hard gender/age-overridden rows are never flipped).
    # A/B this against a real LB submission like every other change.
    min_pos_per_term = int(os.environ.get("TY_MIN_POS_PER_TERM", "0"))
    if min_pos_per_term > 0:
        pairs_term = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv", usecols=["id", "term_id"])
        term_of_id = pd.Series(pairs_term["term_id"].values, index=pairs_term["id"].values)
        row_terms = term_of_id.reindex(small_cols["id"].values).values
        eligible = ~any_override
        df_mk = pd.DataFrame({
            "row": np.arange(len(pred)),
            "term": row_terms,
            "proba": np.where(eligible, proba, -1.0),  # overridden rows sort last, never picked
            "pred": pred,
        })
        pos_per_term = df_mk.groupby("term")["pred"].transform("sum")
        needy = df_mk[pos_per_term < min_pos_per_term]
        # top-K probability rows within each needy term
        topk_rows = (
            needy.sort_values("proba", ascending=False)
                 .groupby("term", sort=False)
                 .head(min_pos_per_term)["row"].values
        )
        flip_rows = topk_rows[(pred[topk_rows] == 0) & eligible[topk_rows]]
        pred[flip_rows] = 1
        n_terms_fixed = needy["term"].nunique()
        print(f"[main] min-{min_pos_per_term} positives/term: flipped {len(flip_rows):,} rows to 1 "
              f"across {n_terms_fixed:,} low-recall terms "
              f"(new positive rate {pred.mean()*100:.2f}%)")

    submission = pd.DataFrame({"id": small_cols["id"].values, "prediction": pred})

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
