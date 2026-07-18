"""
Stage 61 — turn the TRAIN-VALIDATED hard contradictions into a MODEL FEATURE.

Why (from the feature audit): Claude-src already has soft conflict features
(color_conflict, material_conflict, size_conflict, gender_contradiction,
brand_contradiction). But:
  * color_conflict / material_conflict are NOISY — they fire on ~21% of genuinely
    relevant pairs (train), so the GBDT correctly learns to weight them softly.
    That is exactly why HARD colour/material flipping (v5/v6) lost on the LB.
  * The model has NO dedicated feature for the SAFE, high-precision contradictions
    (book subject, iPhone/Galaxy model, tire size, school grade, seating
    capacity, sleeve, season). Those fire on <5% of relevant pairs (train), i.e.
    they are near-perfect negative signals — but the model only sees them
    indirectly through generic lexical features.

This stage computes a single binary `spec_conflict` = 1 when ANY train-validated
(<5% mismatch) contradiction rule fires, for the labeled train set and the
submission set, and writes them as parquet keyed exactly like llm_rel_score so
04_build_features.py / 05_train.py pick it up as a normal GBDT feature.

Feeding it as a FEATURE (not a hard 1->0 flip) lets the model combine it with
behavioural/popularity evidence — strictly more information than the post-hoc
filter (stage 60), and it cannot delete a true positive outright the way a flip
can. The post-hoc filter (60) remains available as the simple, proven variant.

Outputs:
  CACHE/spec_conflict_train.parquet       [term_id, item_id, spec_conflict]
  CACHE/spec_conflict_submission.parquet  [id,               spec_conflict]

Run:  python 61_spec_conflict_feature.py
Then in Claude-src, join it the same way as any feature (or use NewPipeline
40_merge_features.py) and retrain 05_train.py.
"""
import importlib.util, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from config import DATA_DIR, CACHE_DIR, CLAUDE_CACHE_DIR
from data import load_train_labeled, load_submission_pairs

# import contradiction_mask from the digit-prefixed stage-60 module
_spec = importlib.util.spec_from_file_location(
    "cf60", HERE / "60_attribute_contradiction_filter.py")
_cf = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cf)

# train-validated safe ruleset (sinif kept as borderline; drop it for a stricter set)
SAFE_RULES = {"kol_boyu", "sezon", "lastik", "kitap", "cinsiyet", "sinif",
              "kapasite", "telefon", "galaxy", "numara", "hacim", "kisilik", "watt"}


def _compute(pairs, terms, items):
    d = pairs.merge(terms, on="term_id").merge(items, on="item_id")
    for c in ["query", "title", "category", "gender"]:
        d[c] = d[c].fillna("").str.lower()
    flip, _ = _cf.contradiction_mask(d["query"].values, d["title"].values,
                                     d["category"].values, d["gender"].values, SAFE_RULES)
    return flip.astype(np.int8)


def main():
    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")[["term_id", "query"]]
    items = pd.read_csv(f"{DATA_DIR}/items.csv", usecols=["item_id", "title", "category", "gender"])

    tr = load_train_labeled()[["term_id", "item_id"]]
    tr["spec_conflict"] = _compute(tr.copy(), terms, items)
    tr.to_parquet(CACHE_DIR / "spec_conflict_train.parquet", index=False)
    print(f"[61] train  spec_conflict=1 on {int(tr.spec_conflict.sum()):,}/{len(tr):,} "
          f"({100*tr.spec_conflict.mean():.2f}%) -> {CACHE_DIR}/spec_conflict_train.parquet")

    sb = load_submission_pairs()
    sb["spec_conflict"] = _compute(sb[["term_id", "item_id"]].copy(), terms, items)
    sb[["id", "spec_conflict"]].to_parquet(CACHE_DIR / "spec_conflict_submission.parquet", index=False)
    print(f"[61] submission spec_conflict=1 on {int(sb.spec_conflict.sum()):,}/{len(sb):,} "
          f"({100*sb.spec_conflict.mean():.2f}%) -> {CACHE_DIR}/spec_conflict_submission.parquet")

    # sanity: among labeled positives, spec_conflict should be RARE (that's the point)
    lab = load_train_labeled()
    if "label" in lab.columns:
        j = lab.merge(tr, on=["term_id", "item_id"], how="left")
        pos = j[j.label == 1]
        print(f"[61] VALIDATION: spec_conflict=1 among RELEVANT train pairs = "
              f"{100*pos.spec_conflict.mean():.2f}% (should be low; that's why it's a clean negative signal)")


if __name__ == "__main__":
    main()
