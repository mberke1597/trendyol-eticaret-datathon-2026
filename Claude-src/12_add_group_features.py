"""
Stage 12 — per-term (candidate-list) group features.

WHY (detective finding, 2026-07-15): 94.4% of submission_pairs.csv terms have
EXACTLY 100 candidate rows — the test set is per-query retriever top-K output,
not i.i.d. pairs. Relevance is partly RELATIVE: "is this item a better match
than the other 99 candidates for the same query?" The pipeline scored every row
independently and ignored this list structure entirely (a per-term signal the
top teams almost certainly use). In the best 0.894 submission, 5.5% of terms
had ZERO predicted positives — implausible for real user queries whose top-100
came from a live retriever.

WHAT: for a handful of strong base signals, add within-term relative features:
  <col>_pct   — percentile rank of the row's value inside its term's candidates
  <col>_z     — (value - term mean) / term std
plus term_n_candidates (list size) and term_sim_title_mean (how "tight" the
term's candidate list is — ambiguous 1-word queries have loose lists).

No label is used anywhere -> no leakage. For the TRAIN side the "list" is the
mined candidate set (~75 rows/term) instead of a true top-100 — distributions
differ, which is exactly why we use rank/z (scale-free) rather than raw gaps.

RUN (after 04_build_features.py, before 05_train.py):
  python 12_add_group_features.py
It rewrites train_features.parquet and submission_features.parquet in place
(originals backed up to *.bak.parquet once). 05_train.py picks the new columns
up automatically via get_feature_cols().
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CACHE_DIR, DATA_DIR  # noqa: E402

# strong, cheap, always-present base signals worth ranking within the term
GROUP_BASE_COLS = ["sim_title", "sim_max_title_cat", "word_recall", "char_jaccard"]


def add_group_features(df, term_ids):
    g = pd.Series(term_ids, name="term_id")
    grp = df.groupby(g.values, sort=False)
    out = {}
    for col in GROUP_BASE_COLS:
        if col not in df.columns:
            print(f"  [warn] {col} missing, skipping")
            continue
        v = df[col].astype(np.float32)
        gv = v.groupby(g.values, sort=False)
        # percentile rank in [0,1] within the term's candidate list
        out[f"{col}_pct"] = (gv.rank(method="average", pct=True)).astype(np.float32)
        mean = gv.transform("mean")
        std = gv.transform("std").fillna(0.0)
        out[f"{col}_z"] = ((v - mean) / (std + 1e-6)).astype(np.float32)
    out["term_n_candidates"] = grp[df.columns[0]].transform("size").astype(np.float32)
    if "sim_title" in df.columns:
        out["term_sim_title_mean"] = (
            df["sim_title"].astype(np.float32).groupby(g.values, sort=False).transform("mean")
        ).astype(np.float32)
    for k, v in out.items():
        df[k] = v.values
    return df, list(out.keys())


def process(path, term_ids, label):
    t0 = time.time()
    df = pd.read_parquet(path)
    assert len(df) == len(term_ids), f"{label}: row count mismatch ({len(df)} vs {len(term_ids)})"
    bak = path.with_suffix(".bak.parquet")
    if not bak.exists():
        shutil.copy(path, bak)
        print(f"  [{label}] backed up original -> {bak.name}")
    df, new_cols = add_group_features(df, term_ids)
    df.to_parquet(path, index=False)
    print(f"  [{label}] +{len(new_cols)} group features {new_cols} "
          f"-> {path.name} ({time.time()-t0:.1f}s, shape {df.shape})")


def main():
    train_path = Path(f"{CACHE_DIR}/train_features.parquet")
    sub_path = Path(f"{CACHE_DIR}/submission_features.parquet")

    print("[12] train side (term_id is already a column)...")
    tr_terms = pd.read_parquet(train_path, columns=["term_id"])["term_id"].values
    process(train_path, tr_terms, "train")

    print("[12] submission side (term_id joined from submission_pairs.csv by id)...")
    sub_ids = pd.read_parquet(sub_path, columns=["id"])["id"].values
    pairs = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv", usecols=["id", "term_id"])
    id_to_term = pd.Series(pairs["term_id"].values, index=pairs["id"].values)
    sub_terms = id_to_term.reindex(sub_ids).values
    assert not pd.isna(sub_terms).any(), "some submission feature ids missing from submission_pairs.csv"
    process(sub_path, sub_terms, "submission")

    print("[12] done. Re-run 05_train.py — get_feature_cols() picks the new columns up automatically.")


if __name__ == "__main__":
    main()
