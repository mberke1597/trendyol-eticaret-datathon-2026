"""
Stage 20 — build the cross-encoder training dataset + fold assignment.

Reads the canonical labeled pairs (Claude-src stage 03), assigns GroupKFold(term_id)
folds identical to 05_train.py, and writes a small parquet with
[term_id, item_id, label, fold]. Text itself is assembled on the fly in stage 21/22
from the catalog (keeps this file tiny and avoids duplicating 1.34M long strings).

Run:  python 20_build_ce_dataset.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from config import fold_parquet, N_FOLDS
from data import load_train_labeled, assign_folds


def main():
    t0 = time.time()
    train = load_train_labeled()
    print(f"[20] loaded {len(train):,} labeled pairs "
          f"({100*train['label'].mean():.1f}% positive, {train['term_id'].nunique():,} unique terms)")
    train["fold"] = assign_folds(train)
    out = train[["term_id", "item_id", "label", "fold"]]
    out.to_parquet(fold_parquet(), index=False)
    print("[20] fold sizes:")
    print(out.groupby("fold").agg(n=("label", "size"), pos_rate=("label", "mean")))
    print(f"[20] wrote {out.shape} -> {fold_parquet()}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
