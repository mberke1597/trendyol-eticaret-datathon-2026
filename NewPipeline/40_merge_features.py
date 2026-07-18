"""
Stage 40 — merge the new text signals onto Claude-src's 51-feature matrices.

Left-joins onto train_features.parquet / submission_features.parquet (produced by
Claude-src 04_build_features.py):

  - cross-encoder prob   (CE_TRAIN_PARQUET / CE_SUB_PARQUET)   [key: term_id+item_id / id]
  - embedding features   (EMBED_* parquets)                    [key: term_id+item_id / id]
  - LLM query features   (LLM_QUERY_PARQUET, per-term)         [key: term_id]

Any stage you skipped is simply not joined (the pipeline degrades gracefully).
Missing values are neutral-filled. Output: enriched parquets that 41_train_ensemble.py
consumes. Nothing here overrides anything — every column is just another feature
column that get_feature_cols() will pick up, so the meta-learner decides its weight.

Run:  python 40_merge_features.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from config import (
    CLAUDE_CACHE_DIR, CACHE_DIR, CE_COL, CE_TRAIN_PARQUET, CE_SUB_PARQUET,
    EMBED_TRAIN_PARQUET, EMBED_SUB_PARQUET, LLM_QUERY_PARQUET,
)
from data import load_submission_pairs

LLM_NUM_COLS = ["q_has_brand", "q_n_attrs"]   # numeric LLM features safe to feed GBDT


def _opt(path):
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else None


def merge_train(df):
    ce = _opt(CE_TRAIN_PARQUET)
    if ce is not None:
        df = df.merge(ce, on=["term_id", "item_id"], how="left")
        df[CE_COL] = df[CE_COL].fillna(0.5).astype(np.float32)
        print(f"[40][train] joined {CE_COL}")
    emb = _opt(EMBED_TRAIN_PARQUET)
    if emb is not None:
        df = df.merge(emb, on=["term_id", "item_id"], how="left")
        print(f"[40][train] joined embed feats: {[c for c in emb.columns if c not in ('term_id','item_id')]}")
    llm = _opt(LLM_QUERY_PARQUET)
    if llm is not None:
        df = df.merge(llm[["term_id"] + LLM_NUM_COLS], on="term_id", how="left")
        print(f"[40][train] joined LLM query feats: {LLM_NUM_COLS}")
    return df.fillna({c: 0 for c in df.columns if df[c].dtype != object})


def merge_submission(df):
    ce = _opt(CE_SUB_PARQUET)
    if ce is not None:
        df = df.merge(ce, on="id", how="left")
        df[CE_COL] = df[CE_COL].fillna(0.5).astype(np.float32)
        print(f"[40][sub] joined {CE_COL}")
    emb = _opt(EMBED_SUB_PARQUET)
    if emb is not None:
        df = df.merge(emb, on="id", how="left")
        print(f"[40][sub] joined embed feats")
    llm = _opt(LLM_QUERY_PARQUET)
    if llm is not None:
        # submission_features has no term_id -> map via submission_pairs
        sp = load_submission_pairs()[["id", "term_id"]]
        df = df.merge(sp, on="id", how="left").merge(
            llm[["term_id"] + LLM_NUM_COLS], on="term_id", how="left").drop(columns=["term_id"])
        print(f"[40][sub] joined LLM query feats: {LLM_NUM_COLS}")
    return df.fillna({c: 0 for c in df.columns if df[c].dtype != object})


def main():
    tr = pd.read_parquet(CLAUDE_CACHE_DIR / "train_features.parquet")
    sb = pd.read_parquet(CLAUDE_CACHE_DIR / "submission_features.parquet")
    print(f"[40] base train {tr.shape}, submission {sb.shape}")
    tr = merge_train(tr)
    sb = merge_submission(sb)
    tr.to_parquet(CACHE_DIR / "train_features_plus.parquet", index=False)
    sb.to_parquet(CACHE_DIR / "submission_features_plus.parquet", index=False)
    print(f"[40] wrote train_features_plus {tr.shape}, submission_features_plus {sb.shape} -> {CACHE_DIR}")


if __name__ == "__main__":
    main()
