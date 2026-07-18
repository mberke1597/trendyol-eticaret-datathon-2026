"""
Stage 22 — score the 3.36M submission pairs with the fine-tuned cross-encoder.

Loads the N_FOLDS fold checkpoints from stage 21 and averages their P(relevant)
per pair (bagged over folds -> lower variance than any single fold). Sharded +
checkpointed exactly like Claude-src's 10_llm_relevance.py: one parquet shard per
CE_SHARD_SIZE rows, a re-run skips shards already on disk, so a killed Kaggle
session resumes instead of restarting.

Output: CE_SUB_PARQUET  [id, ce_relevance_prob]  (mirrors llm_rel_score plumbing;
40_merge_features.py joins it onto the submission feature matrix).

Run:  python 22_score_crossencoder.py
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch

from config import (
    CE_MAX_LEN, CE_EVAL_BATCH_SIZE, CE_FP16, CE_COL, CE_SUB_PARQUET,
    CE_SHARD_SIZE, MODEL_DIR, N_FOLDS, CACHE_DIR,
)
from data import load_catalog, build_item_text, query_series, load_submission_pairs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_models():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    models, toks = [], []
    for f in range(N_FOLDS):
        d = MODEL_DIR / f"ce_fold{f}"
        if not d.exists():
            raise FileNotFoundError(f"{d} missing -- run 21_train_crossencoder.py --fold {f} first")
        toks.append(AutoTokenizer.from_pretrained(d))
        m = AutoModelForSequenceClassification.from_pretrained(d).to(DEVICE).eval()
        models.append(m)
    return models, toks


@torch.no_grad()
def _score_texts(q, it, models, toks):
    """Average P(relevant) across folds for one batch of (query,item) strings."""
    probs = np.zeros(len(q), dtype=np.float64)
    for m, tok in zip(models, toks):
        enc = tok(q, it, truncation=True, max_length=CE_MAX_LEN, padding=True, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.cuda.amp.autocast(enabled=CE_FP16 and DEVICE == "cuda"):
            logit = m(**enc).logits
        probs += torch.softmax(logit.float(), dim=1)[:, 1].cpu().numpy()
    return probs / len(models)


def main():
    sub = load_submission_pairs()
    items, terms = load_catalog()
    qmap = query_series(terms); imap = build_item_text(items)
    q_all = qmap.reindex(sub["term_id"].values).fillna("").tolist()
    it_all = imap.reindex(sub["item_id"].values).fillna("").tolist()
    ids = sub["id"].values
    n = len(sub)
    print(f"[22] scoring {n:,} submission pairs with {N_FOLDS}-fold CE ensemble on {DEVICE}")

    models, toks = _load_models()
    n_shards = (n + CE_SHARD_SIZE - 1) // CE_SHARD_SIZE
    for s in range(n_shards):
        shard_path = CACHE_DIR / f"_ce_sub_shard_{s}.parquet"
        if shard_path.exists():
            print(f"[22] shard {s+1}/{n_shards} exists -- skip"); continue
        lo, hi = s * CE_SHARD_SIZE, min((s + 1) * CE_SHARD_SIZE, n)
        t0 = time.time(); out = np.empty(hi - lo, dtype=np.float32)
        for b in range(lo, hi, CE_EVAL_BATCH_SIZE):
            e = min(b + CE_EVAL_BATCH_SIZE, hi)
            out[b - lo:e - lo] = _score_texts(q_all[b:e], it_all[b:e], models, toks)
        pd.DataFrame({"id": ids[lo:hi], CE_COL: out}).to_parquet(shard_path, index=False)
        dt = time.time() - t0
        print(f"[22] shard {s+1}/{n_shards} [{lo:,}-{hi:,}] {dt:.0f}s "
              f"({(hi-lo)/dt:.0f} rows/s, ETA {(n-hi)/((hi-lo)/dt)/60:.0f} min)")

    parts = [pd.read_parquet(CACHE_DIR / f"_ce_sub_shard_{s}.parquet") for s in range(n_shards)]
    full = pd.concat(parts, ignore_index=True)
    full.to_parquet(CE_SUB_PARQUET, index=False)
    print(f"[22] wrote {full.shape} (mean={full[CE_COL].mean():.3f}) -> {CE_SUB_PARQUET}")


if __name__ == "__main__":
    main()
