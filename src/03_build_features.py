"""
Build the numeric feature matrix for:
  1. the labeled train set (cache/train_pairs_labeled.parquet from stage 02)
  2. submission_pairs.csv (3.36M rows, processed in memory-safe chunks)

Both reuse a single `LexicalIndex` fitted once on the full item/term catalog so
train and inference features are computed identically (no train/serve skew).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CACHE_DIR, DATA_DIR  # noqa: E402
from features import LexicalIndex, compute_batch_features  # noqa: E402

CHUNK_SIZE = 250_000


def load_embeddings():
    return {
        "query_main": np.load(f"{CACHE_DIR}/query_emb_main.npy"),
        "query_tiny": np.load(f"{CACHE_DIR}/query_emb_tiny.npy"),
        "item_title": np.load(f"{CACHE_DIR}/item_title_emb.npy"),
        "item_attr": np.load(f"{CACHE_DIR}/item_attr_emb.npy"),
        "category_emb": np.load(f"{CACHE_DIR}/category_unique_emb.npy"),
        "item_category_idx": np.load(f"{CACHE_DIR}/item_category_idx.npy"),
    }


def build_popularity(items, training_pairs, item_idx_of_id):
    """Raw counts only -- log-transform + leave-one-out correction happens per-row
    inside compute_batch_features so training rows don't leak their own click."""
    n_items = len(items)
    item_idx = item_idx_of_id.reindex(training_pairs["item_id"].values).values
    click_counts = np.zeros(n_items, dtype=np.int64)
    vc = pd.Series(item_idx).value_counts()
    click_counts[vc.index.values.astype(int)] = vc.values

    cat = items["category"].fillna("").values
    cat_mean_log = (
        pd.Series(np.log1p(click_counts)).groupby(cat).transform("mean").values.astype(np.float32)
    )

    brand = items["brand"].fillna("").values
    brand_click_count = pd.Series(click_counts).groupby(brand).transform("sum").values.astype(np.int64)

    return {
        "item_click_count": click_counts,
        "item_cat_mean_log": cat_mean_log,
        "brand_click_count": brand_click_count,
    }


def featurize_in_chunks(term_idx, item_idx, extra_cols, lex, emb, pop, chunk_size=CHUNK_SIZE, label=None):
    n = len(term_idx)
    parts = []
    t0 = time.time()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        lbl = label[start:end] if label is not None else None
        feat = compute_batch_features(term_idx[start:end], item_idx[start:end], lex, emb, pop, label=lbl)
        for k, v in extra_cols.items():
            feat[k] = v[start:end]
        parts.append(feat)
        print(f"  ...featurized {end:,}/{n:,} ({time.time()-t0:.1f}s)")
    return pd.concat(parts, ignore_index=True)


def main():
    t_start = time.time()
    print(f"DATA_DIR={DATA_DIR} CACHE_DIR={CACHE_DIR}")

    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    training_pairs = pd.read_csv(f"{DATA_DIR}/training_pairs.csv")

    item_id_arr = np.load(f"{CACHE_DIR}/item_id.npy", allow_pickle=True).astype(str)
    term_id_arr = np.load(f"{CACHE_DIR}/term_id.npy", allow_pickle=True).astype(str)
    assert (items["item_id"].values.astype(str) == item_id_arr).all()
    assert (terms["term_id"].values.astype(str) == term_id_arr).all()

    item_idx_of_id = pd.Series(np.arange(len(item_id_arr)), index=item_id_arr)
    term_idx_of_id = pd.Series(np.arange(len(term_id_arr)), index=term_id_arr)

    print("[main] fitting LexicalIndex on full catalog...")
    t0 = time.time()
    lex = LexicalIndex(items, terms)
    print(f"[main] LexicalIndex fit in {time.time()-t0:.1f}s "
          f"(word_vocab={len(lex.word_vec.vocabulary_):,}, char_vocab={len(lex.char_vec.vocabulary_):,})")

    emb = load_embeddings()
    pop = build_popularity(items, training_pairs, item_idx_of_id)

    # ---------------- train set ----------------
    train_labeled = pd.read_parquet(f"{CACHE_DIR}/train_pairs_labeled.parquet")
    t_term_idx = term_idx_of_id.reindex(train_labeled["term_id"].values).values.astype(np.int64)
    t_item_idx = item_idx_of_id.reindex(train_labeled["item_id"].values).values.astype(np.int64)
    print(f"[train] featurizing {len(train_labeled):,} rows...")
    train_labels = train_labeled["label"].values.astype(np.int8)
    train_feat = featurize_in_chunks(
        t_term_idx, t_item_idx,
        {
            "term_id": train_labeled["term_id"].values,
            "item_id": train_labeled["item_id"].values,
            "label": train_labels,
            "neg_source": train_labeled["neg_source"].values,
        },
        lex, emb, pop, label=train_labels,
    )
    train_out = f"{CACHE_DIR}/train_features.parquet"
    train_feat.to_parquet(train_out, index=False)
    print(f"[train] wrote {train_feat.shape} -> {train_out}")

    # ---------------- submission set ----------------
    sub_pairs = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv")
    s_term_idx = term_idx_of_id.reindex(sub_pairs["term_id"].values).values.astype(np.int64)
    s_item_idx = item_idx_of_id.reindex(sub_pairs["item_id"].values).values.astype(np.int64)
    print(f"[submission] featurizing {len(sub_pairs):,} rows...")
    sub_feat = featurize_in_chunks(
        s_term_idx, s_item_idx,
        {"id": sub_pairs["id"].values},
        lex, emb, pop,
    )
    sub_out = f"{CACHE_DIR}/submission_features.parquet"
    sub_feat.to_parquet(sub_out, index=False)
    print(f"[submission] wrote {sub_feat.shape} -> {sub_out}")

    print(f"[main] total time {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
