"""
Build the numeric feature matrix for:
  1. the labeled train set (cache/train_pairs_labeled.parquet from stage 03)
  2. submission_pairs.csv (3.36M rows, processed in memory-safe chunks)

Both reuse a single `LexicalIndex` fitted once on the full item/term catalog so
train and inference features are computed identically (no train/serve skew).
Ported unchanged in structure from ../src/03_build_features.py; the only
change here is threading through the optional LLM-enrichment cache (see
02_llm_enrichment.py) to LexicalIndex, and the popularity function keeping the
same "correct for submission, placeholder for train" design documented below
(05_train.py recomputes fold-safe popularity for train -- see its docstring).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    CACHE_DIR,
    DATA_DIR,
    LLM_REL_FILL,
    USE_LLM_ENRICHMENT,
    USE_LLM_RELEVANCE,
    llm_rel_cache_paths,
)
from features import LexicalIndex, compute_batch_features  # noqa: E402

CHUNK_SIZE = 250_000
# Set by 09_hard_negative_mining.py's round-2 flow: TY_TRAIN_PAIRS_FILE=
# train_pairs_labeled_round2.parquet re-featurizes the round-1 labels PLUS the
# newly mined hard negatives. Defaults to round-1's plain output so this script's
# normal (no env var set) behavior is completely unchanged.
TRAIN_PAIRS_FILE = os.environ.get("TY_TRAIN_PAIRS_FILE", "train_pairs_labeled.parquet")


def load_embeddings():
    return {
        "query_main": np.load(f"{CACHE_DIR}/query_emb_main.npy"),
        "query_tiny": np.load(f"{CACHE_DIR}/query_emb_tiny.npy"),
        "item_title": np.load(f"{CACHE_DIR}/item_title_emb.npy"),
        "item_attr": np.load(f"{CACHE_DIR}/item_attr_emb.npy"),
        "category_emb": np.load(f"{CACHE_DIR}/category_unique_emb.npy"),
        "item_category_idx": np.load(f"{CACHE_DIR}/item_category_idx.npy"),
    }


def load_llm_enrichment_if_present():
    """Both files are optional -- see 02_llm_enrichment.py. Returns (None, None)
    if the enrichment stage was never run, in which case features.py simply
    omits the llm_* columns (checked via LexicalIndex.has_llm_query/item)."""
    q_path = Path(f"{CACHE_DIR}/query_llm_enrichment.parquet")
    i_path = Path(f"{CACHE_DIR}/item_llm_enrichment.parquet")
    q_df = pd.read_parquet(q_path) if q_path.exists() else None
    i_df = pd.read_parquet(i_path) if i_path.exists() else None
    if USE_LLM_ENRICHMENT and (q_df is None or i_df is None):
        print("[main] WARNING: USE_LLM_ENRICHMENT=1 but enrichment cache missing -- "
              "run 02_llm_enrichment.py first. Continuing WITHOUT llm_* features.")
    return q_df, i_df


def build_popularity(items, training_pairs, item_idx_of_id):
    """Raw counts only -- log-transform + leave-one-out correction happens per-row
    inside compute_batch_features so training rows don't leak their own click.
    NOTE: this is computed from the FULL training_pairs.csv, which is correct for
    the submission set (no labels to leak) but NOT fold-safe for CV -- 05_train.py
    recomputes these 3 columns per outer fold from only that fold's training term
    set (see its docstring / Claude-src/DESIGN.md for the leakage mechanism and
    why row-level LOO alone doesn't close it)."""
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


def join_llm_relevance(df, how_key):
    """Left-join the cached LLM pairwise relevance score (10_llm_relevance.py)
    as a numeric `llm_rel_score` column -- get_feature_cols() then picks it up
    automatically as a GBDT feature, and 05_train.py adds it as a 4th ensemble
    member. Optional: if USE_LLM_RELEVANCE is off or the cache is missing, the
    column is simply not added and the pipeline behaves exactly as before.

    how_key: "train" (join on term_id+item_id) or "submission" (join on id).
    Missing pairs are neutral-filled with LLM_REL_FILL so both the feature and
    the ensemble member stay well-defined even at partial coverage."""
    if not USE_LLM_RELEVANCE:
        return df
    train_path, sub_path = llm_rel_cache_paths()
    path = train_path if how_key == "train" else sub_path
    if not Path(path).exists():
        print(f"[llm_rel] WARNING: USE_LLM_RELEVANCE=1 but {path} missing -- "
              "run 10_llm_relevance.py first. Continuing WITHOUT llm_rel_score.")
        return df
    rel = pd.read_parquet(path)
    if how_key == "train":
        rel = rel.drop_duplicates(subset=["term_id", "item_id"])
        df = df.merge(rel[["term_id", "item_id", "llm_rel_score"]], on=["term_id", "item_id"], how="left")
    else:
        rel = rel.drop_duplicates(subset=["id"])
        df = df.merge(rel[["id", "llm_rel_score"]], on="id", how="left")
    n_missing = int(df["llm_rel_score"].isna().sum())
    df["llm_rel_score"] = df["llm_rel_score"].fillna(LLM_REL_FILL).astype(np.float32)
    print(f"[llm_rel] joined llm_rel_score onto {how_key} "
          f"({len(df) - n_missing:,}/{len(df):,} covered, {n_missing:,} neutral-filled @ {LLM_REL_FILL})")
    return df


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

    q_llm, i_llm = load_llm_enrichment_if_present()

    print("[main] fitting LexicalIndex on full catalog...")
    t0 = time.time()
    lex = LexicalIndex(items, terms, llm_query_enrichment=q_llm, llm_item_enrichment=i_llm)
    print(f"[main] LexicalIndex fit in {time.time()-t0:.1f}s "
          f"(word_vocab={len(lex.word_vec.vocabulary_):,}, char_vocab={len(lex.char_vec.vocabulary_):,}, "
          f"cat_vocab={len(lex.cat_vec.vocabulary_):,})")

    emb = load_embeddings()
    pop = build_popularity(items, training_pairs, item_idx_of_id)

    # ---------------- train set ----------------
    print(f"[main] reading labeled train pairs from {TRAIN_PAIRS_FILE} "
          f"(set TY_TRAIN_PAIRS_FILE to override, e.g. for 09_hard_negative_mining.py round 2)")
    train_labeled = pd.read_parquet(f"{CACHE_DIR}/{TRAIN_PAIRS_FILE}")
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
    train_feat = join_llm_relevance(train_feat, "train")
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
    sub_feat = join_llm_relevance(sub_feat, "submission")
    sub_out = f"{CACHE_DIR}/submission_features.parquet"
    sub_feat.to_parquet(sub_out, index=False)
    print(f"[submission] wrote {sub_feat.shape} -> {sub_out}")

    print(f"[main] total time {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
