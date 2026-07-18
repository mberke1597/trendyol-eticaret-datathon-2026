"""
Stage 30 — embedding features from TY-ecomm-embed-multilingual-base-v1.2.0.

Three families of NEW features (beyond Claude-src's existing sim_title/sim_category):

  1. Query->Query kNN label transfer  (your "assign similar scores to similar
     queries" idea). Encode the 50k unique queries, find each query's K nearest
     neighbour queries (cosine), and from training_pairs (positives) transfer:
        knn_item_pos_frac : fraction of the query's K neighbour terms for which
                            THIS item is a known positive,
        knn_item_pos_cnt  : raw count of neighbour terms with this item positive,
        knn_sim_mean      : mean cosine of the query to its K neighbours (density).
     Cheap because the query space is tiny (50k), and it injects cross-query
     behavioural signal the per-pair features can't see.

  2. Multi-field cosine similarity: query vs {title, category, brand, attributes}
     -> sim_field_* and sim_max_field / sim_mean_field. Extends Claude-src's
     title/category-only cosine with brand + attribute fields.

  3. Matryoshka multi-scale: query-title cosine at 768 vs 128 dims. The gap is a
     "coarse-vs-fine agreement" signal (large gap => match is shallow/noisy).

Reuses Claude-src's cached embeddings when present (CLAUDE_CACHE_DIR) to avoid
re-encoding 966k items; otherwise encodes from scratch.

Outputs: EMBED_TRAIN_PARQUET [term_id,item_id,<feat...>], EMBED_SUB_PARQUET [id,<feat...>]
Run:  python 30_embed_features.py            (add TY_EMBED_ITEMS=0 to skip item encode)
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from config import (
    EMBED_MODEL, EMBED_BATCH_SIZE, EMBED_MATRYOSHKA, KNN_K,
    EMBED_TRAIN_PARQUET, EMBED_SUB_PARQUET, CACHE_DIR, CLAUDE_CACHE_DIR, DATA_DIR,
)
from data import load_catalog, load_train_labeled, load_submission_pairs, build_item_text

ENCODE_ITEMS = os.environ.get("TY_EMBED_ITEMS", "1") == "1"


def _encoder(dim):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, trust_remote_code=True, truncate_dim=dim)


def _encode(model, texts, desc):
    t0 = time.time()
    emb = model.encode(list(texts), batch_size=EMBED_BATCH_SIZE, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True)
    print(f"[30] encoded {desc}: {emb.shape} ({time.time()-t0:.0f}s)")
    return emb.astype(np.float32)


def _row_cos(a, b):
    """Row-wise cosine for normalized matrices a,b (same shape)."""
    return np.einsum("ij,ij->i", a, b).astype(np.float32)


def main():
    items, terms = load_catalog()
    term_ids = terms["term_id"].values
    term_pos = {t: i for i, t in enumerate(term_ids)}
    item_ids = items["item_id"].values
    item_pos = {it: i for i, it in enumerate(item_ids)}

    # ---- encode queries (always cheap) ----
    m768 = _encoder(768)
    q768 = _encode(m768, terms["query"].values, "queries@768")

    # ---- query->query kNN + label transfer ----
    print(f"[30] building query kNN (K={KNN_K}) + positive-item transfer...")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=KNN_K + 1, metric="cosine").fit(q768)
    dist, idx = nn.kneighbors(q768)
    dist, idx = dist[:, 1:], idx[:, 1:]           # drop self
    sim = 1.0 - dist                               # cosine sim to neighbours
    knn_sim_mean = sim.mean(axis=1).astype(np.float32)   # per-term density

    tp = pd.read_csv(f"{DATA_DIR}/training_pairs.csv", usecols=["term_id", "item_id"])
    term_pos_items = tp.groupby("term_id")["item_id"].apply(set).to_dict()

    def knn_transfer(pairs):
        tp_idx = np.array([term_pos.get(t, -1) for t in pairs["term_id"].values])
        pos_frac = np.zeros(len(pairs), dtype=np.float32)
        pos_cnt = np.zeros(len(pairs), dtype=np.float32)
        sim_mean = np.zeros(len(pairs), dtype=np.float32)
        it_vals = pairs["item_id"].values
        for r in range(len(pairs)):
            ti = tp_idx[r]
            if ti < 0:
                continue
            sim_mean[r] = knn_sim_mean[ti]
            neigh = idx[ti]
            it = it_vals[r]
            c = 0
            for nb in neigh:
                s = term_pos_items.get(term_ids[nb])
                if s and it in s:
                    c += 1
            pos_cnt[r] = c
            pos_frac[r] = c / KNN_K
        return pos_frac, pos_cnt, sim_mean

    # ---- item field embeddings (optional, reuse Claude-src cache if present) ----
    item_title_emb = cat_emb = item_attr_emb = brand_emb = None
    q128 = item_title128 = None
    if ENCODE_ITEMS:
        item_text = build_item_text(items)  # not used for encode; kept for parity
        titles = items["title"].fillna("").values
        cats = items["category"].fillna("").values
        brands = items["brand"].fillna("").values
        attrs = items["attributes"].fillna("").str.slice(0, 300).values
        item_title_emb = _encode(m768, titles, "item titles@768")
        cat_emb = _encode(m768, cats, "item categories@768")
        brand_emb = _encode(m768, brands, "item brands@768")
        item_attr_emb = _encode(m768, attrs, "item attributes@768")
        # Matryoshka 128 for multi-scale (queries + titles only)
        m128 = _encoder(128)
        q128 = _encode(m128, terms["query"].values, "queries@128")
        item_title128 = _encode(m128, titles, "item titles@128")

    def field_sims(pairs):
        out = {}
        t_idx = np.array([term_pos.get(t, 0) for t in pairs["term_id"].values])
        i_idx = np.array([item_pos.get(it, 0) for it in pairs["item_id"].values])
        if item_title_emb is not None:
            qv = q768[t_idx]
            s_title = _row_cos(qv, item_title_emb[i_idx])
            s_cat = _row_cos(qv, cat_emb[i_idx])
            s_brand = _row_cos(qv, brand_emb[i_idx])
            s_attr = _row_cos(qv, item_attr_emb[i_idx])
            stack = np.vstack([s_title, s_cat, s_brand, s_attr])
            out["sim_field_title"] = s_title
            out["sim_field_category"] = s_cat
            out["sim_field_brand"] = s_brand
            out["sim_field_attr"] = s_attr
            out["sim_max_field"] = stack.max(axis=0)
            out["sim_mean_field"] = stack.mean(axis=0)
            # Matryoshka multi-scale on query-title
            s_title128 = _row_cos(q128[t_idx], item_title128[i_idx])
            out["sim_title_128"] = s_title128
            out["sim_scale_gap"] = (s_title - s_title128).astype(np.float32)
        return out

    def build(pairs, key_cols):
        pf, pc, sm = knn_transfer(pairs)
        d = {c: pairs[c].values for c in key_cols}
        d.update({"knn_item_pos_frac": pf, "knn_item_pos_cnt": pc, "knn_sim_mean": sm})
        d.update(field_sims(pairs))
        return pd.DataFrame(d)

    train = load_train_labeled()
    sub = load_submission_pairs()
    print("[30] featurizing train...")
    tr = build(train, ["term_id", "item_id"])
    tr.to_parquet(EMBED_TRAIN_PARQUET, index=False)
    print(f"[30] wrote {tr.shape} -> {EMBED_TRAIN_PARQUET}")
    print("[30] featurizing submission...")
    sb = build(sub, ["id"])
    sb.to_parquet(EMBED_SUB_PARQUET, index=False)
    print(f"[30] wrote {sb.shape} -> {EMBED_SUB_PARQUET}")


if __name__ == "__main__":
    main()
