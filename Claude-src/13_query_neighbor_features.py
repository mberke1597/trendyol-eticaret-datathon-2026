"""
Stage 13 — query-neighbor label-transfer features (the "fake wall" exploit).

DETECTIVE FINDING (2026-07-15, measured on this dataset):
The zero train/test query overlap is a fake wall. Test queries have extremely
close TRAIN-query neighbors in embedding space:
  - mean top-1 cosine (test query -> nearest train query) = 0.81
  - 27.6% of test queries have a train neighbor at cosine >= 0.90
And those neighbors' CLICK BEHAVIOUR transfers (proxied against the best 0.894
submission's predictions):
  - item leaf-category IN top-10 neighbors' clicked categories:
      P(pred=1 | match) = 0.53  vs  P(pred=1 | no match) = 0.047
      (for neighbors >= 0.9: 0.66 vs 0.035)
  - max token-Jaccard between the row's query and the queries this ITEM was
    clicked for in train: mean 0.246 on predicted-relevant rows vs 0.051 on
    predicted-irrelevant; P(pred=1 | sim>=0.5) = 0.66 vs 0.10 baseline.
The existing 51 features are all pairwise query<->item text/embedding — NONE
transfers actual click behaviour from similar train queries. That transfer is
the single biggest untapped signal in the dataset.

FEATURES ADDED (to BOTH feature parquets, leakage-safe):
  nbq_top1_sim        term-level: cosine to nearest train query (self excluded
                      on the train side)
  nbq_item_click_sim  max cosine between this row's query and the train queries
                      that clicked THIS item (own term excluded on train side;
                      0 when the item has no click history)
  nbq_cat_weight      sim-weighted share of top-K train-query neighbors whose
                      positives include this item's leaf category
  nbq_brand_weight    same for the item's brand

Leakage design: for TRAIN rows the term itself is masked out of both the kNN
and the item click profile (term-level leave-one-out). Val-fold terms still see
other terms' clicks — identical to what the model sees at test time, where all
17,968 train terms are available. No label is ever read.

RUN (after 04_build_features.py, any order vs stage 12):
  python 13_query_neighbor_features.py
Rewrites both feature parquets in place (*.bak13.parquet backups).
05_train.py picks the new columns up automatically.
"""
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CACHE_DIR, DATA_DIR  # noqa: E402

K_NEIGHBORS = 10
TERM_CHUNK = 2000

# DISABLED BY DEFAULT (2026-07-16 postmortem): nbq_item_click_sim is a
# train/test semantic flip. Train side uses term-level LOO (own click removed;
# 92% of items have exactly ONE click) so TRUE POSITIVES score ~0 while mined
# negatives keep their click profiles -> the GBDT anti-learns "high click-query
# similarity = irrelevant". At test time there is no LOO, so genuinely relevant
# clicked items get pushed DOWN. Real-LB evidence: rate-matched sweeps of a
# model trained with this feature scored 0.698 (vs 0.894 old / 0.90074 clean),
# and inverted macro-F1 math showed head-of-ranking precision (top 8%) of only
# 0.43-0.52 vs 0.58-0.64 at 26% -- confident-and-wrong, the signature of a
# flipped feature. nbq_top1_sim / nbq_cat_weight / nbq_brand_weight have
# symmetric train/test semantics and remain on.
import os
WRITE_ITEM_CLICK_SIM = os.environ.get("TY_NBQ_CLICK_SIM", "0") == "1"


def load_embeddings():
    qemb = np.load(f"{CACHE_DIR}/query_emb_main.npy").astype(np.float32)
    qemb /= np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-9
    tid = np.load(f"{CACHE_DIR}/term_id.npy", allow_pickle=True).astype(str)
    return qemb, {t: k for k, t in enumerate(tid)}


def main():
    t_start = time.time()
    qemb, tpos = load_embeddings()

    tr_pairs = pd.read_csv(f"{DATA_DIR}/training_pairs.csv", usecols=["term_id", "item_id"])
    items = pd.read_csv(f"{DATA_DIR}/items.csv", usecols=["item_id", "category", "brand"])
    leaf_of = dict(zip(items.item_id, items.category.fillna("")))
    brand_of = dict(zip(items.item_id, items.brand.fillna("").astype(str).str.lower()))

    tr_terms = tr_pairs.term_id.unique()
    tr_rows = np.array([tpos[t] for t in tr_terms], dtype=np.int64)
    TRE = qemb[tr_rows]  # (n_train_terms, 768)
    col_of_train_term = {t: k for k, t in enumerate(tr_terms)}

    # per train term: clicked categories / brands; per item: clicked-query emb rows
    tr_pairs["leaf"] = tr_pairs.item_id.map(leaf_of)
    tr_pairs["brand"] = tr_pairs.item_id.map(brand_of)
    cats_by_term = tr_pairs.groupby("term_id")["leaf"].apply(set)
    brands_by_term = tr_pairs.groupby("term_id")["brand"].apply(set)
    item_profile = defaultdict(list)  # item_id -> [qemb row indices]
    for t, i in zip(tr_pairs.term_id.values, tr_pairs.item_id.values):
        item_profile[i].append(tpos[t])
    item_profile = {i: np.asarray(v, dtype=np.int64) for i, v in item_profile.items()}

    def term_neighbor_tables(term_ids, exclude_self):
        """For each unique term: top-1 sim + sim-weighted neighbor cat/brand dicts."""
        uniq = pd.unique(term_ids)
        rows = np.array([tpos[t] for t in uniq], dtype=np.int64)
        top1 = np.zeros(len(uniq), dtype=np.float32)
        cat_w, brand_w = [], []
        for s in range(0, len(uniq), TERM_CHUNK):
            e = min(s + TERM_CHUNK, len(uniq))
            S = qemb[rows[s:e]] @ TRE.T  # (chunk, n_train_terms)
            if exclude_self:
                for k in range(s, e):
                    c = col_of_train_term.get(uniq[k])
                    if c is not None:
                        S[k - s, c] = -1.0
            nn = np.argpartition(-S, K_NEIGHBORS, axis=1)[:, :K_NEIGHBORS]
            nns = np.take_along_axis(S, nn, axis=1)
            top1[s:e] = nns.max(axis=1)
            for k in range(e - s):
                cw, bw, tot = defaultdict(float), defaultdict(float), 0.0
                for j, w in zip(nn[k], nns[k]):
                    w = float(max(w, 0.0))
                    tot += w
                    tterm = tr_terms[j]
                    for c in cats_by_term.get(tterm, ()):
                        cw[c] += w
                    for b in brands_by_term.get(tterm, ()):
                        bw[b] += w
                tot = tot or 1.0
                cat_w.append({c: v / tot for c, v in cw.items()})
                brand_w.append({b: v / tot for b, v in bw.items()})
            if (s // TERM_CHUNK) % 5 == 0:
                print(f"    ...kNN {e:,}/{len(uniq):,} terms ({time.time()-t_start:.0f}s)")
        idx = {t: k for k, t in enumerate(uniq)}
        return idx, top1, cat_w, brand_w

    def compute(df_terms, df_items, exclude_self):
        idx, top1, cat_w, brand_w = term_neighbor_tables(df_terms, exclude_self)
        n = len(df_terms)
        f_top1 = np.zeros(n, np.float32)
        f_cat = np.zeros(n, np.float32)
        f_brand = np.zeros(n, np.float32)
        f_click = np.zeros(n, np.float32)
        for r in range(n):
            k = idx[df_terms[r]]
            f_top1[r] = top1[k]
            it = df_items[r]
            f_cat[r] = cat_w[k].get(leaf_of.get(it, ""), 0.0)
            f_brand[r] = brand_w[k].get(brand_of.get(it, ""), 0.0)
        out = {"nbq_top1_sim": f_top1, "nbq_cat_weight": f_cat, "nbq_brand_weight": f_brand}
        if WRITE_ITEM_CLICK_SIM:
            # see module-level warning: train/test-asymmetric, anti-learned on
            # the real LB. Only enable for explicit experiments.
            qrow = np.array([tpos[t] for t in df_terms], dtype=np.int64)
            for r in range(n):
                prof = item_profile.get(df_items[r])
                if prof is None:
                    continue
                if exclude_self:
                    prof = prof[prof != qrow[r]]
                    if len(prof) == 0:
                        continue
                f_click[r] = float(np.max(qemb[prof] @ qemb[qrow[r]]))
                if r % 500_000 == 0 and r:
                    print(f"    ...click-profile {r:,}/{n:,} ({time.time()-t_start:.0f}s)")
            out["nbq_item_click_sim"] = f_click
        return out

    def apply_to(path, term_arr, item_arr, exclude_self, label):
        feats = compute(term_arr, item_arr, exclude_self)
        df = pd.read_parquet(path)
        assert len(df) == len(term_arr), f"{label}: row mismatch"
        bak = path.with_suffix(".bak13.parquet")
        if not bak.exists():
            shutil.copy(path, bak)
        for k, v in feats.items():
            df[k] = v
        df.to_parquet(path, index=False)
        print(f"  [{label}] +{list(feats)} -> {path.name} (shape {df.shape})")

    train_path = Path(f"{CACHE_DIR}/train_features.parquet")
    tmeta = pd.read_parquet(train_path, columns=["term_id", "item_id"])
    print("[13] train side (self-excluded kNN + LOO click profile)...")
    apply_to(train_path, tmeta.term_id.values, tmeta.item_id.values, True, "train")

    sub_path = Path(f"{CACHE_DIR}/submission_features.parquet")
    sub_ids = pd.read_parquet(sub_path, columns=["id"])["id"].values
    pairs = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv", usecols=["id", "term_id", "item_id"])
    aligned = pd.Series(np.arange(len(pairs)), index=pairs["id"].values).reindex(sub_ids).values
    assert not pd.isna(aligned).any()
    aligned = aligned.astype(np.int64)
    print("[13] submission side...")
    apply_to(sub_path, pairs.term_id.values[aligned], pairs.item_id.values[aligned], False, "submission")

    print(f"[13] done in {time.time()-t_start:.0f}s. Re-run 05_train.py.")


if __name__ == "__main__":
    main()
