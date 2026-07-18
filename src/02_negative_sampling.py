"""
Generate label=0 pairs to complement the 250k positive-only training_pairs.csv.

Why not random negatives: submission_pairs.csv is built from a BM25+FAISS
retriever's top-K, i.e. every test candidate is already "plausible" (shares
vocabulary or embedding neighborhood with the query). A model trained against
uniform-random negatives will see a much easier decision boundary at train
time than at test time (domain shift) and its scores will not be calibrated
for the hard-candidate regime it is actually evaluated on.

We mine 4 negative sources per training term_id, matching the retriever the
organizers describe and the "noisy metadata" / "popularity bias" risks in
prompt.md section 4:
  1. dense_ann          - FAISS top-neighbors of the query embedding (mimics the FAISS leg)
  2. lexical             - inverted-index token overlap (mimics the BM25 leg)
  3. category_sibling     - other items in the same leaf category as a true positive
                            (forces the model to learn fine-grained attribute/gender
                            distinctions instead of coarse category match)
  4. popularity_random    - half uniform, half popularity-weighted random item
                            (exposes the model to "popular but irrelevant" items so
                            it can't shortcut on brand/category popularity)

Output: cache/train_pairs_labeled.parquet with columns
  term_id, item_id, label, neg_source
"""
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    CACHE_DIR,
    DATA_DIR,
    MAX_NEG_PER_TERM,
    MIN_NEG_PER_TERM,
    NEG_PER_POS_MULTIPLIER,
    NEG_SOURCE_RATIOS,
    RANDOM_SEED,
)
from text_utils import tokenize  # noqa: E402

RNG = np.random.default_rng(RANDOM_SEED)

ANN_SEARCH_K = 80          # over-fetch then filter out true positives
LEXICAL_POOL_CAP = 800     # cap union-of-postings pool size per query
POSTING_SAMPLE_PER_TOKEN = 400  # cap how many item ids we pull from one token's postings
POPULARITY_POOL_SIZE = 4_000_000  # pre-shuffled popularity-weighted draw pool


def build_ann_index(item_emb_f32):
    n, d = item_emb_f32.shape
    if n <= 200_000:
        index = faiss.IndexFlatIP(d)
        index.add(item_emb_f32)
        return index
    # PQ-compressed IVF for memory efficiency at catalog scale (~1M items):
    # ~46 bytes/vector vs 3072 bytes/vector for a flat float32 index. This is only
    # used for approximate negative mining (not final ranking), so we deliberately
    # keep nlist small -- a lower cluster count means much less transient memory
    # during k-means training, at the cost of some recall we don't need here.
    nlist = min(512, max(128, int(np.sqrt(n))))
    m = 48  # 768 / 48 = 16 dims per sub-quantizer, must divide d evenly
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    train_size = min(n, max(nlist * 40, 50_000))
    train_sample = item_emb_f32[RNG.choice(n, size=train_size, replace=False)]
    index.train(train_sample)
    del train_sample
    index.add(item_emb_f32)
    index.nprobe = 24
    return index


def build_inverted_index(token_lists):
    inv = defaultdict(list)
    for idx, toks in enumerate(token_lists):
        for t in set(toks):
            inv[t].append(idx)
    return inv


def main():
    t_start = time.time()
    print(f"DATA_DIR={DATA_DIR} CACHE_DIR={CACHE_DIR}")

    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    train_pairs = pd.read_csv(f"{DATA_DIR}/training_pairs.csv")

    item_id_arr = np.load(f"{CACHE_DIR}/item_id.npy", allow_pickle=True).astype(str)
    term_id_arr = np.load(f"{CACHE_DIR}/term_id.npy", allow_pickle=True).astype(str)
    item_title_emb = np.load(f"{CACHE_DIR}/item_title_emb.npy")  # float16, aligned to item_id_arr
    query_emb_main = np.load(f"{CACHE_DIR}/query_emb_main.npy")  # float16, aligned to term_id_arr

    assert (items["item_id"].values.astype(str) == item_id_arr).all(), "item order mismatch vs cache"
    assert (terms["term_id"].values.astype(str) == term_id_arr).all(), "term order mismatch vs cache"

    n_items = len(item_id_arr)
    item_idx_of_id = pd.Series(np.arange(n_items), index=item_id_arr)
    term_idx_of_id = pd.Series(np.arange(len(term_id_arr)), index=term_id_arr)

    train_pairs = train_pairs.assign(
        item_idx=item_idx_of_id.reindex(train_pairs["item_id"].values).values,
        term_idx=term_idx_of_id.reindex(train_pairs["term_id"].values).values,
    )
    assert train_pairs[["item_idx", "term_idx"]].isna().sum().sum() == 0

    # ---------------- popularity ----------------
    click_counts = np.zeros(n_items, dtype=np.int64)
    vc = train_pairs["item_idx"].value_counts()
    click_counts[vc.index.values] = vc.values
    pop_weight = np.log1p(click_counts).astype(np.float64) + 1.0
    pop_weight /= pop_weight.sum()
    print(f"[popularity] building weighted draw pool of {POPULARITY_POOL_SIZE:,}...")
    pop_pool = RNG.choice(n_items, size=POPULARITY_POOL_SIZE, p=pop_weight)
    pop_pool_ptr = 0

    # ---------------- dense ANN index (built first, and freed immediately) ----------------
    # This is the most memory-hungry step (float32 copy of the full catalog + FAISS
    # k-means training buffers), so we do it before the inverted index exists, and
    # free every embedding-related array the moment we're done with it.
    import gc

    print("[dense_ann] building FAISS index...")
    t0 = time.time()
    item_emb_f32 = np.ascontiguousarray(item_title_emb.astype(np.float32))
    del item_title_emb
    gc.collect()
    ann_index = build_ann_index(item_emb_f32)
    print(f"[dense_ann] index built in {time.time()-t0:.1f}s (n={n_items:,})")
    del item_emb_f32
    gc.collect()

    # ---------------- category sibling lookup ----------------
    category_to_indices = {
        cat: grp.index.values for cat, grp in pd.Series(items["category"].fillna("").values).groupby(
            items["category"].fillna("").values
        )
    }
    item_category = items["category"].fillna("").values

    # ---------------- lexical inverted index ----------------
    print("[lexical] tokenizing item text...")
    item_text = (
        items["title"].fillna("") + " " + items["brand"].fillna("") + " " + items["category"].fillna("")
    ).values
    t0 = time.time()
    item_tokens = [tokenize(s) for s in item_text]
    print(f"[lexical] tokenized {n_items:,} items in {time.time()-t0:.1f}s")
    inv_index = build_inverted_index(item_tokens)
    print(f"[lexical] inverted index: {len(inv_index):,} unique tokens")
    del item_text, item_tokens
    gc.collect()

    query_texts = terms.set_index("term_id")["query"].reindex(term_id_arr).values

    grouped = train_pairs.groupby("term_idx")["item_idx"].apply(lambda s: np.array(sorted(set(s))))
    unique_term_idx = grouped.index.values
    n_terms = len(unique_term_idx)
    print(f"[main] mining negatives for {n_terms:,} unique training terms...")

    # batch ANN search for all terms at once
    q_emb_f32 = np.ascontiguousarray(query_emb_main[unique_term_idx].astype(np.float32))
    _, ann_neighbors = ann_index.search(q_emb_f32, ANN_SEARCH_K)

    records_term, records_item, records_label, records_src = [], [], [], []

    def add(term_idx, item_idx_list, source):
        k = len(item_idx_list)
        if k == 0:
            return
        records_term.extend([term_idx] * k)
        records_item.extend(item_idx_list)
        records_label.extend([0] * k)
        records_src.extend([source] * k)

    t0 = time.time()
    for row_i, term_idx in enumerate(unique_term_idx):
        pos_items = grouped.loc[term_idx]
        pos_set = set(pos_items.tolist())
        n_pos = len(pos_items)
        n_neg_target = int(np.clip(n_pos * NEG_PER_POS_MULTIPLIER, MIN_NEG_PER_TERM, MAX_NEG_PER_TERM))

        n_ann = round(n_neg_target * NEG_SOURCE_RATIOS["dense_ann"])
        n_lex = round(n_neg_target * NEG_SOURCE_RATIOS["lexical"])
        n_cat = round(n_neg_target * NEG_SOURCE_RATIOS["category_sibling"])
        n_pop = n_neg_target - n_ann - n_lex - n_cat

        chosen = set()

        # 1) dense ANN
        cand = [i for i in ann_neighbors[row_i] if i >= 0 and i not in pos_set and i not in chosen]
        pick = cand[:n_ann]
        add(term_idx, pick, "dense_ann")
        chosen.update(pick)

        # 2) lexical
        toks = tokenize(query_texts[term_idx])
        pool = []
        for t in toks:
            postings = inv_index.get(t)
            if not postings:
                continue
            if len(postings) > POSTING_SAMPLE_PER_TOKEN:
                pool.extend(RNG.choice(postings, size=POSTING_SAMPLE_PER_TOKEN, replace=False))
            else:
                pool.extend(postings)
            if len(pool) >= LEXICAL_POOL_CAP:
                break
        pool = [i for i in set(pool) if i not in pos_set and i not in chosen]
        if len(pool) > n_lex:
            pool = list(RNG.choice(pool, size=n_lex, replace=False))
        add(term_idx, pool, "lexical")
        chosen.update(pool)

        # 3) category sibling
        cat_cand = []
        seen_cats = set()
        for pi in pos_items:
            cat = item_category[pi]
            if cat in seen_cats:
                continue
            seen_cats.add(cat)
            sibs = category_to_indices.get(cat)
            if sibs is None or len(sibs) <= 1:
                continue
            take = min(len(sibs), max(4, n_cat))
            sample = RNG.choice(sibs, size=take, replace=False)
            cat_cand.extend(sample.tolist())
        cat_cand = [i for i in set(cat_cand) if i not in pos_set and i not in chosen]
        if len(cat_cand) > n_cat:
            cat_cand = list(RNG.choice(cat_cand, size=n_cat, replace=False))
        add(term_idx, cat_cand, "category_sibling")
        chosen.update(cat_cand)

        # 4) popularity / random (half pool draw, half uniform)
        n_pop_weighted = n_pop // 2
        n_pop_uniform = n_pop - n_pop_weighted
        pop_pick = []
        tries = 0
        while len(pop_pick) < n_pop_weighted and tries < n_pop_weighted * 5:
            i = int(pop_pool[pop_pool_ptr % POPULARITY_POOL_SIZE])
            pop_pool_ptr += 1
            tries += 1
            if i not in pos_set and i not in chosen and i not in pop_pick:
                pop_pick.append(i)
        uni_pick = []
        tries = 0
        while len(uni_pick) < n_pop_uniform and tries < n_pop_uniform * 5:
            i = int(RNG.integers(0, n_items))
            tries += 1
            if i not in pos_set and i not in chosen and i not in pop_pick and i not in uni_pick:
                uni_pick.append(i)
        add(term_idx, pop_pick + uni_pick, "popularity_random")

        if (row_i + 1) % 5000 == 0:
            print(f"  ...{row_i+1:,}/{n_terms:,} terms in {time.time()-t0:.1f}s")

    print(f"[main] negative mining done in {time.time()-t0:.1f}s, {len(records_term):,} negatives")

    neg_df = pd.DataFrame(
        {
            "term_id": term_id_arr[np.array(records_term, dtype=np.int64)],
            "item_id": item_id_arr[np.array(records_item, dtype=np.int64)],
            "label": np.array(records_label, dtype=np.int8),
            "neg_source": records_src,
        }
    )
    pos_df = pd.DataFrame(
        {
            "term_id": train_pairs["term_id"].values,
            "item_id": train_pairs["item_id"].values,
            "label": np.ones(len(train_pairs), dtype=np.int8),
            "neg_source": "positive",
        }
    )
    out = pd.concat([pos_df, neg_df], ignore_index=True)
    out = out.drop_duplicates(subset=["term_id", "item_id"], keep="first").reset_index(drop=True)

    out_path = f"{CACHE_DIR}/train_pairs_labeled.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[main] wrote {len(out):,} rows ({out['label'].mean()*100:.1f}% positive) -> {out_path}")
    print(out["neg_source"].value_counts())
    print(f"[main] total time {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
