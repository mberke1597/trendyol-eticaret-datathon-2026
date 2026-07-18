"""
Generate label=0 pairs to complement the 250k positive-only training_pairs.csv.

Core 4-source mining logic ported unchanged (proven) from ../src/02_negative_sampling.py:
  1. dense_ann          - FAISS top-neighbors of the query embedding
  2. lexical             - inverted-index token overlap
  3. category_sibling     - other items in the same leaf category as a true positive
  4. popularity_random    - half uniform, half popularity-weighted random item

Negative *volume* per term (`n_neg_target`) is calibrated via
config.NEG_PER_POS_MULTIPLIER/MIN_NEG_PER_TERM/MAX_NEG_PER_TERM to track the
real test click density discovered through actual leaderboard experiments
(~28-31%), NOT an artificial 50/50 positive/negative balance -- a teammate's
train_stacking.py made exactly that mistake (see Claude-src/DESIGN.md).

NEW optional 5th source, `llm_guided` (only runs if config.USE_LLM_ENRICHMENT
and cache/query_llm_enrichment.parquet from 02_llm_enrichment.py exist): for
long-tail terms (<=2 positives, i.e. little category-sibling signal available),
use the LLM's guessed category for the query to sample "plausible but wrong
category" negatives -- a lightweight, embedding-free approximation of the
SyNeg/IKEA-style "LLM-guided hard negative" pattern from Claude-src/DESIGN.md,
without requiring an extra generation+embedding round-trip per term.
"""
import os
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
    USE_LLM_ENRICHMENT,
)
from text_utils import tokenize  # noqa: E402

RNG = np.random.default_rng(RANDOM_SEED)

ANN_SEARCH_K = 80
LEXICAL_POOL_CAP = 800

# NEGATIVE VETO (2026-07-15): never label a mined candidate 0 when every query
# token appears (as a token) in the item's title+brand+category+attributes.
# Measured on the previous run's output: 45% of dense_ann and 16% of lexical
# "negatives" had full query-token containment (true positives: 75%) -- i.e. a
# large slice of mined negatives were almost certainly RELEVANT, teaching the
# model that well-matching items are irrelevant. Vetoed picks are backfilled
# from popularity_random (safe negatives). Set TY_NEG_VETO=0 to reproduce the
# old (poisoned) behaviour for A/B comparison. Deeper cleaning (embedding-sim
# rule B) lives in NewPipeline/62_clean_negatives.py.
NEG_VETO = os.environ.get("TY_NEG_VETO", "1") == "1"
POSTING_SAMPLE_PER_TOKEN = 400
POPULARITY_POOL_SIZE = 4_000_000
LONG_TAIL_POS_THRESHOLD = 2  # terms with <= this many positives get llm_guided negatives
LLM_GUIDED_FRAC_OF_TARGET = 0.15  # fraction of n_neg_target reserved for llm_guided, long-tail only


def build_ann_index(item_emb_f32):
    n, d = item_emb_f32.shape
    if n <= 200_000:
        index = faiss.IndexFlatIP(d)
        index.add(item_emb_f32)
        return index
    nlist = min(512, max(128, int(np.sqrt(n))))
    m = 48
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


def load_llm_guided_categories(term_id_arr):
    """Returns dict term_id -> category_guess string, or {} if LLM enrichment
    wasn't run. Never raises -- this source is purely additive."""
    path = Path(f"{CACHE_DIR}/query_llm_enrichment.parquet")
    if not (USE_LLM_ENRICHMENT and path.exists()):
        return {}
    df = pd.read_parquet(path)
    df = df[df["category_guess"].notna()]
    return dict(zip(df["term_id"], df["category_guess"]))


def main():
    t_start = time.time()
    print(f"DATA_DIR={DATA_DIR} CACHE_DIR={CACHE_DIR}")

    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    train_pairs = pd.read_csv(f"{DATA_DIR}/training_pairs.csv")

    item_id_arr = np.load(f"{CACHE_DIR}/item_id.npy", allow_pickle=True).astype(str)
    term_id_arr = np.load(f"{CACHE_DIR}/term_id.npy", allow_pickle=True).astype(str)
    item_title_emb = np.load(f"{CACHE_DIR}/item_title_emb.npy")
    query_emb_main = np.load(f"{CACHE_DIR}/query_emb_main.npy")

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

    click_counts = np.zeros(n_items, dtype=np.int64)
    vc = train_pairs["item_idx"].value_counts()
    click_counts[vc.index.values] = vc.values
    pop_weight = np.log1p(click_counts).astype(np.float64) + 1.0
    pop_weight /= pop_weight.sum()
    print(f"[popularity] building weighted draw pool of {POPULARITY_POOL_SIZE:,}...")
    pop_pool = RNG.choice(n_items, size=POPULARITY_POOL_SIZE, p=pop_weight)
    pop_pool_ptr = 0

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

    item_category = items["category"].fillna("").values
    category_to_indices = {
        cat: grp.index.values for cat, grp in pd.Series(item_category).groupby(item_category)
    }

    print("[lexical] tokenizing item text...")
    item_text = (
        items["title"].fillna("") + " " + items["brand"].fillna("") + " " + items["category"].fillna("")
    ).values
    t0 = time.time()
    item_tokens = [tokenize(s) for s in item_text]
    print(f"[lexical] tokenized {n_items:,} items in {time.time()-t0:.1f}s")
    inv_index = build_inverted_index(item_tokens)
    print(f"[lexical] inverted index: {len(inv_index):,} unique tokens")

    item_tok_veto = None
    if NEG_VETO:
        # Veto token sets must ALSO include attributes ("128 gb", "renk: siyah"),
        # which the inverted index deliberately excludes -- a candidate matching
        # the query only through attributes is still likely relevant.
        print("[veto] building containment token sets (title+brand+category+attributes)...")
        t0 = time.time()
        attr_text = items["attributes"].fillna("").values
        item_tok_veto = [
            frozenset(toks).union(tokenize(a)) if a else frozenset(toks)
            for toks, a in zip(item_tokens, attr_text)
        ]
        print(f"[veto] token sets built in {time.time()-t0:.1f}s")
    del item_text, item_tokens
    gc.collect()

    llm_guided_cat = load_llm_guided_categories(term_id_arr)
    if llm_guided_cat:
        print(f"[llm_guided] {len(llm_guided_cat):,} terms have an LLM category guess available")
    else:
        print("[llm_guided] no LLM enrichment cache found -- 5th negative source disabled (fine, optional)")

    query_texts = terms.set_index("term_id")["query"].reindex(term_id_arr).values

    grouped = train_pairs.groupby("term_idx")["item_idx"].apply(lambda s: np.array(sorted(set(s))))
    unique_term_idx = grouped.index.values
    n_terms = len(unique_term_idx)
    print(f"[main] mining negatives for {n_terms:,} unique training terms...")

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
    veto_total = 0
    for row_i, term_idx in enumerate(unique_term_idx):
        pos_items = grouped.loc[term_idx]
        pos_set = set(pos_items.tolist())
        n_pos = len(pos_items)
        n_neg_target = int(np.clip(n_pos * NEG_PER_POS_MULTIPLIER, MIN_NEG_PER_TERM, MAX_NEG_PER_TERM))

        term_id_str = term_id_arr[term_idx]
        is_long_tail = n_pos <= LONG_TAIL_POS_THRESHOLD and term_id_str in llm_guided_cat
        n_llm = round(n_neg_target * LLM_GUIDED_FRAC_OF_TARGET) if is_long_tail else 0
        remaining_target = n_neg_target - n_llm

        n_ann = round(remaining_target * NEG_SOURCE_RATIOS["dense_ann"])
        n_lex = round(remaining_target * NEG_SOURCE_RATIOS["lexical"])
        n_cat = round(remaining_target * NEG_SOURCE_RATIOS["category_sibling"])
        n_pop = remaining_target - n_ann - n_lex - n_cat

        chosen = set()

        toks = tokenize(query_texts[term_idx])
        q_set = set(toks) if (NEG_VETO and toks) else None

        def not_vetoed(idx_list):
            """Drop candidates where every query token appears in the item's
            text -- those are likely RELEVANT and must not be labeled 0."""
            if q_set is None:
                return idx_list, 0
            kept = [i for i in idx_list if not q_set <= item_tok_veto[i]]
            return kept, len(idx_list) - len(kept)

        cand = [i for i in ann_neighbors[row_i] if i >= 0 and i not in pos_set and i not in chosen]
        cand, v_ann = not_vetoed(cand)
        pick = cand[:n_ann]
        add(term_idx, pick, "dense_ann")
        chosen.update(pick)
        veto_total += v_ann
        # backfill dense_ann shortfall (veto ate into the candidate list) with
        # safe popularity_random negatives at the end of this term
        n_pop_extra = max(0, n_ann - len(pick))

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
        pool, v_lex = not_vetoed(pool)
        veto_total += v_lex
        if len(pool) > n_lex:
            pool = list(RNG.choice(pool, size=n_lex, replace=False))
        add(term_idx, pool, "lexical")
        chosen.update(pool)
        n_pop_extra += max(0, n_lex - len(pool))

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
        cat_cand, v_cat = not_vetoed(cat_cand)
        veto_total += v_cat
        if len(cat_cand) > n_cat:
            cat_cand = list(RNG.choice(cat_cand, size=n_cat, replace=False))
        add(term_idx, cat_cand, "category_sibling")
        chosen.update(cat_cand)
        n_pop_extra += max(0, n_cat - len(cat_cand))

        if n_llm > 0:
            # "plausible-but-wrong category" negatives: items living in the category the
            # LLM guessed for this query, excluding any category that's actually a true
            # positive's category (that would just duplicate category_sibling).
            guess_cat = llm_guided_cat[term_id_str]
            llm_cand = [] if guess_cat in seen_cats else [
                i for i in category_to_indices.get(guess_cat, [])
                if i not in pos_set and i not in chosen
            ]
            if len(llm_cand) > n_llm:
                llm_cand = list(RNG.choice(llm_cand, size=n_llm, replace=False))
            add(term_idx, llm_cand, "llm_guided")
            chosen.update(llm_cand)
            n_pop += n_llm - len(llm_cand)  # backfill shortfall from popularity_random

        n_pop += n_pop_extra  # backfill veto/candidate shortfall with safe random negatives
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
    if NEG_VETO:
        print(f"[veto] containment veto blocked {veto_total:,} likely-relevant candidates from "
              f"becoming negatives (backfilled with popularity_random)")

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
