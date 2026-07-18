"""
Optional round-2 negative mining, run AFTER a normal 05_train.py pass.

Round-1 negatives (03_negative_sampling.py) are mined by similarity to the QUERY
(ANN-embedding neighbors, lexical token overlap, same-category siblings) -- they
approximate the retriever's candidate distribution, not the trained model's actual
confusion. This script instead scores a WIDER per-term candidate pool with each
training term's own out-of-fold model (whichever GroupKFold fold never trained on
that term_id -- recomputed deterministically via training_utils.assign_term_folds,
the same split 05_train.py itself produces), and keeps candidates the model
confidently mislabels as positive. Those are the model's real blind spots, which is
what the hard-negative-mining literature (ANCE, RocketQA, and similar dense-
retrieval work) consistently finds gives the single biggest gain over static
negatives alone -- because they're chosen based on what THIS model gets wrong,
not on a fixed heuristic (embedding/lexical distance) that doesn't know what the
model has already learned.

Caveat, stated plainly rather than hidden: training_pairs.csv is implicit-feedback
(clicks), not an exhaustive relevance judgment -- a small fraction of "hard
negatives" mined here could actually be true-relevant items the click log just
never happened to surface for that term. This is a known, accepted trade-off in
the hard-negative-mining literature (not unique to this pipeline); config.
HARD_NEG_SCORE_THRESHOLD defaults conservatively high (0.5) specifically to limit
how often a genuinely-relevant item gets mislabeled this way.

Usage (round 2, starting from round-1 models/pairs):
    python 09_hard_negative_mining.py
    # writes cache/train_pairs_labeled_round2.parquet (round-1 rows + new hard
    # negatives, deduplicated) -- does NOT touch train_pairs_labeled.parquet.

    # back up round-1 models before overwriting them:
    cp -r models models_round1   # (or MODEL_DIR equivalent on Kaggle)

    TY_TRAIN_PAIRS_FILE=train_pairs_labeled_round2.parquet python 04_build_features.py
    python 05_train.py
    python 07_predict.py --save-proba
    # compare submission.csv's LB score against the round-1 submission before
    # deciding to keep round 2 -- this is an experiment, not an assumed win.

Usage (round 3+, iterating again on top of round 2's models/pairs):
    FIXED 2026-07-05: TY_INPUT_PAIRS_FILE / TY_OUTPUT_PAIRS_FILE env vars added
    -- without them, running this script again after round 2 would silently
    read round-1's pairs as the base again (missing round 2's mined negatives
    in the exclusion/dedup set) AND overwrite train_pairs_labeled_round2.parquet,
    losing round 2's negatives entirely. Must set BOTH explicitly for round 3:
        cp -r models models_round2   # back up round-2 models first
        TY_INPUT_PAIRS_FILE=train_pairs_labeled_round2.parquet \\
        TY_OUTPUT_PAIRS_FILE=train_pairs_labeled_round3.parquet \\
        python 09_hard_negative_mining.py
        TY_TRAIN_PAIRS_FILE=train_pairs_labeled_round3.parquet python 04_build_features.py
        python 05_train.py
        python 07_predict.py --save-proba
    Note MODEL_DIR/meta.json must be round 2's (i.e. run this right after round
    2's 05_train.py, before overwriting models/ with anything else) so the
    out-of-fold scoring in this script uses round 2's models, not round 1's.
"""
import importlib
import json
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
    HARD_NEG_ANN_K,
    HARD_NEG_MAX_PER_TERM,
    HARD_NEG_SCORE_THRESHOLD,
    LLM_REL_FILL,
    MODEL_DIR,
    N_FOLDS,
    RANDOM_SEED,
)
from features import LexicalIndex, compute_batch_features  # noqa: E402
from training_utils import (  # noqa: E402
    assign_term_folds,
    build_fold_popularity,
    fold_X_with_fresh_popularity,
    load_item_catalog_for_popularity,
    select_hard_negatives,
)

# 03_negative_sampling.py / 04_build_features.py / 07_predict.py all start with a
# digit, so `import 03_negative_sampling` is invalid syntax -- same importlib
# pattern 06_rescore_uncertain_band.py already uses for 02_llm_enrichment.
neg_mod = importlib.import_module("03_negative_sampling")
feat_mod = importlib.import_module("04_build_features")
predict_mod = importlib.import_module("07_predict")

# FIXED 2026-07-05: both used to be hardcoded ("train_pairs_labeled.parquet"
# in, "train_pairs_labeled_round2.parquet" out). That's correct for round 2
# (starting from round-1 pairs), but running this script again for round 3
# with no changes would read round-1's pairs as the base AGAIN (missing round
# 2's mined negatives in the already-labeled exclusion set) and overwrite
# round 2's output file, silently losing its negatives. See module docstring's
# "Usage (round 3+)" section for the required env vars each round.
INPUT_PAIRS_FILE = os.environ.get("TY_INPUT_PAIRS_FILE", "train_pairs_labeled.parquet")
OUTPUT_PAIRS_FILE = os.environ.get("TY_OUTPUT_PAIRS_FILE", "train_pairs_labeled_round2.parquet")

RNG = np.random.default_rng(RANDOM_SEED)
LEXICAL_POOL_CAP = 800
POSTING_SAMPLE_PER_TOKEN = 400
# FIXED 2026-07-05: unlike the ANN pool (capped at HARD_NEG_ANN_K) and the
# lexical pool (capped at LEXICAL_POOL_CAP), the category-sibling pool below
# originally had NO cap at all -- it added every item sharing a positive's
# category. Trendyol's broad leaf categories can hold tens of thousands of
# items, so across ~18K training terms this produced an effectively unbounded
# candidate set and reliably OOM'd on a real Kaggle run. Capped the same way
# the lexical pool already is: sample down to this size if a category is
# bigger than it.
CATEGORY_POOL_CAP = 500


def build_candidate_pool_for_term(term_idx, pos_set, already_labeled_set, ann_neighbors_row,
                                   query_tokens, inv_index, item_category, category_to_indices,
                                   pos_items):
    """Wider candidate pool than round-1 mining: union of ANN neighbors (K=
    HARD_NEG_ANN_K, vs. round-1's 80), full lexical postings pool, and same-category
    items -- minus true positives and anything already a labeled row (round-1
    already covered those; re-mining them would just duplicate, not add signal)."""
    exclude = pos_set | already_labeled_set
    cand = set(int(i) for i in ann_neighbors_row if i >= 0 and int(i) not in exclude)

    pool = []
    for t in query_tokens:
        postings = inv_index.get(t)
        if not postings:
            continue
        if len(postings) > POSTING_SAMPLE_PER_TOKEN:
            pool.extend(RNG.choice(postings, size=POSTING_SAMPLE_PER_TOKEN, replace=False))
        else:
            pool.extend(postings)
        if len(pool) >= LEXICAL_POOL_CAP:
            break
    cand.update(int(i) for i in pool if int(i) not in exclude)

    seen_cats = set()
    for pi in pos_items:
        cat = item_category[pi]
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        sibs = category_to_indices.get(cat)
        if sibs is None:
            continue
        if len(sibs) > CATEGORY_POOL_CAP:
            sibs = RNG.choice(sibs, size=CATEGORY_POOL_CAP, replace=False)
        cand.update(int(i) for i in sibs if int(i) not in exclude)

    return cand


def main():
    t_start = time.time()
    print(f"DATA_DIR={DATA_DIR} CACHE_DIR={CACHE_DIR}")

    with open(MODEL_DIR / "meta.json") as fh:
        meta = json.load(fh)
    feature_cols = meta["feature_cols"]
    assert meta["n_folds"] == N_FOLDS, (
        f"meta.json says n_folds={meta['n_folds']} but config.N_FOLDS={N_FOLDS} -- "
        "these must match (this script reproduces the exact GroupKFold split "
        "05_train.py used, which depends on N_FOLDS)."
    )

    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    train_pairs_raw = pd.read_csv(f"{DATA_DIR}/training_pairs.csv")
    print(f"[main] reading base labeled pairs from {INPUT_PAIRS_FILE} "
          f"(set TY_INPUT_PAIRS_FILE to override for round 3+)")
    labeled = pd.read_parquet(f"{CACHE_DIR}/{INPUT_PAIRS_FILE}")
    train_feat = pd.read_parquet(f"{CACHE_DIR}/train_features.parquet")

    item_id_arr = np.load(f"{CACHE_DIR}/item_id.npy", allow_pickle=True).astype(str)
    term_id_arr = np.load(f"{CACHE_DIR}/term_id.npy", allow_pickle=True).astype(str)
    item_title_emb = np.load(f"{CACHE_DIR}/item_title_emb.npy")
    query_emb_main = np.load(f"{CACHE_DIR}/query_emb_main.npy")

    n_items = len(item_id_arr)
    item_idx_of_id = pd.Series(np.arange(n_items), index=item_id_arr)
    term_idx_of_id = pd.Series(np.arange(len(term_id_arr)), index=term_id_arr)

    train_pairs = train_pairs_raw.assign(
        item_idx=item_idx_of_id.reindex(train_pairs_raw["item_id"].values).values,
        term_idx=term_idx_of_id.reindex(train_pairs_raw["term_id"].values).values,
    )

    # ---- term_id -> fold assignment (which fold's model is OOF-safe for this term) ----
    groups = train_feat["term_id"].values
    term_fold = assign_term_folds(groups, N_FOLDS)
    print(f"[main] recomputed term->fold assignment for {len(term_fold):,} terms")

    print("[main] loading round-1 models...")
    models = predict_mod.load_models(meta)

    # ---- fold-safe popularity per fold, matching what each fold's model actually
    # trained with (see training_utils.build_fold_popularity docstring) ----
    item_pos, category, brand = load_item_catalog_for_popularity(DATA_DIR)
    fold_pop_stats = {}
    for f in range(N_FOLDS):
        other_fold_terms = np.array([t for t, ff in term_fold.items() if ff != f])
        fold_pop_stats[f] = build_fold_popularity(other_fold_terms, train_pairs_raw, item_pos, category, brand, n_items)

    print("[dense_ann] building FAISS index (wider K than round-1)...")
    import gc

    t0 = time.time()
    item_emb_f32 = np.ascontiguousarray(item_title_emb.astype(np.float32))
    ann_index = neg_mod.build_ann_index(item_emb_f32)
    del item_emb_f32
    gc.collect()
    print(f"[dense_ann] index built in {time.time()-t0:.1f}s")

    item_category = items["category"].fillna("").values
    category_to_indices = {
        cat: grp.index.values for cat, grp in pd.Series(item_category).groupby(item_category)
    }

    print("[lexical] tokenizing item text...")
    from text_utils import tokenize

    item_text = (
        items["title"].fillna("") + " " + items["brand"].fillna("") + " " + items["category"].fillna("")
    ).values
    item_tokens = [tokenize(s) for s in item_text]
    inv_index = neg_mod.build_inverted_index(item_tokens)
    del item_text, item_tokens
    gc.collect()

    print("[main] fitting LexicalIndex on full catalog (needed to featurize new candidates)...")
    lex = LexicalIndex(items, terms)
    emb = feat_mod.load_embeddings()

    query_texts = terms.set_index("term_id")["query"].reindex(term_id_arr).values
    grouped = train_pairs.groupby("term_idx")["item_idx"].apply(lambda s: np.array(sorted(set(s))))
    unique_term_idx = grouped.index.values
    n_terms = len(unique_term_idx)
    print(f"[main] building wide candidate pools for {n_terms:,} terms (ANN_K={HARD_NEG_ANN_K})...")

    q_emb_f32 = np.ascontiguousarray(query_emb_main[unique_term_idx].astype(np.float32))
    _, ann_neighbors = ann_index.search(q_emb_f32, HARD_NEG_ANN_K)

    already_labeled_by_term = labeled.groupby("term_id")["item_id"].apply(
        lambda s: set(item_idx_of_id.reindex(s.values).values.astype(int))
    )

    cand_term_idx, cand_item_idx = [], []
    t0 = time.time()
    for row_i, term_idx in enumerate(unique_term_idx):
        pos_items = grouped.loc[term_idx]
        pos_set = set(pos_items.tolist())
        term_id_str = term_id_arr[term_idx]
        already = already_labeled_by_term.get(term_id_str, set())
        toks = tokenize(query_texts[term_idx])
        cand = build_candidate_pool_for_term(
            term_idx, pos_set, already, ann_neighbors[row_i], toks, inv_index,
            item_category, category_to_indices, pos_items,
        )
        cand_term_idx.extend([term_idx] * len(cand))
        cand_item_idx.extend(cand)
        if (row_i + 1) % 5000 == 0:
            print(f"  ...{row_i+1:,}/{n_terms:,} terms in {time.time()-t0:.1f}s "
                  f"({len(cand_term_idx):,} candidates so far)")

    cand_term_idx = np.array(cand_term_idx, dtype=np.int64)
    cand_item_idx = np.array(cand_item_idx, dtype=np.int64)
    n_cand = len(cand_term_idx)
    print(f"[main] {n_cand:,} candidate (term,item) pairs to score "
          f"(pool building took {time.time()-t0:.1f}s)")

    if n_cand == 0:
        print("[main] no new candidates found beyond round-1's mining + true positives -- nothing to do")
        return

    # ---- featurize + score candidates in chunks (placeholder pop; POP_COLS get
    # overwritten per-fold below, same pattern as 05_train.py's
    # fold_X_with_fresh_popularity). FIXED 2026-07-05: this used to call
    # compute_batch_features(cand_term_idx, cand_item_idx, ...) on ALL candidates
    # at once -- the same "chunked inference but not chunked read/featurize" bug
    # found in 07_predict.py, except worse here because n_cand itself was also
    # unbounded before the CATEGORY_POOL_CAP fix above. Chunking this loop is a
    # second, independent line of defense: even with capped candidate pools,
    # n_cand across ~18K training terms can still be tens of millions of rows,
    # and materializing the full feature_cols matrix for all of them at once is
    # not necessary -- each chunk only needs its own fold-adjusted scores. ----
    cand_term_id_str = term_id_arr[cand_term_idx]
    cand_fold = np.array([term_fold[t] for t in cand_term_id_str])
    placeholder_pop = fold_pop_stats[0]

    print(f"[main] featurizing + scoring {n_cand:,} candidates in "
          f"{predict_mod.CHUNK_SIZE:,}-row chunks...")
    scores = np.empty(n_cand, dtype=np.float32)
    t0 = time.time()
    for start in range(0, n_cand, predict_mod.CHUNK_SIZE):
        end = min(start + predict_mod.CHUNK_SIZE, n_cand)
        term_idx_c = cand_term_idx[start:end]
        item_idx_c = cand_item_idx[start:end]
        fold_c = cand_fold[start:end]

        feat_df_c = compute_batch_features(term_idx_c, item_idx_c, lex, emb, placeholder_pop, label=None)
        # If the models were trained WITH the optional LLM relevance feature
        # (llm_rel_score, 2026-07-11), meta["feature_cols"] contains it but
        # compute_batch_features here does NOT produce it -- these mined
        # candidates were never scored by 10_llm_relevance.py. Neutral-fill any
        # such missing column so the feature matrix matches what the models
        # expect. Honest approximation: mined candidates get a neutral LLM
        # signal, so round-2 mining is slightly less informed when the LLM
        # stage is on (it isn't run over the wider candidate pool). If that
        # matters, score the mined pairs with 10_llm_relevance.py before this.
        for col in feature_cols:
            if col not in feat_df_c.columns:
                feat_df_c[col] = LLM_REL_FILL
        X_c = feat_df_c[feature_cols].values.astype(np.float32)
        zero_labels_c = np.zeros(len(term_idx_c), dtype=np.float32)  # no LOO subtraction -- not training rows yet

        combiner_members = (meta["ensemble"].get("blend_weights")
                            or meta["ensemble"].get("meta_coef") or {}).keys()
        chunk_scores = np.empty(len(term_idx_c), dtype=np.float32)
        for f in range(N_FOLDS):
            row_idx = np.nonzero(fold_c == f)[0]
            if len(row_idx) == 0:
                continue
            X_f = fold_X_with_fresh_popularity(X_c, feature_cols, row_idx, item_idx_c, zero_labels_c, fold_pop_stats[f])
            preds = {
                "lgb": models["lgb"][f].predict(X_f),
                "xgb": models["xgb"][f].predict_proba(X_f)[:, 1],
                "cat": models["cat"][f].predict_proba(X_f)[:, 1],
            }
            if "llm" in combiner_members:
                preds["llm"] = np.full(len(row_idx), LLM_REL_FILL, dtype=np.float32)
            chunk_scores[row_idx] = predict_mod.combine_predictions(preds, meta["ensemble"])

        scores[start:end] = chunk_scores
        print(f"  ...scored {end:,}/{n_cand:,} candidates ({time.time()-t0:.1f}s)")
        del feat_df_c, X_c, chunk_scores

    # ---- select hard negatives per term (all candidates here are already
    # guaranteed non-positive by construction, so is_positive is all-False) ----
    print(f"[main] selecting hard negatives (threshold={HARD_NEG_SCORE_THRESHOLD}, "
          f"max_per_term={HARD_NEG_MAX_PER_TERM})...")
    new_rows_term, new_rows_item = [], []
    df_cand = pd.DataFrame({"term_idx": cand_term_idx, "item_idx": cand_item_idx, "score": scores})
    for term_idx, grp in df_cand.groupby("term_idx"):
        keep_items = select_hard_negatives(
            grp["score"].values, np.zeros(len(grp), dtype=bool), grp["item_idx"].values,
            HARD_NEG_SCORE_THRESHOLD, HARD_NEG_MAX_PER_TERM,
        )
        new_rows_term.extend([term_idx] * len(keep_items))
        new_rows_item.extend(keep_items)

    print(f"[main] mined {len(new_rows_term):,} new hard-negative rows across "
          f"{df_cand['term_idx'].nunique():,} terms with >=1 candidate above threshold")

    # neg_source tagged with the output file's stem (e.g. "round2"/"round3") so
    # concatenated multi-round files still show which round mined which rows.
    round_tag = Path(OUTPUT_PAIRS_FILE).stem.replace("train_pairs_labeled_", "") or "round2"
    new_df = pd.DataFrame({
        "term_id": term_id_arr[np.array(new_rows_term, dtype=np.int64)] if new_rows_term else [],
        "item_id": item_id_arr[np.array(new_rows_item, dtype=np.int64)] if new_rows_item else [],
        "label": np.zeros(len(new_rows_term), dtype=np.int8),
        "neg_source": f"hard_negative_mined_{round_tag}",
    })

    out = pd.concat([labeled, new_df], ignore_index=True)
    out = out.drop_duplicates(subset=["term_id", "item_id"], keep="first").reset_index(drop=True)
    out_path = f"{CACHE_DIR}/{OUTPUT_PAIRS_FILE}"
    out.to_parquet(out_path, index=False)
    print(f"[main] wrote {len(out):,} rows ({out['label'].mean()*100:.1f}% positive, "
          f"+{len(new_df):,} new hard negatives) -> {out_path}")
    print(out["neg_source"].value_counts())
    print(f"[main] total time {time.time()-t_start:.1f}s")
    print(f"[main] next: TY_TRAIN_PAIRS_FILE={OUTPUT_PAIRS_FILE} python 04_build_features.py "
          "&& python 05_train.py")


if __name__ == "__main__":
    main()
