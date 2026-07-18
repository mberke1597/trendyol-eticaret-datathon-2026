# Trendyol Search Relevance — Pipeline Design

Implements the solution requested in `prompt.md` against the real competition data
described in `DatasetDescription.txt` / `Trendyol_EDA_Raporu.docx`. Code lives in
`src/`, numbered in run order. Every script auto-detects Kaggle vs local paths via
`src/config.py` (`IS_KAGGLE = os.path.exists("/kaggle/input")`), so the same code
runs unmodified in a Kaggle P100/2xT4 notebook or locally.

## Pipeline

| Script | Purpose |
|---|---|
| `01_encode_embeddings.py` | Encode queries/titles/categories (768d, `TY-ecomm-embed-multilingual-base-v1.2.0`) and queries/attributes (128d, Turkish TinyBERT). Cached as float16 `.npy`. |
| `02_negative_sampling.py` | Turn the positive-only `training_pairs.csv` into a labeled train set by mining 4 negative sources per term. |
| `text_utils.py`, `item_meta.py`, `features.py` | Shared feature-engineering library (tokenization/stemming, attribute parsing + controlled vocabularies, vectorized feature computation). |
| `03_build_features.py` | Apply `features.py` to the labeled train set and to all 3.36M `submission_pairs.csv` rows. |
| `04_train.py` | GroupKFold(term_id) CV, LightGBM+XGBoost+CatBoost, OOF blend-weight + threshold search. |
| `05_predict.py` | Chunked ensemble inference + hard gender/age override → `output/submission.csv`. |

`requirements.txt` pins `transformers==4.51.3` — the embedding model's
`trust_remote_code` implementation (RoPE cache) breaks under `transformers>=5`.

## 1. Negative sampling (prompt.md §6.1)

`training_pairs.csv` is positive-only; `submission_pairs.csv` is the *hard*
top-K output of a BM25+FAISS retriever, so uniform-random negatives would be
easier than what the model sees at test time (domain shift). `02_negative_sampling.py`
mines 4 sources per unique training term, mirroring the retriever and the
biases called out in prompt.md §4:

1. **dense_ann** (35%): FAISS top-neighbors of the query embedding — mimics the FAISS leg.
2. **lexical** (25%): union of an inverted-index token-postings sample — mimics the BM25 leg.
3. **category_sibling** (20%): other items in the same leaf category as a true positive — forces
   the model to learn fine-grained attribute/gender distinctions instead of coarse category match.
4. **popularity_random** (20%, half uniform / half popularity-weighted): exposes the model to
   "popular but irrelevant" items so it can't shortcut on brand/category fame (prompt.md's
   popularity-bias concern).

Target negatives per term: `clip(n_pos * 6, 20, 150)` — scales with how many positives a term
has while staying bounded for the outlier terms with 1000+ positives (e.g. "erkek ayakkabı").
Item catalog scale (962,873 items) forced `IndexIVFPQ` (compressed, ~46 bytes/vector) instead of
a flat index once n_items > 200k, to keep this feasible on both Kaggle and a 6GB-VRAM laptop.

## 2. Validation strategy (prompt.md §6.2)

The EDA found **zero term overlap** between `training_pairs.csv` and `submission_pairs.csv`
(cold-start on terms is total). A random row split would let a term's other rows leak across
train/val and overstate CV score. `04_train.py` uses `GroupKFold(term_id)` for the outer split,
and — since even the outer training fold shouldn't influence early-stopping on the OOF fold — a
further `GroupShuffleSplit` *inside* each outer-train fold for early stopping only. The OOF
macro-F1 is therefore a leakage-free estimate of cold-start performance.

## 3. Feature engineering (prompt.md §6.3)

`features.py` computes everything with vectorized numpy/scipy — no cross-encoder, no per-pair
Python loop on the hot path (required for 3.36M inference rows):

- **Dense semantic**: cosine sim(query, title), sim(query, category), sim(query_tiny, attrs),
  plus a few cheap cross terms (max/mean/diff of title vs. category sim).
- **Lexical/token-level**: word overlap, *stemmed* word overlap (light Turkish suffix stripper
  in `text_utils.stem`), and **character 3-gram Jaccard** — the char-ngram signal is what
  survives spacing/compounding mismatches ("multi vitamin" vs "multivitamin") that no tokenizer
  fixes. All computed as row-aligned sparse-matrix elementwise multiplication
  (`CountVectorizer(binary=True)` fit once on the item catalog), not per-row set intersections.
- **Constraint matching**: `item_meta.py` regex-extracts `renk:`/`color detail:`/`materyal:`/
  `desen:` values from the messy `attributes` string (values themselves contain commas, so a
  naive `split(",")` shreds them — used a bounded-lookahead regex instead), matched against
  ~40/35/20-word Turkish color/material/pattern vocabularies. Gender/age contradiction is a
  vectorized comparison of query-intent-code vs. item-code, where the item code falls back to a
  **title-derived** gender/age word when the seller-supplied `gender`/`age_group` metadata is
  missing — and a `gender_meta_title_conflict` feature flags when metadata and title disagree,
  addressing prompt.md's "robust contradiction checker that bypasses noisy seller metadata".
- **Popularity**: item/brand click counts from `training_pairs.csv`, log1p + category-relative
  z-style feature. Computed with **leave-one-out correction** — a row's own click is subtracted
  before log-transform, because ~92% of items click exactly once, so without LOO this feature
  would be a near-perfect label proxy at train time that doesn't exist at test time.

  **Cross-fold leak fix (code review, 2026-07-03):** row-level LOO alone doesn't stop leakage
  *across* CV folds — GroupKFold only splits on `term_id`, so an item positive for term A
  (train fold) and term B (val fold) still leaked A's click into B's validation row via the
  item/brand/category aggregates (`item_cat_mean_log` had no LOO at all — fully global category
  means). ~7.8% of items are positive for 2+ terms (`Trendyol_EDA_Raporu.docx` §5.1), so this
  was a real, if moderate, source of OOF optimism. `04_train.py` now recomputes
  `item_click_log`/`item_click_cat_rel`/`brand_click_log` **per outer fold**, from only that
  fold's training term set (`build_fold_popularity`), and overwrites those 3 columns in the
  `X_tr`/`X_es`/`X_val` slices before training (`fold_X_with_fresh_popularity`) — verified with
  a synthetic-catalog unit test that a validation-fold term's click no longer inflates the
  training fold's item/category counts. The static popularity columns baked into
  `train_features.parquet` by `03_build_features.py` are effectively placeholders now (only the
  *submission*-side columns in `submission_features.parquet` are used as-is, correctly, since
  those legitimately reflect all of `training_pairs.csv` with no labels to leak).

## 4. Modeling & ensemble (prompt.md §6.4)

3-model GBDT ensemble (LightGBM + XGBoost + CatBoost), each trained per outer fold with
`scale_pos_weight`/`class_weights` for the ~1:6–1:15 imbalance our negative sampling produces,
early-stopped on the inner split. Final inference bags all fold models per algorithm, then blends
algorithms with weights grid-searched (step 0.1) on OOF macro-F1. No cross-encoder or neural
re-ranker at inference time, per the "computationally efficient... 3.3M pairs" constraint —
everything is precomputed-embedding cosine sim + sparse lexical overlap + a GBDT forward pass.

## 5. Threshold calibration (prompt.md §6.5)

Macro-F1 is sensitive to the predicted-positive rate, and the true test click density is
unknown (prompt.md says "~30%" as a prior). `best_threshold_for_macro_f1` in `04_train.py` does
an **exact O(n log n) scan** over every possible cut point (sort predictions once, cumulative
TP/FP/FN/TN, closed-form macro-F1 at each point) rather than sampling ~1000 thresholds with
`sklearn.f1_score` — the naive version would need 66 weight-combos × 1000 thresholds ×
O(n) sklearn calls ≈ untenable at 1.8M rows. Both the F1-optimal threshold *and* the
density-matching (~30%) threshold are logged; `05_predict.py` uses the F1-optimal one by default
but honors a `TY_THRESHOLD` env var so it can be re-calibrated against public leaderboard
feedback without retraining.

**Caveat to flag in the hackathon report**: the F1-optimal threshold is calibrated against our
*synthetic* negative distribution, not the real hidden-test distribution — if the real test
click density deviates a lot from our OOF's implied rate, re-check both the density-matching
threshold and the negative-sampling mix ratios in `config.py` (`NEG_SOURCE_RATIOS`) against
early public LB signal.

## Hard constraint override

Even though `gender_contradiction`/`age_contradiction` are model features, prompt.md frames them
as *absolute* ("a search for 'kadın ayakkabı' must never return a male shoe"). `05_predict.py`
force-flips any prediction to 0 when either fires, regardless of model score, so a
noisy/undertrained model can't violate the hard constraint.
