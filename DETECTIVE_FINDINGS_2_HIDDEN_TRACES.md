# Detective Report #2 — Hidden Traces in the Raw Dataset

Forensic pass over `trendyol-e-ticaret-yarismasi-2026-kaggle/` itself. Every number below was measured on your actual files. Proxy note: where "P(pred=1)" appears, ground truth is unavailable, so the 0.894 submission's predictions are used as a relevance proxy — the *mechanisms* are independent signals the model does not currently see.

---

## TRACE 1 (the big one): the "zero query overlap" wall is fake

Train and test share zero query text — but in **embedding space they are the same distribution**:

- Mean top-1 cosine from a test query to its nearest **train** query: **0.81**
- **27.6%** of test queries have a train neighbor at cosine ≥ 0.90
- Query length distributions are identical (mean 2.63 vs 2.60 tokens)

That means train click behavior transfers to test queries through neighbors. Measured transfer strength:

**(a) Neighbor-category transfer.** Take a test query's top-10 nearest train queries and the leaf categories they clicked. Is the candidate item's category in that set?

| | P(pred=1) |
|---|---|
| category IN neighbors' clicked set | **0.529** |
| category NOT in set | **0.047** |
| …restricted to neighbors ≥0.9 | **0.664 vs 0.035** |

An 11–19x odds split from a feature that costs one matrix multiply. Your 51 features contain *nothing* like this — they are all pairwise query↔item text/embedding comparisons.

**(b) Item click-query profile.** Your pipeline used click *counts* only, never the *texts* of the queries an item was clicked for. Max token-Jaccard between the row's query and this item's train click-queries:

| | mean profile-sim |
|---|---|
| predicted relevant rows | **0.246** |
| predicted irrelevant rows | **0.051** |

P(pred=1 | sim ≥ 0.5) = **0.656** vs 0.102 when sim = 0. Covers the 21% of test rows whose item has click history — precisely where confidence matters.

**IMPLEMENTED:** `Claude-src/13_query_neighbor_features.py` adds 4 features (`nbq_top1_sim`, `nbq_item_click_sim`, `nbq_cat_weight`, `nbq_brand_weight`), leakage-safe (train side: self-term excluded from kNN and profiles). Smoke-tested end-to-end. Run after `04_build_features.py`; `05_train.py` picks them up automatically.

Caveat measured on smoke: `nbq_item_click_sim` suffers the same train/test asymmetry as click counts (LOO makes it near-zero for train positives since 92% of items have 1 click). Watch its SHAP: if the GBDT anti-learns it, keep only `nbq_cat_weight`/`nbq_brand_weight` (label-corr +0.11/+0.10 even on the tiny smoke set) or use profile-sim ≥ 0.5 as a post-hoc 0→1 candidate rule.

**Bigger version of the same trace (for the cross-encoder):** append each item's clicked train queries to its text before encoding ("doc2query" enrichment), and each query's top train neighbors to its text. This is the classic winning move in e-commerce relevance competitions and likely how the 0.96+ teams broke the cold-start wall.

## TRACE 2: 1,807 oversize candidate lists = head queries

94.4% of terms have exactly 100 candidates; the 1,807 terms with more (up to 3,680) are head queries ("halı", "cüzdan", "erkek sneaker", "kadın çanta"). These lists are deeper retriever output → likely different positive density than the top-100 lists. `term_n_candidates` (already added in stage 12) lets the model condition on this; also consider a separate threshold for >100-row terms.

## TRACE 3: structured attribute keys you never parsed

Attribute key census (300k items): beyond renk/materyal/desen (which you parse), high-coverage keys sit unused — `ürün tipi` (product type — direct query-type matching), `ortam`, `kalıp`, `siluet`, `yaka tipi`, `kol tipi`, `tema / stil`, `persona`, `koleksiyon`, `sezon`. `sim_attr` is weak (0.05 corr) because it embeds the whole attribute blob with a 128d TinyBERT. Parsing `ürün tipi` + `kalıp` + `yaka tipi` as exact-match features against query tokens is cheap and targets exactly the fine-grained distinctions category_sibling negatives are supposed to teach.

## TRACE 4: title-duplicate label propagation (small, free)

23.5% of test items' exact titles match a train-positive title vs 21.1% direct item overlap → +2.4pp coverage by propagating click profiles across identical titles. Also force identical predictions for duplicate-title items within the same term (consistency, zero risk).

## Dead ends (checked so you don't waste time)

- **Row order:** submission_pairs.csv is fully shuffled (3.36M blocks); within-term position has zero correlation with predictions. No retrieval-rank leak.
- **IDs:** TST/TERM/ITEM hashes carry no ordering signal.
- **Duplicate test queries:** zero — every test term text is unique.
- **Query-text overlap with train:** zero, even after token reordering (0.04%).
- **Metadata:** gender/age 60% unknown (your title-fallback already covers this).

## What this adds up to for 0.95+

The 0.96–0.977 teams almost certainly have: (1) clean negatives, (2) a fine-tuned cross-encoder, and (3) some form of Trace 1 — neighbor/click-behavior transfer, either as features or as text enrichment into the encoder. You now have (1) done, (2) queued in the runbook, and (3) implemented as stage 13. Updated Kaggle order:

```bash
export TY_TRAIN_PAIRS_FILE=train_pairs_labeled_clean.parquet
python Claude-src/04_build_features.py
python Claude-src/12_add_group_features.py
python Claude-src/13_query_neighbor_features.py   # NEW — Trace 1
python Claude-src/05_train.py
python Claude-src/07_predict.py --save-proba
```

Expected: stages 12+13 on clean labels are your best shot at 0.91–0.93 without GPU; the CE with click-query text enrichment is the 0.95 play.
