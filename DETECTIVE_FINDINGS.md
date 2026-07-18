# Detective Findings — Why You're Stuck at 0.894

Evidence gathered from your actual data, cached training set (`cache/train_pairs_labeled.parquet`), code, logs, and best submission (`891_validated_v8.csv`).

---

## Finding 1 (SMOKING GUN): Your training negatives are poisoned

Your negatives are *unlabeled* candidates you assumed are irrelevant. I measured how many contain **every query token** inside the item text (title+category+brand+attributes) — a strong proxy for actual relevance:

| Source | Full query-token containment | Share of negatives |
|---|---|---|
| **dense_ann "negatives"** | **45.0%** | 35% |
| lexical "negatives" | 15.6% | 25% |
| category_sibling | 9.1% | 20% |
| popularity_random | 0.1% | 20% |
| *true positives (reference)* | *74.6%* | — |

dense_ann negatives look almost as "relevant" as real positives. Roughly **1 in 5 of your negative labels is likely wrong** — you are explicitly teaching the model *"items that match the query well are irrelevant."* This caps AUC at ~0.93 no matter what you stack on top.

This single problem explains three mysteries in your own SUREC_RAPORU:
- Round-3 hard-negative mining crashed LB to 0.51 (harder mining = more false negatives).
- CV never predicted LB ("kör uçuş") — your validation labels are noisy too.
- The cross-encoder "failed" (F1_rel 0.61) — it was trained on the same poisoned labels. The CE didn't fail; the labels sabotaged it.

**Fix:** add a *negative veto* before labeling 0: drop any mined negative with full query-token containment, very high embedding sim, or (better) high score from an LLM/CE spot-check. Then retrain everything, including the cross-encoder.

## Finding 2: The test set is top-100 retrieval lists — you ignore this structure

**94.4% of test terms have exactly 100 candidate rows** (median = min = 100). This is per-query retriever output, not i.i.d. pairs. Your pipeline scores each row independently and cuts with one global threshold.

Unexploited signals: within-term score rank/percentile, score minus term-mean, term-level features (candidate list tightness, query length interactions), and cross-row consistency. Top teams almost certainly treat this as per-query ranking + calibrated cut, not global binary classification.

Related symptom: **5.5% of terms get ZERO predicted positives** in your best submission. These are real user queries whose top-100 came from a live retriever — nearly all should have at least a few relevant items. That's concentrated F1_relevant recall loss.

## Finding 3: Your strongest behavioral feature barely exists at test time

- Training positives with click history: **100%** (by construction — positives ARE the click log)
- Training negatives with click history: 27%
- **Test rows with click history: 21%**

Even with your LOO and fold-safe fixes, the model learned to lean on item/brand click features that are absent for ~79% of test items. It systematically underscores unclicked-but-relevant items.

## Finding 4: Train/test distribution mismatch → threshold chaos

Training set: 18.7% positive, 20% easy random negatives. Test: ~28–31% positive, all-hard retrieval candidates. That's why your OOF-optimal threshold (0.425) was useless and the real working threshold was ~0.20, found by burning LB submissions. Every OOF number you compute is calibrated to a distribution the test doesn't have.

## Finding 5: The model class itself is capped — 0.95 is unreachable with this architecture

GBDT over cosine sims + lexical overlap = AUC 0.928 → macro-F1 ceiling ~0.89. You've proven post-processing tops out at +0.003. Leaders at 0.95–0.977 are near-certainly running a **fine-tuned cross-encoder / LLM** as the core scorer. Your one CE attempt is not a valid negative result (see Finding 1).

## Minor findings

- 8.3% of items share an exact duplicate title → free consistency win: propagate train-positive labels and force identical predictions across duplicate items within a term.
- Zero query-text overlap between train and test (verified, even with token reordering) — no leakage exploit exists; cold-start ranking is the whole game.
- HPO (`hpo_search.py`) and round-2 mining (`09`) were never run — but these are second-order vs. the above.
- CatBoost logs: learn logloss 0.249 vs val 0.363 — moderate overfit to the noisy negatives.

---

## Verdict: the road from 0.894 → 0.95 (in order)

1. **Clean the negatives** (veto rule + optional LLM spot-labeling of ~50K borderline pairs). Cheapest, unblocks everything.
2. **Retrain the cross-encoder on clean labels** (Trendyol embed model or XLM-R-large, 3.36M-row inference is fine chunked on T4x2). This is the core-model upgrade that leaders have and you don't.
3. **Add per-term ranking features + per-term calibration** (rank, percentile, score−term_mean; guarantee min-K positives per term to kill the 5.5% zero-positive terms).
4. Stack CE score into the GBDT ensemble, recalibrate threshold on LB with 2–3 submissions.
5. Keep the validated spec-conflict filter as the final polish (+0.003, already proven).

Realistic expectation: steps 1–3 are what separates 0.89 pipelines from 0.93+; 0.95 requires step 2 to work well. Nothing else in the repo will bridge +0.056 alone.
