# RUNBOOK — 0.894 → 0.95 Plan (implemented 2026-07-15)

Everything below is already coded and committed to this folder. What ran locally
is DONE; what needs your Kaggle GPU box is marked RUN-ON-KAGGLE.

## What already happened (DONE, no action needed)

`cache/train_pairs_labeled_clean.parquet` **already exists** — I ran the
negative veto on your real cache:

| | before | after |
|---|---|---|
| negatives | 1,090,200 | 774,959 |
| dropped (poisoned) | — | **315,241 (28.9%)** |
| rule A (token containment) | — | 235,380 |
| rule B (cosine ≥ 0.7502 = P75 of positive sims) | — | 79,861 |
| pos_rate | 18.7% | **24.4%** (test is ~28–31%) |
| dense_ann negatives surviving | 384k | 135k |

## Code changes made (all backward-compatible, env-gated)

1. `Claude-src/03_negative_sampling.py` — containment veto at mining time
   (`TY_NEG_VETO=1` default; set 0 to reproduce old behaviour). Only matters if
   you ever re-mine; the clean parquet above already covers the current cache.
2. `NewPipeline/62_clean_negatives.py` — hardened: token-set containment (not
   substring), Turkish İ/I normalization, auto sim threshold from positive-pair
   cosines. Re-run anytime with different `--sim-veto-pos-pct`.
3. `Claude-src/12_add_group_features.py` — NEW. Per-term candidate-list features
   (rank/z within term, list size, list tightness). Run after 04, before 05.
4. `Claude-src/training_utils.py` — `TY_DROP_CLICK_FEATURES=1` drops the 3
   click/popularity columns (train-only signal; 21% test coverage).
5. `Claude-src/07_predict.py` — `TY_MIN_POS_PER_TERM=K` guarantees K positives
   per term (fixes the 5.5% zero-positive terms). Hard-overridden rows never flipped.
6. `NewPipeline/data.py` — stage 20 (CE dataset) now automatically uses the
   CLEAN labels when present. The old CE result (F1_rel 0.61) is void — it was
   trained on poisoned labels.

## RUN-ON-KAGGLE — exact order

### Step 1 — retrain GBDT on clean labels + group features (~2-4h, CPU ok)
```bash
export TY_TRAIN_PAIRS_FILE=train_pairs_labeled_clean.parquet
python Claude-src/04_build_features.py
python Claude-src/12_add_group_features.py
python Claude-src/05_train.py
python Claude-src/07_predict.py --save-proba
```
Watch `meta.json`: OOF AUC should move up from 0.928. Submit → **LB test A**.

### Step 2 — threshold + min-pos sweep (minutes, uses cached proba)
```bash
TY_THRESHOLD=0.20 TY_MIN_POS_PER_TERM=3 python Claude-src/07_predict.py --from-cached-proba
```
Remember: OOF threshold is miscalibrated by design (train 24.4% pos vs test
~28-31%); trust the LB, not OOF. Target predicted-positive rate 27–29%.
Submit → **LB test B**. Also try `TY_MIN_POS_PER_TERM=0` vs `3` vs `5` — one
submission each, keep the winner.

### Step 3 — click-feature ablation (one retrain + one submission)
```bash
TY_DROP_CLICK_FEATURES=1 python Claude-src/05_train.py && python Claude-src/07_predict.py
```
If LB doesn't drop → keep the flag on permanently.

### Step 4 — cross-encoder on CLEAN labels (the 0.93+ move, GPU)
```bash
python NewPipeline/20_build_ce_dataset.py    # now auto-uses clean labels
python NewPipeline/21_train_crossencoder.py  # fold 0 ONLY first
```
Gate: if fold-0 OOF F1_rel doesn't clearly beat 0.61, stop and reassess.
If it does: all folds → `22_score_crossencoder.py` → add CE score as a feature
column to both feature parquets → retrain 05 → submit. This is where
0.92→0.95 lives; nothing else bridges that gap.

### Step 5 — final polish (already proven, keep last)
Apply the validated spec-conflict filter (NewPipeline/60) on top of the best
new submission (+0.003 historically).

## LB submission budget (suggested)
A: clean retrain · B: threshold/min-pos · C: ablation · D: CE ensemble ·
E: D + spec-filter. Five submissions, each isolating one change — same
discipline as your Grup A-F experiments.

## Expectation management
Steps 1–3: realistic +0.01–0.03 (0.90–0.92). Step 4 decides whether 0.95 is
reachable — if the clean-label CE lands around leader-level relevance modeling,
0.93+; if not, you still land top-20 territory with a defensible, explainable
pipeline (worth 40% of the hackathon score anyway).
