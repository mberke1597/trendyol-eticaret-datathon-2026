# Detective Report #4 — Why the Sweep Scored 0.698 at Every Rate

## The evidence

Three rate-calibrated cuts (26% / 27.5% / 29%) all scored ~0.698. A pure
threshold problem would show a peak; a flat-low curve means the *ranking* is
broken. Inverting the macro-F1 formula on every known (rate, LB) point:

| submission | pos rate | LB | implied precision (π=0.30) |
|---|---|---|---|
| OLD 891_v8 | 0.274 | 0.894 | **0.889** |
| OLD 846 | 0.269 | 0.846 | 0.825 |
| NEW cut@8% (top-confidence rows only) | 0.080 | 0.506 | **0.474** |
| NEW cut@13.9% | 0.139 | 0.603 | 0.601 |
| NEW rate 26–29% | 0.26–0.29 | ~0.698 | 0.58–0.61 |

The smoking gun: the NEW model's **top 8% most-confident positives are its
LEAST precise** (0.47 vs 0.60 at looser cuts). Healthy rankers are monotone the
other way. Confident-and-wrong at the head = a feature whose meaning flips
between train and test.

## The culprit: `nbq_item_click_sim` (confessed in advance)

Report #2 flagged this exact risk. Train side used term-level LOO (own click
excluded). 92% of items have exactly one click, so in training:

- true positives → profile emptied by LOO → feature ≈ **0.003**
- mined negatives (dense_ann items clicked by *other* similar queries) → keep
  profiles → feature ≈ **0.052**

The GBDT anti-learned it: *"high click-query similarity ⇒ irrelevant."* At test
time there is no LOO, so the most genuinely-relevant rows (item clicked for a
near-identical train query) get pushed DOWN, and unclicked lookalikes float to
the top. One feature, inverted semantics, poisoned head. (Same family of bug as
Finding 3 — behavioral features that exist differently in train vs test.)

Note: proba (11) is therefore NOT the 0.90074 model. The 0.90074 run evidently
didn't include this feature (or its inference differed). AUC vs the 0.894
solution was 0.92 with uniform chunk stats — no alignment bug, no corruption;
purely the anti-learned feature.

## The fix (already coded)

1. `13_query_neighbor_features.py` — `nbq_item_click_sim` now DISABLED by
   default (`TY_NBQ_CLICK_SIM=1` to re-enable for experiments). The three
   symmetric features (`nbq_top1_sim`, `nbq_cat_weight`, `nbq_brand_weight`)
   remain on — their semantics are identical in train and test.
2. `14_drop_bad_feature.py` — NEW hotfix: drops the bad column from both
   existing feature parquets so you DON'T need to re-run 04/12/13:

```bash
python Claude-src/14_drop_bad_feature.py
python Claude-src/05_train.py
python Claude-src/07_predict.py --save-proba
# rate-match again: threshold at ~27.5% final positive rate, then submit
```

## ADDENDUM (2026-07-16, after anchor forensics) — the refined mechanism

Anchor-overlap analysis initially looked healthy (top-8% proba rows were 95%
anchor-positives), yet every new-family file ran a consistent ~-0.10 LB gap vs
what that overlap implies (proxy math validates to ±0.005 on the old family:
846 predicted 0.858, actual 0.846). The resolution, proven directly on the
proba:

**Among anchor-POSITIVE rows: unclicked items median proba 0.177, clicked
items 0.052 (3.4x lower).** Clicked items are 21% of rows but only 9.3% of the
top-8%. The model penalizes click history itself — so its head is filled with
*behaviorally-unconfirmed lookalikes* (exactly where the anchor is also
weakest) while burying confirmed-relevant clicked items. Anchor agreement
can't see this bias; the truth can. Hence -0.10 at every threshold.

This is the anti-learned `nbq_item_click_sim` signature end to end. The fix is
unchanged (stage 14 hotfix → retrain). `15_preflight_check.py` now has check
[5] (click-bias ratio >= 0.60) which this proba fails at 0.29 — this exact
failure mode can never reach the leaderboard again.

## Lessons bank (add to DESIGN.md)

- Any feature built FROM training_pairs.csv (clicks, click texts, profiles) is
  train/test-asymmetric by construction; LOO fixes the leak but can create an
  inverse proxy. Always compare the feature's label-conditional means on train
  vs its distribution on submission rows before trusting it.
- The rate-matched 3-point sweep (26/27.5/29) is a cheap model-health check:
  flat-low = broken ranking, peaked = calibration only.
- Implied-precision inversion from (rate, LB) pairs is a free diagnostic —
  no submission wasted.
