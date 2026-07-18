# Detective Report #3 — Autopsy of the 0.506 / 0.603 Submissions

## Verdict: the model is fine. The threshold killed them.

Your new model's probability distribution is radically different from the old one — compressed toward zero (median 0.016, 75th pct 0.048, mean 0.096). This is *expected* after cleaning: the poisoned high-similarity negatives used to push borderline scores up; without them the model is confidently low on most irrelevant rows.

| file | implied cut | pos rate | LB |
|---|---|---|---|
| submission (3) | 0.437 (old meta-style threshold) | **8.0%** | 0.506 |
| submission (4) | mixed (~0.1 + extra flips) | **13.9%** | 0.603 |
| every good submission ever | rate-matched | 23–29% | 0.846–0.901 |

At 8% positives, F1_relevant recall collapses — the score crash is purely mechanical. **Absolute threshold values are meaningless across retrains; only the predicted-positive RATE transfers.** (The runbook's Step-2 warning, now demonstrated empirically.)

## Evidence recovered from the files

- Your proba file's row order matches `submission_pairs.csv` exactly — no alignment bug.
- The hard gender/age override set was recovered from the crashed files: 67,099 rows (2.0%) forced to 0 — normal size, working correctly.
- Zero-positive terms are down to **1.5%** (was 5.5%) — stages 12/13 already fixed most of the recall holes.
- New model vs ALL old scored submissions: flat ~85.4% agreement (0.846 through 0.894 alike). The +0.007 didn't come from polishing the old solution — 490k rows genuinely changed.

## READY TO SUBMIT (in `submissions/`, built from your proba, override preserved)

| file | final pos rate | threshold used |
|---|---|---|
| **new_rate275.csv** | 27.5% | 0.03286 |
| new_rate29.csv | 29.0% | 0.02944 |
| new_rate26.csv | 26.0% | 0.03717 |
| **new_rate275_minpos3.csv** | 27.6% | 275 + 3,012 min-pos flips |

## Submission plan (4 LB shots, in order)

1. **new_rate275.csv** — the calibrated baseline. Expect ≥0.90074 if your 0.90074 used a rate near 27%; expect a jump if it didn't.
2. Whichever direction wins between **new_rate26 / new_rate29** (submit the far one from #1's implied direction — you learn the curve's shape with one shot).
3. **new_rate275_minpos3.csv** — isolates the min-pos effect (only 3,012 rows differ from #1, low risk).
4. Best of the above + the validated spec-conflict filter (60) re-applied → your new best.

For future runs: `TY_THRESHOLD=0.0329 python 07_predict.py --from-cached-proba`, but re-derive the rate-matched value after every retrain — never reuse a threshold number.

## Strategic answer: threshold sweep or CE?

Both, in that order — the sweep costs zero retraining (files above are ready) and likely banks +0.003–0.008 today. The CE (Step 4 of the runbook, now on clean labels + click-query enrichment) is the only path past ~0.92. Start the CE fold-0 run while the sweep submissions are in flight.
