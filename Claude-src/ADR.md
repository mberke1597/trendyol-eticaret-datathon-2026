# ADR-01: Ensemble combiner — grid-searched blend weights vs. logistic-regression stacking

**Status:** Accepted
**Date:** 2026-07-03
**Deciders:** Trendyol Datathon team (Kaggle-stage ML core)

## Context

`../src/04_train.py` (the pre-existing, proven pipeline) combines the 3 GBDT
models' OOF predictions via a grid-searched linear blend (`search_blend_weights`,
step 0.1) and reports the resulting macro-F1 as the OOF score. A teammate's
independent `train_stacking.py` instead trained a logistic-regression
meta-model on the same 3 OOF prediction columns — a more principled combiner
in general (it can learn non-uniform, non-grid-constrained weights and an
intercept) — but evaluated it by re-predicting on the *same* OOF rows it was
fit on, which is mildly optimistic (the meta-model's own training data is not
a fair test of its generalization).

Both approaches are legitimate; the question is which to ship, and how to
score the choice honestly.

## Decision

Claude-src's `05_train.py` / `training_utils.nested_compare_ensemble_methods`
implements **both** combiners and picks between them via an honest,
non-optimistic comparison:

1. Split the OOF index set via `GroupShuffleSplit(term_id)` into a
   meta-fit portion and a meta-holdout portion (`config.META_NESTED_HOLDOUT_FRAC`,
   default 20%) — grouping by `term_id` is preserved even at this second
   level, for the same cold-start reason the outer CV uses `GroupKFold(term_id)`.
2. Fit blend-weight search on the meta-fit portion only; fit logistic
   regression stacking on the meta-fit portion only.
3. Score **both** on the held-out meta-holdout portion (neither was fit on
   it) — whichever wins is the real answer to "does stacking actually
   generalize better here."
4. Refit the winning method on 100% of the OOF data for the final production
   combiner (standard practice once the method type itself has been
   validated), but persist BOTH the honest holdout score and the full-data
   score in `meta.json` so it's unambiguous which number is which.

## Options Considered

### Option A: Keep simple blend-weight grid search only (as in `../src/`)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Risk of overfitting the combiner | Low (3 weights, coarse grid) |
| Ceiling | Lower — can't express weights outside the 0.1 grid or a learned intercept |

**Pros:** simple, already proven, no meta-model variance risk.
**Cons:** leaves points on the table if the true optimal combination isn't on
the grid.

### Option B: Always use stacking (as in the teammate's script)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium |
| Risk of overfitting the combiner | Medium (only 5-fold's worth of OOF rows to fit 3 coefficients + intercept — small but non-zero risk) |
| Ceiling | Higher in principle |

**Pros:** more expressive combiner.
**Cons:** the teammate's own reported score for this method was measured on
the same data it was fit on — we don't actually know if it generalizes
better without a fair test.

### Option C (chosen): Honest nested comparison, pick the winner
| Dimension | Assessment |
|-----------|------------|
| Complexity | Medium-high |
| Risk of overfitting the combiner | Low — the reported score is from a portion neither combiner touched |
| Ceiling | Same as Option B if stacking wins, same as Option A if it doesn't |

**Pros:** never worse than the better of A/B, and the choice is backed by
real evidence instead of assumption.
**Cons:** more code, one extra GroupShuffleSplit, slightly more complex
`meta.json` schema.

## Trade-off Analysis

The nested split costs one extra `GroupShuffleSplit` and doubles the
combiner-fitting work (train both, not just one) — negligible next to 5 folds
× 3 GBDT models at 3000 estimators each. The downside risk (stacking
overfitting its 3-4 parameters to a small holdout) is bounded because logistic
regression with `C=1.0` on 3 features is low-variance; if it doesn't
generalize, blend weights win the comparison and get used instead, so the
final choice can't be worse than the simpler option.

## Consequences

- `meta.json`'s `ensemble` key now has a `method` field (`"blend"` or
  `"stacking"`) that `07_predict.py` branches on — inference code must handle
  both, not assume a single combiner shape.
- The honest holdout score (`selection_holdout_macro_f1`) and the full-data
  score (`oof_macro_f1_all_data`) are different numbers for different reasons;
  the hackathon report should cite the holdout score as the more trustworthy
  one when arguing about ensemble quality.
- If a future teammate wants to add a 4th base model, both combiners need to
  accept a variable-length `oof` dict — already true of the current
  implementation (`names = list(oof.keys())`), so no further change needed.

## Action Items
1. [x] Implement `nested_compare_ensemble_methods` in `training_utils.py`.
2. [x] Wire `07_predict.py` to branch on `ensemble.method`.
3. [ ] After a real Kaggle run, record which method won and by how much in
       the final hackathon report (this will differ run-to-run since it
       depends on the actual OOF prediction quality — not knowable in advance
       without running 05_train.py on a GPU).

---

# ADR-02: LLM enrichment and uncertain-band re-scoring are optional, flag-gated stages

**Status:** Accepted
**Date:** 2026-07-03
**Deciders:** Trendyol Datathon team

## Context

Industry research (Amazon, Walmart, Alibaba/Taobao — see DESIGN.md) shows
LLMs are used in production search-relevance systems in two ways: offline
label/feature generation (distilled into a lightweight model for serving) and
narrow, bounded re-scoring of only the most uncertain cases. Neither pattern
puts an LLM in the high-volume serving path. The competition's own prompt.md
explicitly bans heavy cross-encoders/LLMs from the 3.36M-row inference path.

At the same time, this sandbox has no GPU and no access to
Trendyol-LLM-Asure-12B or any hosted LLM API, so neither `02_llm_enrichment.py`
nor `06_rescore_uncertain_band.py` could be run end-to-end here — only their
logic (JSON parsing, plumbing, caching, graceful-fallback behavior) could be
verified, using a deterministic `MockLLMClient`.

## Decision

Both LLM-touching stages are OFF by default (`config.USE_LLM_ENRICHMENT`,
`config.USE_UNCERTAIN_BAND_RESCORE`), and every other script
(`04_build_features.py`, `05_train.py`, `07_predict.py`) runs correctly and
completely without them — the LLM columns/re-scoring are purely additive.
`--dry-run` flags on both scripts exercise the full code path against a
`MockLLMClient` so the plumbing is testable without GPU/API access.

## Consequences

- A team member with Kaggle GPU + internet access can turn either flag on
  without touching any other file.
- If `USE_LLM_ENRICHMENT=1` but the enrichment cache is missing,
  `04_build_features.py` warns loudly and continues without the LLM columns
  rather than crashing — a partial/failed LLM run never blocks the rest of
  the pipeline.
- These two scripts are the least-verified part of this codebase (mock-only
  testing) — flag this explicitly in the final report rather than presenting
  them as equally proven as the rest of the pipeline.

---

# ADR-03: Four post-baseline improvement stages, all optional and staged, not folded into the default run

**Status:** Accepted
**Date:** 2026-07-04
**Deciders:** Trendyol Datathon team

## Context

After fixing the brand-override regression and threshold miscalibration
(see DESIGN.md "2026-07-04 real-run findings"), a real Kaggle submission
scored 0.84 macro-F1 — a genuine improvement over the previous 0.83 baseline,
but still behind other teams on the leaderboard. Four candidate improvements
were identified, each with a different effort/risk/potential-gain profile:
embedding fine-tuning, LLM enrichment/re-scoring (code already existed,
un-activated), hyperparameter search, and iterative hard-negative mining.

## Decision

Implement all four, but as **standalone, optional scripts** rather than
folding any of them into the default `01→03→04→05→07→08` run:

- `00_finetune_embeddings.py` — only takes effect if
  `01_encode_embeddings.py` finds a checkpoint at `FINETUNE_OUTPUT_DIR` (or
  `TY_FINETUNED_MODEL_DIR` is set); otherwise the base hub model is used,
  unchanged.
- `hpo_search.py` — only takes effect if `05_train.py` finds
  `models/best_hyperparams.json`; otherwise the hardcoded, already-proven
  hyperparameters are used, unchanged.
- `09_hard_negative_mining.py` — writes a *separate*
  `train_pairs_labeled_round2.parquet`, never overwrites the round-1 file;
  `04_build_features.py` only reads it if `TY_TRAIN_PAIRS_FILE` is set.
- `02_llm_enrichment.py` / `06_rescore_uncertain_band.py` — already
  flag-gated (ADR-02), unchanged here beyond a T4 dtype fix (bfloat16 has no
  hardware support on Turing GPUs; now auto-detects and falls back to
  float16) and an optional 8-bit-loading escape hatch for a 12B model that's
  genuinely tight even across 2×T4's combined 32GB.

## Rationale

Every one of these four changes touches a part of the pipeline that already
works end-to-end and produced a real 0.84 leaderboard score. Wiring any of
them in as the new default would mean the *next* real Kaggle run conflates
"did this specific change help" with "does the pipeline still work at all" —
exactly the kind of untested-assumption risk this whole rebuild has been
trying to systematically avoid (see the "Dersler" section's running theme).
Keeping each one an opt-in, separately-runnable stage means:

1. The known-good 0.84 baseline is always reachable by just not running the
   new scripts — a bad experiment can never regress the submission below a
   state that's already been verified on the real leaderboard.
2. Each stage can be A/B'd independently (fine-tuned embeddings only, HPO
   only, hard-negatives only, or combinations) by comparing its
   `submission.csv`'s real LB score against the 0.84 baseline — the ordering
   in DESIGN.md's "Running on Kaggle" section is a suggestion for a full
   push, not a requirement to run all four together.
3. None of the four have been verified against real data/GPU in this sandbox
   (no GPU here) — their logic is unit-tested (`assign_term_folds`,
   `select_hard_negatives`, LLM batching/parsing) but a real Kaggle run is
   still required before trusting any of their outputs, same caveat the
   original baseline carried before its own first real run.

## Consequences

- Running the full "try everything" sequence costs meaningfully more
  Kaggle GPU-hours/quota than the baseline (embedding fine-tuning ~tens of
  minutes, HPO ~1 hour/algorithm by default `HPO_TIMEOUT_SECONDS`, a second
  full 5-fold training round for hard-negative mining) — budget accordingly
  against the ~30h/week Kaggle GPU quota mentioned elsewhere in this repo.
- `meta.json` now has a `hyperparameter_overrides_used` field so it's always
  possible to tell, after the fact, whether a given model was trained with
  searched or hardcoded hyperparameters.
- If a future teammate adds a 5th improvement idea, the same pattern applies:
  a standalone script, a config flag/file it checks for, and a documented
  fallback to current (proven) behavior when that file/flag is absent.

## Action Items
1. [x] Implement and unit-test all four stages' pure logic.
2. [ ] Run each stage once on real Kaggle data/GPU and record whether its
       `submission.csv` beats the 0.84 baseline, individually and combined —
       not yet done (this sandbox has no GPU).
