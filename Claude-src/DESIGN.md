# Claude-src — Trendyol Search Relevance Pipeline (rebuilt from scratch)

Independent of `../src/` (kept as reference/fallback — nothing here modifies
it). Every architectural choice below is grounded in either a real finding
from this competition (EDA numbers, leakage bugs found and fixed, real
leaderboard experiments) or a documented industry pattern (Amazon/Walmart/
Alibaba search-relevance papers) — see the "Dersler" (lessons) section for
the specific evidence behind each one.

## Pipeline

| Script | Purpose |
|---|---|
| `config.py` | Kaggle/local path auto-detection, all constants, feature flags. |
| `text_utils.py`, `item_meta.py`, `features.py` | Shared feature-engineering library. |
| `01_encode_embeddings.py` | TY-ecomm-embed (768d) + Turkish TinyBERT (128d) encoding. |
| `02_llm_enrichment.py` | **Optional**, off by default. Offline LLM query/item enrichment. |
| `10_llm_relevance.py` | **Optional**, off by default. Full-coverage LLM *pairwise relevance* scoring (vLLM, 1-token 1/0 logprob), used as both a GBDT feature and a 4th ensemble member. |
| `03_negative_sampling.py` | 4-source (+ optional 5th LLM-guided) negative mining, density-calibrated. |
| `04_build_features.py` | Applies `features.py` to the labeled train set + submission_pairs.csv. |
| `05_train.py` | GroupKFold(term_id) CV, 3-model GBDT ensemble, fold-safe popularity, honest blend-vs-stacking comparison, exact threshold scan. |
| `06_rescore_uncertain_band.py` | **Optional**, off by default. LLM re-scoring of near-threshold predictions. |
| `07_predict.py` | Chunked inference + hard gender/age/brand override → `output/submission.csv`. |
| `08_explainability.py` | SHAP feature importance + example explanations (hackathon 10% criterion). |
| `training_utils.py` | Pure-logic (numpy/pandas/sklearn only) training helpers, factored out so they're unit-testable without lightgbm/xgboost/catboost. |
| `tests/` | pytest suite — runs without any GPU/heavy-ML dependency (see "Testing" below). |

`requirements.txt` pins `transformers==4.51.3` (the embedding model's
`trust_remote_code` RoPE cache implementation breaks under `transformers>=5`).

## Dersler (grounding for every non-obvious decision)

### 1. Cold-start is total — `GroupKFold(term_id)` is not optional
EDA found **0% `term_id` overlap** between `training_pairs.csv` and
`submission_pairs.csv`. A random row split lets a term's other rows leak
across train/val and overstates CV score. Every CV split in this codebase
(outer 5-fold, inner early-stopping split, AND the nested meta-combiner
comparison in `05_train.py`) groups by `term_id`.

### 2. `submission_pairs.csv` is hard-candidate, not random
It's a BM25+FAISS retriever's top-K output (~100 candidates/term, min/median).
Negative sampling in `03_negative_sampling.py` mines 4 sources (dense_ann,
lexical, category_sibling, popularity_random) to mimic this, calibrated by
`config.NEG_PER_POS_MULTIPLIER/MIN_NEG_PER_TERM/MAX_NEG_PER_TERM` toward the
**empirically discovered real test density (~28-31%)**, found through 7 real
Kaggle leaderboard submissions this competition (OOF-calibrated threshold at
14% predicted-positive gave 0.68; the true plateau at 28-31% gave 0.83). A
teammate's independent `train_stacking.py` instead trained on an artificially
balanced 50/50 positive/negative set — a real, verified difference from what
actually works here (see ADR.md and the code review that surfaced it).

### 3. Cross-fold popularity leakage (found + fixed in `../src/`, carried forward here)
Popularity features (`item_click_log`, `item_click_cat_rel`, `brand_click_log`)
computed globally leak an item's click from term A (train fold) into term B's
validation row when that item is positive for both (~7.8% of items are). Fixed
by recomputing these 3 columns **per outer fold**, from only that fold's own
training term set (`training_utils.build_fold_popularity`), with row-level
leave-one-out on top. Verified in `tests/test_popularity_leakage.py` with a
synthetic catalog where a val-fold term's click must NOT appear in the
recomputed training-fold stats.

### 4. Category-path literal overlap was a measured gap
`items.csv`'s `category` field is a real, hierarchical, `/`-separated path
(2-6 levels deep, matching Trendyol's actual site navigation). Measured on
30K sampled positive training pairs: query tokens overlap **some** category
level 71.8% of the time, the **deepest (most specific)** level 64.7% of the
time. Only a semantic (embedding) category similarity existed before this
rebuild; `features.py` now also computes literal word-overlap against the
category text specifically (`cat_word_overlap_n`, `cat_last_level_overlap_n`,
etc.), fit as its own vocabulary separate from the title+brand+category blend.

### 5. Domain synonyms and brand contradiction — vectorized, not per-pair
A teammate's `train_stacking.py` had a genuinely good Turkish e-commerce
synonym dictionary (ayakkabı↔sneaker, mont↔kaban, kot↔denim, color synonyms
with collocation exceptions) and a fully-implemented `BrandMatcher`, but (a)
the Turkish-character string literals had a mojibake/double-encoding bug that
crashed `str.maketrans()` at import time — verified by actually running it —
and (b) both were applied with a per-(query,item)-pair Python loop, and
`BrandMatcher` was never even wired into the feature pipeline.

Here: the same ideas are re-verified as real UTF-8 (`tests/test_text_utils.py`
asserts every Turkish special character is a single codepoint), and the
synonym/bigram expansion is done **once per unique query** in
`LexicalIndex.__init__` (`text_utils.expand_query_tokens`), then run through
the existing sparse CountVectorizer machinery — so 3.36M rows never pay a
per-pair Python cost for the richer matching. `BrandMatcher` is actually
called in `compute_batch_features` (masked partial loop: only the small
fraction of rows whose query names a specific brand goes through Python).

### 6. Ensemble combiner: honest comparison, not just "use stacking"
See `ADR.md` ADR-01 for the full writeup — grid-searched blend weights and
logistic-regression stacking are both implemented and compared on a
`GroupShuffleSplit(term_id)` holdout neither was fit on, avoiding the mild
optimism of scoring a meta-model on its own training rows (a gap found in the
teammate's stacking script).

### 7. Hard gender/age override is non-negotiable — brand override was tried and reverted
prompt.md section 4 frames query gender/age constraints as **absolute**
("a search for 'kadın ayakkabı' must never return a male shoe"). The
teammate's script removed this override entirely, relying on the GBDT to have
learned it as a soft feature — a real regression against the competition's
own stated requirement. `07_predict.py` force-flips any prediction to 0 when
`gender_contradiction` OR `age_contradiction` fires, regardless of model
score, and `tests/test_hard_override.py` proves this with a synthetic example
where the model is 99% confident and the override still wins.

`brand_contradiction` was ALSO added to this override for a while, past what
the proven `../src/05_predict.py` does (that script only ever overrode on
gender/age — never brand — and scored 0.83 on the real leaderboard). This was
a speculative extension, not something prompt.md states as an absolute rule
the way it does for gender/age. On a real Kaggle run (2026-07-04) it flagged
~27% of all 3.36M submission rows and forced the predicted-positive rate down
to 12.32% against a target of ~28-31% (see "Threshold calibration reality
check" below) — almost certainly a net loss. **Reverted**: `brand_contradiction`
is still computed and used as a normal GBDT feature, but is no longer a hard
override. See `07_predict.py`'s "REVERTED 2026-07-04" docstring note and
`tests/test_hard_override.py::test_brand_contradiction_does_not_override`,
which locks the decision in. If brand is reintroduced as a hard override in
the future, A/B it against a real leaderboard submission first.

### 8. LLM usage mirrors Amazon/Walmart/Alibaba's "offline-only" pattern
See `ADR.md` ADR-02. Both LLM-touching stages are optional and off by default;
the rest of the pipeline is complete without them.

### 9b. `sentence-transformers>=5` silently drags in an incompatible `transformers` (found on a real Kaggle run, 2026-07-03)
The single-device fix in lesson 9 did NOT resolve the crash -- same assertion,
same line, this time on a plain single-GPU `model.encode()` call. Root cause
(confirmed via the sentence-transformers maintainer, GitHub issue #3717):
`Alibaba-NLP/new-impl` (the remote code TY-ecomm-embed's config points
`trust_remote_code=True` at) has not been updated for `transformers>=5` and
crashes with exactly this CUDA index-out-of-bounds signature; the fix is
`transformers<5`. This repo's own `requirements.txt` had a self-contradiction:
it pinned `transformers==4.51.3` AND `sentence-transformers>=5,<6` -- but
sentence-transformers 5.x requires transformers>=5 internally, so pip
resolves that conflict by upgrading transformers past the pin (silently, no
error). Fixed by pinning `sentence-transformers==3.4.1` instead (last major
line built against transformers 4.x). **If you already installed the old
requirements.txt in a running Kaggle session, `pip install` alone will not
fix it** -- transformers is already imported in that Python process; you
must reinstall with the corrected versions AND restart the kernel/session
before loading any model.

**Recurred 2026-07-04 on a brand-new Kaggle session**: the exact same crash
came back, this time on `00_finetune_embeddings.py`'s very first model load
(before any fine-tuning even started) and then again on `01_encode_embeddings.py`.
Proof it's the same root cause, not a new one: the traceback's own file paths
(`sentence_transformers/sentence_transformer/evaluation/information_retrieval.py`,
`sentence_transformers/base/model.py`, `sentence_transformers/base/modules/transformer.py`)
are the sentence-transformers **5.x** package layout (5.0 restructured the
flat 3.x package into `sentence_transformer/`/`cross_encoder/`/`base/`
submodules) — meaning this fresh session was running on Kaggle's *default*
preinstalled transformers/sentence-transformers (>=5), not the pinned
versions in `requirements.txt`. A brand-new Kaggle session resets installed
packages exactly like it resets `/kaggle/working` (see the numpy/scipy
finding below) — `pip install -r requirements.txt` from a previous session
does not carry over. **Checklist for every new session, not just the first
one**: `pip install -r Claude-src/requirements.txt` -> restart kernel/session
-> only then run any script that loads `MAIN_MODEL`. As a second line of
defense (in case the upstream `Alibaba-NLP/new-impl` repo's unpinned remote
code changes again in a way version-pinning alone doesn't catch), both
`00_finetune_embeddings.py` and `01_encode_embeddings.py` now also pass
`model_kwargs={"attn_implementation": "eager", "unpad_inputs": False,
"use_memory_efficient_attention": False}` when constructing the
`SentenceTransformer`, forcing the same safe/standard code path regardless
of what the installed transformers version would otherwise pick.

### 9. `encode_multi_process` is incompatible with TY-ecomm-embed's remote code (found on a real Kaggle T4 x2 run, 2026-07-03)
`01_encode_embeddings.py` originally used `sentence-transformers`'
multi-GPU `encode_multi_process` path whenever 2+ GPUs were visible (ported
from `../src/`). On a real Kaggle T4 x2 run this crashed with a reproducible
`torch.AcceleratorError: CUDA error: device-side assert triggered` inside the
model's own `trust_remote_code` forward pass (`modeling.py` line ~400,
`token_type_ids = position_ids.mul(0)`) — the model's cached RoPE buffer gets
built against one CUDA device, and the worker process for the *other* GPU
ends up indexing into it, producing an out-of-bounds device-side assert.
`sentence-transformers` also marks `encode_multi_process` as deprecated in
recent versions. Fixed by always encoding on a single CUDA device
(`model.encode(..., device=DEVICE)`) regardless of GPU count — this stage
isn't the pipeline's bottleneck, so not using the 2nd T4 here is an
acceptable trade-off versus a hard crash. `05_train.py` still uses both GPUs
via XGBoost/CatBoost's own `device="cuda"` multi-GPU support at the training
stage, where it actually matters.

## Threshold calibration reality check

OOF-based threshold calibration was **not reliable on its own** this
competition — the OOF-optimal threshold (~14% predicted-positive) scored 0.68
on the real public leaderboard, while the true optimum (found through 7 real
submissions) sits at ~28-31% predicted-positive (0.83, a plateau). `05_train.py`
logs both the F1-optimal and density-matching thresholds, and `07_predict.py`
honors a `TY_THRESHOLD` env var so the threshold can be re-calibrated against
real leaderboard feedback without retraining — treat the OOF number as a
starting point, not a final answer.

## Running on Kaggle

Same pattern as `../KAGGLE_SETUP.md` — upload this folder as a private
Kaggle Dataset, use GPU T4 x2 (not P100 — sm_60 is incompatible with current
PyTorch), Internet On.

**Baseline run** (what's been validated end-to-end on a real Kaggle LB
submission, see "2026-07-04 real-run findings" below):

```bash
!python Claude-src/01_encode_embeddings.py
!python Claude-src/03_negative_sampling.py
!python Claude-src/04_build_features.py
!python Claude-src/05_train.py
!python Claude-src/07_predict.py --save-proba
!python Claude-src/08_explainability.py
```

**Optional quality-upgrade stages** (added 2026-07-04, none change the
baseline's correctness if skipped — see each stage's own module docstring for
the full rationale/cost). Recommended order if trying all four for a hackathon
push, cheapest/safest first:

```bash
# 1. Embedding fine-tuning -- run BEFORE 01_encode_embeddings.py, changes what
#    that stage encodes with. Biggest potential upside (sim_title/sim_mean_title_cat
#    are consistently the top-ranked GBDT features -- see feature_importance.csv),
#    but the least proven change here; check the printed base-vs-fine-tuned IR
#    scores before trusting the checkpoint.
!python Claude-src/00_finetune_embeddings.py
!python Claude-src/01_encode_embeddings.py   # auto-detects and prefers the fine-tuned checkpoint

# 2. LLM enrichment -- optional query/item cleanup, needs GPU + internet for a
#    12B model. Set TY_LLM_LOAD_IN_8BIT=1 if you hit an OOM (a 12B model in fp16
#    needs ~24GB, tight even across 2xT4's 32GB combined).
!TY_USE_LLM_ENRICHMENT=1 python Claude-src/02_llm_enrichment.py --force

!python Claude-src/03_negative_sampling.py
!python Claude-src/04_build_features.py

# 3. Hyperparameter search -- standalone, run after features exist. Needs
#    `pip install optuna` (not in requirements.txt by default, opt-in).
!pip install -q optuna
!python Claude-src/hpo_search.py    # writes models/best_hyperparams.json

!python Claude-src/05_train.py      # auto-loads best_hyperparams.json if present

# 4. Iterative hard-negative mining -- needs round-1 models to already exist
#    (the 05_train.py call right above). Back up round-1 models first if you
#    want to compare against them later.
!cp -r /kaggle/working/models /kaggle/working/models_round1
!python Claude-src/09_hard_negative_mining.py
!TY_TRAIN_PAIRS_FILE=train_pairs_labeled_round2.parquet python Claude-src/04_build_features.py
!python Claude-src/05_train.py      # round-2 model, trained with the mined hard negatives

# optional uncertain-band re-scoring, same GPU/8-bit caveats as stage 2:
!TY_USE_UNCERTAIN_BAND_RESCORE=1 python Claude-src/06_rescore_uncertain_band.py --force

!python Claude-src/07_predict.py --save-proba
!python Claude-src/08_explainability.py
```

Each of these is a real experiment, not an assumed win — compare the
resulting `submission.csv`'s real leaderboard score against the baseline
before keeping a given stage's output. `07_predict.py --from-cached-proba`
(re-threshold a saved `submission_proba.npy` in seconds instead of re-running
the ~30-minute inference pass) makes threshold-only comparisons especially
cheap to run several of before spending a leaderboard submission on it.

## 2026-07-04 real-run findings (grounding for the recommendations above)

A real Kaggle run of the baseline pipeline surfaced three concrete, fixed
issues plus one still-open one, in order of how much they mattered:

1. **Brand hard-override was a net loss, reverted.** `07_predict.py` briefly
   force-zeroed predictions on `brand_contradiction` in addition to
   gender/age — flagged ~27% of all 3.36M submission rows and collapsed the
   predicted-positive rate to ~12-13%. The proven `../src/05_predict.py`
   never overrode on brand at all. Reverted; `brand_contradiction` remains a
   normal GBDT feature (SHAP confirms the model already learned a sensible
   negative weight for it on its own — see `feature_importance.csv` /
   `example_explanations.json`).
2. **OOF-based threshold badly underestimates the real submission's
   predicted-positive rate.** OOF said ~19% positive at the F1-optimal
   threshold; the actual `submission_pairs.csv` predictions at that same
   threshold came out at ~13%. Re-thresholding directly against
   `submission_proba.npy` (not the OOF-derived density threshold) to hit the
   empirically-known ~28-31% target found a real threshold near 0.17 —
   nowhere close to the OOF-suggested 0.45-0.65 range. Always sanity-check
   the ACTUAL submission-set positive rate before trusting an OOF number.
3. **`numpy`/`scipy` should not be pinned/reinstalled on Kaggle.**
   `requirements.txt` no longer lists them — Kaggle's preinstalled versions
   are already mutually compatible; letting pip "resolve" them again risked
   an ABI-inconsistent pair (`ImportError: cannot import name '_center' from
   numpy._core.umath`, a real failure seen on this competition's Kaggle image).
4. **Open**: even after fixes 1-2, the real leaderboard score (0.84) trails
   other teams — this is the motivation for the four optional stages above.
   None of them have a real-leaderboard-verified result yet; treat each as a
   hypothesis to test, not an assumed improvement.
5. **`07_predict.py`'s "chunked inference" only chunked the `.predict()` calls,
   not the read** (found 2026-07-05: a real Kaggle kernel silently restarted,
   RAM OOM, partway through this stage). `pd.read_parquet(submission_features.parquet)`
   loaded all 3.36M rows x every feature column in one shot before the chunk
   loop even began. Fixed by streaming the parquet file itself via
   `pyarrow.parquet.ParquetFile.iter_batches(columns=feature_cols, batch_size=CHUNK_SIZE)`,
   so peak memory for the feature matrix is one chunk's worth. The 3 small
   always-needed columns (`id`, `gender_contradiction`, `age_contradiction`)
   are still read for all rows up front — cheap on their own.
6. **`09_hard_negative_mining.py` had an unbounded candidate pool AND the same
   "not actually chunked" bug as #5** (found 2026-07-05, same symptom: OOM on
   every real run). Two separate issues, both fixed: (a) the category-sibling
   candidate source had no size cap at all — unlike the ANN pool
   (`HARD_NEG_ANN_K`) and lexical pool (`LEXICAL_POOL_CAP`), it added every
   item sharing a positive's category, and Trendyol's broad leaf categories
   can hold tens of thousands of items, so across ~18K training terms this
   was an effectively unbounded candidate set — capped it at the new
   `CATEGORY_POOL_CAP=500` (sampled down if larger), matching the discipline
   `03_negative_sampling.py`'s round-1 mining already uses. (b) even with that
   cap, `compute_batch_features` was called on the entire candidate array at
   once; featurizing + scoring is now chunked in `predict_mod.CHUNK_SIZE`-row
   batches, same pattern as fix #5.
7. **Round-3 hard-negative mining (mining again on top of round 2) was a real,
   significant regression** (found 2026-07-05): round 1 -> round 2 improved the
   real leaderboard 0.84 -> 0.88 despite round 2's OOF macro-F1 dropping
   (0.8001 -> 0.7066) -- that gap was already flagged as expected (harder
   negatives make OOF validation harder without necessarily hurting real
   generalization). Round 3 continued the same pattern in OOF (0.7066 -> 0.6852)
   but this time the real leaderboard score collapsed to 0.51, far below even
   the original 0.84 baseline. Most likely explanation: this script's own
   docstring caveat ("training_pairs.csv is implicit-feedback, not an
   exhaustive relevance judgment -- some 'hard negatives' could actually be
   true-relevant items") compounds across rounds -- round 3 mines candidates
   the round-2 model is confident about, but round-2's model was ALREADY
   trained partly on round-2's own mined (occasionally mislabeled) negatives,
   so round 3 has a real risk of confidently mislabeling more true positives
   as negatives than round 2 did. **Action: do not iterate hard-negative
   mining past round 2 without a real leaderboard check at each step** -- one
   round clearly helped here, a second round clearly hurt. Rolled back to the
   round-2 models (backed up as `models_round2`) as the current best (0.88).
8. **The 28-31% predicted-positive "plateau" does NOT hold for the round-2
   (0.891) model** (found 2026-07-06). That plateau was measured on the OLDER
   0.83 model (see "Threshold calibration reality check" above). On the
   current model, three real submissions from the SAME cached proba
   (`submission_proba.npy`) gave a clean monotonic result:
   `TY_THRESHOLD=0.17` (29.6% positive) -> 0.882, `0.19` (28.2% positive) ->
   0.889, `0.20` (27.5% positive, the shipped baseline) -> 0.891. Lowering the
   threshold to chase the old plateau center made things WORSE here, not
   better -- the optimum shifted with the model. **Action: do not assume a
   threshold finding from one model transfers to the next; re-sweep after any
   retrain.** Next real test to run: thresholds ABOVE 0.20 (e.g. 0.21, 0.22,
   0.23), since the monotonic trend below 0.20 suggests the true optimum for
   this model may sit at or above the current value, not below it.
9. **Turkish morphology/typo-tolerance integrated as an optional feature
   stage** (2026-07-07): `turkish_morphology.py` (real root/suffix extraction
   via `zeyrek`) and `typo_tolerance.py` (diacritic + typo correction) were
   built and unit-tested standalone in `New-Pipeline/` first, then wired into
   `features.py` as `root_overlap_n`/`root_recall`, gated behind
   `config.USE_TURKISH_MORPHOLOGY` (off by default -- see that file's
   docstring in `LexicalIndex.__init__`). A real integration bug was found
   and fixed before it ever reached a real run: the typo-correction
   vocabulary must be built from ITEM text only, never query text --
   including query text let a query's own typo get counted as an "already
   known real word" (it's literally in the corpus the vocabulary was just
   built from), so the correction step silently never fired for exactly the
   typos it exists to fix. Caught via a toy-catalog integration test
   (`tests/test_features.py::test_turkish_morphology_root_overlap_recovers_typo_match`)
   before trusting the feature at all -- same "verify before trusting"
   discipline as everywhere else in this file. **Validated against a real
   Kaggle submission on 2026-07-07: it HURT the real leaderboard score --
   see lesson 11 below. `USE_TURKISH_MORPHOLOGY` stays off by default; do
   not turn it on without a real, separate A/B submission next time.**
10. **zeyrek's "APPENDING RESULT" debug spam root-caused and fixed**
    (2026-07-07): a real Kaggle run of `04_build_features.py` with
    `TY_USE_TURKISH_MORPHOLOGY=1` on the actual 345k unique catalog tokens
    produced a log of 60,000+ lines that was NOTHING but this noise (user had
    to upload the raw log to get it read at all). It's `logger.warning(...)`
    inside `zeyrek/rulebasedanalyzer.py`, not a `print()` -- that's exactly
    why the earlier `contextlib.redirect_stdout`/fd-1 `os.dup2` attempts
    (see lesson 9 and `New-Pipeline/DESIGN.md`) never touched it: Python's
    `logging` module writes through its own handler machinery, independent
    of whatever `sys.stdout`/fd 1 currently points at. Fixed with one line,
    `logging.getLogger("zeyrek").setLevel(logging.ERROR)`, set once at
    `turkish_morphology.py` import time -- verified silent in a standalone
    test (`zeyrek.MorphAnalyzer()._parse(...)` with the level set produces
    zero log lines). This does NOT change anything about the unresolved
    throughput question (item 9 above) -- 345k tokens still have to be
    analyzed one by one, this only stops the log from being unusable while
    that happens.
11. **Turkish morphology feature (`root_overlap_n`/`root_recall`) validated
    on a REAL Kaggle submission (2026-07-07) -- RESULT: NEGATIVE, feature
    stays OFF.** Full pipeline rerun with `TY_USE_TURKISH_MORPHOLOGY=1` AND
    the correct `TY_TRAIN_PAIRS_FILE=train_pairs_labeled_round2.parquet`
    (confirmed with the user this time, specifically to rule out the earlier
    0.846 silent-round-1-fallback mistake as the cause) scored **0.879** vs.
    the 0.891 baseline -- a real **-0.012 regression**, not an OOF artifact.
    The underlying proba array differs meaningfully from baseline
    (corr=0.90, 5.5% of rows flip sides of the 0.20 threshold -- see chat
    history for the exact comparison), so this is a genuine model-behavior
    change, not a no-op retrain. Working theory (untested): morphological
    root-overlap is coarser than word/stem overlap -- stripping a query token
    down to its bare root can make unrelated items look lexically related
    (e.g. two products sharing a common, generic root after suffix removal
    but meaning different things), adding noise rather than recall for this
    catalog's actual query patterns. **Decision: leave
    `config.USE_TURKISH_MORPHOLOGY` at its default (off, already the case --
    no rollback needed) and do not pursue this feature further** without a
    more targeted signal (e.g. restricting root-overlap to only fire when
    word/stem overlap is exactly 0, instead of always adding it as an extra
    column) -- not worth the zeyrek throughput/log-noise cost otherwise.
12. **HPO (Optuna) + threshold sweep, IN PROGRESS (2026-07-08).** First real
    HPO run (`hpo_search.py`, clean baseline -- morphology off, size feature
    on): threshold=0.22 (25.85% predicted-positive) scored **0.884**, BELOW
    the 0.891 baseline (which was threshold=0.20, no HPO). This continues the
    lesson-8 monotonic pattern one step further: 0.17->29.6%->0.882,
    0.19->28.2%->0.889, 0.20->27.5%->0.891 (peak so far), and now
    0.22->25.85%->0.884 -- moving further from the ~27% region hurts in
    EITHER direction, not just downward. **Confounded, not yet resolved:**
    this compares a different THRESHOLD on a different (HPO-tuned) MODEL
    against the original baseline -- can't yet tell whether HPO itself
    helped/hurt independent of threshold choice. The critical next data
    point is threshold=0.20 on this SAME HPO model's cached proba (already
    generated as `submission_thr020.csv`/`submission (25).csv`, no Kaggle
    re-run needed) -- that's the apples-to-apples comparison against 0.891.
    If it's still below 0.891, HPO itself is the regression (plausible: 
    `hpo_search.py`'s own docstring flags it uses a single
    `GroupShuffleSplit`, not full CV, to keep the search fast -- a config
    that wins that one split isn't guaranteed to generalize as well as the
    proven hardcoded values). If it's at/above 0.891, HPO helped and 0.22
    was simply the wrong threshold for this particular model.

    **RESOLVED (2026-07-08): threshold=0.20 on the HPO model scored 0.882**
    -- same threshold as the 0.891 baseline, so HPO itself (not the
    threshold) explains most of the drop, IF that HPO run's features were
    otherwise identical to the original 0.891 run. **Caveat, not fully
    clean:** this HPO run also included the size/beden feature
    (`size_match`/`size_conflict`), added to `features.py` AFTER the
    original 0.891 run and never independently validated on its own (its
    first test hit the `TY_TRAIN_PAIRS_FILE` round-1-fallback bug and scored
    0.846 -- never cleanly re-tested afterward). So 0.882 has TWO bundled
    changes vs. the 0.891 baseline (size feature + HPO), not one.
    **Next isolation step:** retrain on the SAME cached features (round-2
    pairs, size feature included) but WITHOUT `best_hyperparams.json`
    (move/delete it so `05_train.py` falls back to the proven hardcoded
    values) -- reproduces ~0.891 => HPO alone is the regression and size
    feature is cleared as harmless; still below 0.891 => size feature needs
    its own isolated test too. **Working decision either way: do not use
    `best_hyperparams.json` for the next real submission** -- HPO has not
    produced a single result at or above baseline yet (0.882 and 0.884 vs.
    0.891, at two different thresholds).
13. **LLM integration redesigned around Trendyol-LLM-Asure-12B's actual
    capabilities (2026-07-08).** Two things happened together:

    (a) **Real bug fixed, never previously caught because 02_llm_enrichment.py
    had only ever been `--dry-run` tested (MockLLMClient) before now.**
    `LLMClient` loaded the model via `AutoModelForCausalLM` + `AutoTokenizer`
    with raw string prompts. Trendyol-LLM-Asure-12B is a Gemma3-based
    MULTIMODAL checkpoint -- its own HF model card's documented usage loads
    it via `AutoModelForImageTextToText` + `AutoProcessor`, applying the chat
    template (`content: [{"type": "text", ...}]` message structure) rather
    than tokenizing a bare string. Switched to the card's own classes/pattern
    in `LLMClient.__init__`/`generate_batch()`. `transformers==4.51.3`
    (already pinned) is new enough for Gemma3 + `AutoModelForImageTextToText`,
    so no new pin was needed. Confirmed via `grep` that this competition's
    `items.csv`/`terms.csv` have no image field at all -- the model's vision
    half is simply never exercised here, text-only chat-template messages are
    sufficient. **Still not run on a real GPU yet -- run `--dry-run` first
    (still MockLLMClient, just re-verifies plumbing), then a REAL model on a
    tiny batch, before trusting this at any scale.**

    (b) **Added a "blend" re-scoring mode** (`config.RESCORE_MODE`,
    `TY_RESCORE_MODE=blend`, default stays `"override"` = original behavior)
    to `06_rescore_uncertain_band.py`. Original behavior hard-replaces the
    GBDT's uncertain-band predictions with the LLM's binary relevant/not
    call. New `rescore_band_blend()` instead converts the LLM's
    relevant+confidence into a probability and WEIGHTED-AVERAGES it with the
    GBDT's own `model_proba` (`config.RESCORE_BLEND_LLM_WEIGHT`, default
    0.5), re-thresholding the blend -- this is the literal "combine two ways
    of computing relevancy" the user asked for, architected the same way
    `05_train.py`'s own lgb/xgb/cat ensemble already blends multiple models
    instead of hard-voting. Covered by
    `tests/test_llm_batching.py::test_rescore_band_blend_*` (fallback-to-
    model-proba-on-parse-failure, and correct confidence->proba conversion),
    all via MockLLMClient -- **not yet validated on real data or a real
    Kaggle submission.**

    **What was scoped OUT, and why (grounded in this project's own prior
    throughput findings, not guessed):** a full-corpus LLM-based relevance
    classifier (LLM judges all 3.36M submission pairs directly, replacing or
    standing alongside the GBDT) was considered and rejected for now. Even
    after the `max_new_tokens` fix, `06_rescore_uncertain_band.py`'s LLM pass
    over just the narrow uncertain band was never benchmarked above roughly
    single-digit rows/sec on a much SMALLER model than this 12B one -- at
    that rate 3.36M rows is weeks, not hours, of GPU time. The existing
    two-tier design (bounded structured extraction over unique
    queries/capped items in `02_llm_enrichment.py`; bounded re-scoring over
    just the uncertain band in `06_rescore_uncertain_band.py`) is the
    version of "combine LLM + GBDT" that's actually compute-feasible on
    Kaggle's quota. Of the HF card's six highlighted task types (summarisation
    & paraphrasing, textual+visual QA, structured extraction, controlled
    generation, text classification, e-commerce relevancy): structured
    extraction and text classification map directly to (a)'s query/item
    enrichment; e-commerce relevancy maps directly to (b)'s re-scoring;
    visual QA doesn't apply (no images in this dataset); summarisation/
    paraphrasing and controlled generation have no identified use in this
    pipeline yet -- not pursued without a concrete feature idea behind them,
    same "don't build speculative capability" discipline as everywhere else
    in this file.

### 14. LLM pairwise relevance as a GBDT feature AND a 4th ensemble member (2026-07-11)
`10_llm_relevance.py` is a NEW, heavier LLM use than `02_llm_enrichment.py`.
Instead of offline structured extraction, it asks an LLM the relevance question
directly -- "is this item relevant to this query?" -- for EVERY (query, item)
pair, both the labeled train set and all 3.36M `submission_pairs`, and turns the
answer into a continuous 0..1 `llm_rel_score`. The user asked to "combine llm
relevance score and classic ensemble model at train.py", so the score is used
BOTH ways: (a) joined into `train_features`/`submission_features` as a column
that `get_feature_cols` picks up automatically, so lgb/xgb/cat train on it, and
(b) added as a 4th member ("llm") of `05_train.py`'s nested blend/stacking
combiner alongside lgb/xgb/cat. Both default OFF (`USE_LLM_RELEVANCE`,
`LLM_REL_AS_ENSEMBLE_MEMBER`).

Why this is feasible at 3.36M scale when lesson 13 scoped a full-corpus LLM pass
OUT as "weeks, not hours" -- three changes stack:
  1. **vLLM, not HF `generate()`.** Continuous batching + paged KV-cache + real
     tensor-parallelism across both T4s (`tensor_parallel_size=2`), NOT HF's
     `device_map="auto"` per-layer pipeline split that re-crosses PCIe every
     token (the exact bottleneck 02's docstring + the earlier chat diagnosed).
  2. **One generated token per pair, not ~100.** We prompt for a single "1"/"0"
     verdict and read the *logprobs* of the "1" and "0" tokens, returning a
     calibrated `P(relevant) = softmax over {1,0}` -- a continuous score (better
     for ensembling than a hard 0/1) at the cost of a single decode step, so
     prefill dominates and short (truncated) item text keeps prefill cheap. The
     1/0 form (vs. Turkish "evet"/"hayır") makes the logprob read tokenizer-
     agnostic -- digits are reliably their own single token.
  3. **A small model by default** (`Qwen/Qwen2.5-3B-Instruct`, ~6GB fp16).

**No leakage from using it as an ensemble member.** The LLM score is a fixed
EXTERNAL predictor with no access to labels and no fold dependence -- its "OOF"
prediction is just its own score on every row, identical at train and inference.
So unlike a GBDT it never had an unfair view of any train-fold row; it drops
into the combiner cleanly. `07_predict.py` mirrors this by feeding the
submission `llm_rel_score` as the "llm" member at inference (it's already a
streamed feature column, so no extra read). The combiner (`combine_ensemble_
predictions`, moved to `training_utils.py` so it's unit-testable without
lightgbm/xgboost/catboost) is member-generic: it combines exactly the keys
`05_train.py` wrote into `blend_weights`/`meta_coef`, so 3-model and 4-member
runs share one code path. `search_blend_weights` drops to a 0.2 grid step once
there are 4+ members so the combo count stays ~1.3k instead of ~14.6k.

Robustness for the multi-hour run: scoring is sharded and checkpointed
(`LLM_REL_SHARD_SIZE` rows/shard -> `cache/_llm_rel_ckpt/*.npy`), so a killed
Kaggle session resumes instead of restarting. Interaction with
`09_hard_negative_mining.py`: mined candidates have no LLM score, so 09 neutral-
fills `llm_rel_score`/the "llm" member (`LLM_REL_FILL`) for them -- an honest
approximation (round-2 mining is slightly less informed when the LLM stage is
on); score the mined pairs with `10_llm_relevance.py` first if that matters.

**NOT yet validated on a real Kaggle submission** -- same discipline as every
other optional stage here. The 1/0-logprob scoring, prompt build, shard/resume,
and 3-vs-4-member combiner logic are unit-tested (`tests/test_llm_relevance.py`,
GPU-free via MockRelevanceScorer + pure combiner), but whether the LLM score
actually *helps* macro-F1 -- and whether the 3B or the e-commerce-tuned Asure-12B
is better -- must be A/B'd on the real leaderboard. Treat it as a hypothesis.

**Run recipe (Qwen-3B first, then A/B Asure-12B on the same cached pairs):**
```bash
# 1) score relevance with the small fast model (full train + 3.36M submission)
!TY_USE_LLM_RELEVANCE=1 TY_LLM_REL_MODEL=Qwen/Qwen2.5-3B-Instruct \
  TY_LLM_REL_TAG=qwen3b python Claude-src/10_llm_relevance.py
# 2) build features (joins llm_rel_score) + train with the 4th ensemble member
!TY_USE_LLM_RELEVANCE=1 TY_LLM_REL_TAG=qwen3b python Claude-src/04_build_features.py
!TY_USE_LLM_RELEVANCE=1 TY_LLM_REL_ENSEMBLE=1 python Claude-src/05_train.py
!TY_USE_LLM_RELEVANCE=1 TY_LLM_REL_ENSEMBLE=1 python Claude-src/07_predict.py --save-proba
# 3) A/B the e-commerce-tuned model on the SAME pairs (distinct tag = both caches coexist)
!TY_USE_LLM_RELEVANCE=1 TY_LLM_REL_MODEL=Trendyol/Trendyol-LLM-Asure-12B \
  TY_LLM_REL_TAG=asure12b python Claude-src/10_llm_relevance.py
!TY_USE_LLM_RELEVANCE=1 TY_LLM_REL_TAG=asure12b python Claude-src/04_build_features.py
!TY_USE_LLM_RELEVANCE=1 TY_LLM_REL_ENSEMBLE=1 python Claude-src/05_train.py
!TY_USE_LLM_RELEVANCE=1 TY_LLM_REL_ENSEMBLE=1 python Claude-src/07_predict.py --save-proba
```
Verify a handful of `10_llm_relevance.py` scores by eye first (`--dry-run` uses a
GPU-free mock to check plumbing; a real tiny run checks the model's JSON/1-0
behavior) before trusting either at full scale. Note: the Asure-12B full-3.36M
pass is much slower than Qwen-3B -- if GPU quota is tight, A/B it on a sample of
`submission_pairs` first, or accept a longer run.

## Testing

The full test suite runs on numpy/pandas/scipy/scikit-learn only — **no
lightgbm/xgboost/catboost/torch/sentence-transformers required** — because
`training_utils.py` factors the testable logic (popularity fold-safety,
exact threshold scan, blend/stacking comparison, hard-negative selection,
term->fold assignment) out of `05_train.py`'s heavy-dependency imports, and
`tests/test_llm_batching.py` tests the LLM batching/parsing logic against
`MockLLMClient` rather than a real model:

```bash
pip install -r requirements.txt  # or just: pip install numpy pandas scipy scikit-learn pytest
cd Claude-src
pytest tests/ -v
```

50 tests, covering: UTF-8 round-trip safety for every Turkish string literal,
tokenization/stemming/synonym-cluster symmetry, category-path splitting,
`LexicalIndex`/`compute_batch_features` end-to-end sanity on a synthetic
catalog, the cross-fold popularity leak fix (synthetic catalog reproduction),
the exact threshold scan (cross-checked against a brute-force
`sklearn.f1_score` reference), the nested blend-vs-stacking comparison, the
hard gender/age override (and that brand_contradiction does NOT trigger it),
hard-negative-mining selection logic (`assign_term_folds` cross-checked
directly against `sklearn.GroupKFold`, `select_hard_negatives` threshold/cap/
exclusion behavior), and LLM batching correctness (every row covered
regardless of batch-size/row-count boundary, graceful fallback on unparseable
responses).

**Still NOT verified end-to-end** (no GPU in this sandbox): the actual
embedding fine-tuning run (`00_finetune_embeddings.py`), a real
`hpo_search.py` Optuna run, a real `09_hard_negative_mining.py` mining pass
against trained models, and both LLM scripts against the real 12B model —
only their logic has been verified (`py_compile`, unit tests against
`MockLLMClient`/synthetic data). Each needs a real Kaggle run to confirm it
actually helps before being treated as proven, same caveat as the original
baseline pipeline before its own first real run.

**What is NOT verified end-to-end**: this sandbox has no GPU and no access to
the real embedding/LLM models or the actual competition data at full scale, so
`01_encode_embeddings.py`, `03_negative_sampling.py` (FAISS index build),
`05_train.py`'s actual GBDT training, and both LLM-touching scripts against a
real model have only been verified for syntax/logic correctness (`py_compile`,
and `--dry-run`/`MockLLMClient` plumbing tests), not a full real run — that
must happen on Kaggle, same as `../src/` before it.
