"""
Shared paths/constants for the Claude-src Trendyol search-relevance pipeline.

Independent of `../src/` (kept as reference/fallback). Auto-detects Kaggle vs
local (WSL) environment exactly like `../src/config.py` so every script here
can be dropped into a Kaggle notebook unmodified.

Key differences vs. `../src/config.py` (see ../Claude-src/DESIGN.md for the
full rationale of every decision below):
  - Negative sampling density is calibrated toward the *empirically discovered*
    real test click density (~28-31%, found via 7 real leaderboard submissions
    this competition), NOT an artificial 50/50 balance (a mistake found in a
    teammate's train_stacking.py -- see DESIGN.md "Dersler" section).
  - Adds LLM-enrichment and uncertain-band re-scoring feature flags, both
    OFF by default so the pipeline runs fully without any LLM/extra GPU
    budget -- these are additive, optional stages, never required.
"""
import os
from pathlib import Path

IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    _DEFAULT_DATA_DIR = "/kaggle/input/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle"
    _DEFAULT_WORK_DIR = "/kaggle/working"
else:
    _DEFAULT_DATA_DIR = "/mnt/c/Users/berke/Desktop/TrendyolE-Ticaret/trendyol-e-ticaret-yarismasi-2026-kaggle"
    _DEFAULT_WORK_DIR = "/mnt/c/Users/berke/Desktop/TrendyolE-Ticaret/Claude-src"

DATA_DIR = Path(os.environ.get("TY_DATA_DIR", _DEFAULT_DATA_DIR))
WORK_DIR = Path(os.environ.get("TY_WORK_DIR", _DEFAULT_WORK_DIR))

CACHE_DIR = WORK_DIR / "cache"
OUTPUT_DIR = WORK_DIR / "output"
MODEL_DIR = WORK_DIR / "models"
for _d in (CACHE_DIR, OUTPUT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------- embedding models (real, verified Trendyol/HF models) ----------------
MAIN_MODEL = "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0"  # 768d: query/title/category
TINY_MODEL = "atasoglu/turkish-tiny-bert-uncased-mean-nli-stsb-tr"  # 128d: query/attributes

RANDOM_SEED = 42
N_FOLDS = 5

# README: click-through density in the hidden test set was ~30% a priori; real LB
# experiments this competition converged on a plateau at ~28-31% predicted-positive
# rate (0.83 macro-F1). Negative sampling volume below is calibrated to produce a
# natural OOF positive rate in that neighborhood -- NOT an artificial 50/50 balance
# (see DESIGN.md for why 50/50 balancing was a real bug found in a teammate's script).
TARGET_POS_RATE = 0.30

# negative sampling mix (see 03_negative_sampling.py) -- mirrors the org's "BM25 +
# FAISS" retriever so training negatives match the hard-candidate distribution of
# submission_pairs.csv instead of easy random pairs.
NEG_PER_POS_MULTIPLIER = 6   # candidates mined per positive before capping
MIN_NEG_PER_TERM = 20
MAX_NEG_PER_TERM = 150
NEG_SOURCE_RATIOS = {
    "dense_ann": 0.35,
    "lexical": 0.25,
    "category_sibling": 0.20,
    "popularity_random": 0.20,
}

# ---------------- LLM enrichment (optional, offline/nearline only) ----------------
# NEVER used in the 3.36M-row hot inference path (see DESIGN.md "buyuk sirket
# arastirmasi" section: Walmart/Amazon pattern is LLM-for-labels, GBDT-for-serving).
# Both flags default to False so the pipeline is fully runnable without any LLM
# access or extra GPU budget -- they are additive quality upgrades, not requirements.
USE_LLM_ENRICHMENT = os.environ.get("TY_USE_LLM_ENRICHMENT", "0") == "1"
LLM_MODEL_NAME = os.environ.get("TY_LLM_MODEL", "Trendyol/Trendyol-LLM-Asure-12B")
# Only enrich items that actually appear in train+candidate pairs, not the full
# 962k catalog -- Kaggle's ~30h/week GPU quota can't cover a 12B-param pass over
# the whole catalog.
LLM_ENRICHMENT_MAX_ITEMS = int(os.environ.get("TY_LLM_MAX_ITEMS", "50000"))
# Real-use hardening (2026-07-04): the original one-prompt-at-a-time generate()
# loop would need ~50k SEQUENTIAL forward passes for terms.csv alone -- at even
# 1-2s/call for a 12B model that's 14-28 CPU/GPU-hours, likely blowing past a
# single Kaggle session. LLMClient now batches prompts (see 02_llm_enrichment.py);
# this is the batch size used for that.
LLM_BATCH_SIZE = int(os.environ.get("TY_LLM_BATCH_SIZE", "16"))

# ---------------- LLM pairwise relevance score (optional) ----------------
# Added 2026-07-11. A SEPARATE, heavier use of an LLM than 02_llm_enrichment.py:
# instead of extracting structured attributes offline, 10_llm_relevance.py asks
# an LLM directly "is this item relevant to this query?" for EVERY (query,item)
# pair -- both the labeled train set AND all 3.36M submission_pairs -- producing
# a continuous 0..1 relevance score per pair (from the 1-vs-0 first-token
# logprob, so one generated token per pair -> feasible at full scale via vLLM,
# unlike the ~100-token structured extraction in 02). The score is then used
# TWO ways (both, per the user's request):
#   (a) as an extra GBDT feature (llm_rel_score column, joined in
#       04_build_features.py -- get_feature_cols picks it up automatically), and
#   (b) as a 4th member of 05_train.py's blend/stacking ensemble alongside
#       lgb/xgb/cat (LLM_REL_AS_ENSEMBLE_MEMBER).
# Both default OFF so the baseline pipeline is unchanged if the LLM stage is
# skipped. NOT the 02_llm_enrichment "offline structured extraction" flag --
# that one stays independent (USE_LLM_ENRICHMENT).
USE_LLM_RELEVANCE = os.environ.get("TY_USE_LLM_RELEVANCE", "0") == "1"
LLM_REL_AS_ENSEMBLE_MEMBER = os.environ.get("TY_LLM_REL_ENSEMBLE", "0") == "1"
# Default to the small/fast text-only model (my earlier recommendation): makes a
# full 3.36M-pair pass feasible in a few GPU-hours on Kaggle's 2xT4. Swap to the
# e-commerce-tuned (but ~10-40x slower) Asure-12B via TY_LLM_REL_MODEL to A/B.
LLM_REL_MODEL = os.environ.get("TY_LLM_REL_MODEL", "Qwen/Qwen2.5-3B-Instruct")
# Cache-file suffix so two models' scores can coexist for comparison, e.g.
# TY_LLM_REL_TAG=qwen3b then TY_LLM_REL_TAG=asure12b. 04_build_features.py reads
# the SAME tag to pick which score to join, so set it consistently across
# 10_llm_relevance.py -> 04_build_features.py for a given experiment.
LLM_REL_TAG = os.environ.get("TY_LLM_REL_TAG", "")
# Neutral value for any pair with no LLM score (shouldn't happen at full
# coverage, but keeps both the GBDT feature and the ensemble member well-defined
# -- 0.5 = "model is maximally unsure", a sensible prior for a relevance proba).
LLM_REL_FILL = float(os.environ.get("TY_LLM_REL_FILL", "0.5"))
# vLLM knobs. TP=2 uses both T4s (real tensor-parallelism, NOT HF's naive
# per-layer pipeline split -- see the 02_llm_enrichment.py / earlier-chat
# finding on why device_map="auto" was ruinous for generation). max_model_len
# kept small since prompts are short (one query + a truncated item blurb) and a
# smaller KV-cache window means more room for continuous batching throughput.
# vLLM compute dtype. Default fp16 ("half") -- correct for T4 (Turing has no
# bf16) and fine for text-only models like Qwen2.5. NOTE: Gemma3-based models
# (e.g. Trendyol-LLM-Asure-12B) REFUSE fp16 in vLLM ("numerical instability")
# and want bf16/fp32 -- but T4 has no bf16 and a 12B fp32 model won't fit 2xT4,
# so Asure-12B can't run on a T4 at all (needs Ampere+). Override with
# TY_LLM_REL_DTYPE=bfloat16 only on an Ampere+/A100/L4 GPU.
LLM_REL_DTYPE = os.environ.get("TY_LLM_REL_DTYPE", "half")
LLM_REL_TENSOR_PARALLEL = int(os.environ.get("TY_LLM_REL_TP", "2"))
LLM_REL_MAX_MODEL_LEN = int(os.environ.get("TY_LLM_REL_MAX_LEN", "1024"))
LLM_REL_GPU_MEM_UTIL = float(os.environ.get("TY_LLM_REL_GPU_MEM_UTIL", "0.90"))
# Char caps on the item text put into the prompt -- attributes strings in
# items.csv can be enormous (hundreds of chars); truncating keeps prompts short
# so prefill (which dominates single-token scoring cost) stays cheap.
LLM_REL_TITLE_CHARS = int(os.environ.get("TY_LLM_REL_TITLE_CHARS", "200"))
LLM_REL_ATTR_CHARS = int(os.environ.get("TY_LLM_REL_ATTR_CHARS", "300"))
# Rows per shard checkpoint. A full 3.36M pass is long enough that a Kaggle
# session can die mid-run; 10_llm_relevance.py writes one parquet shard per
# SHARD_SIZE rows and skips shards already on disk, so a re-run resumes instead
# of restarting from zero.
LLM_REL_SHARD_SIZE = int(os.environ.get("TY_LLM_REL_SHARD_SIZE", "200000"))


def llm_rel_cache_paths(tag=None):
    """(train_path, submission_path) for the cached relevance scores, suffixed
    by tag so multiple models' scores coexist. Kept as a helper so
    10_llm_relevance.py (writer) and 04_build_features.py (reader) can never
    disagree on the filename."""
    tag = LLM_REL_TAG if tag is None else tag
    suffix = f"_{tag}" if tag else ""
    return (
        CACHE_DIR / f"llm_relevance_train{suffix}.parquet",
        CACHE_DIR / f"llm_relevance_submission{suffix}.parquet",
    )


# ---------------- Turkish morphology / typo-tolerance features (optional) ----
# Added 2026-07-07. OFF by default -- see New-Pipeline/DESIGN.md for the full
# rationale and what's been verified vs. not. NOT yet validated against a real
# Kaggle submission (fire-rate/throughput must be checked first via
# New-Pipeline/benchmark_on_real_data.py); do not assume this helps without
# that check, same discipline as every other optional stage in this file.
# Requires `pip install zeyrek symspellpy` (not in the default requirements.txt
# install -- see that file's comment, same pattern as optuna).
USE_TURKISH_MORPHOLOGY = os.environ.get("TY_USE_TURKISH_MORPHOLOGY", "0") == "1"

# ---------------- uncertain-band re-scoring (optional) ----------------
USE_UNCERTAIN_BAND_RESCORE = os.environ.get("TY_USE_RESCORE", "0") == "1"
# FIXED 2026-07-05: these were hardcoded, assuming the F1-optimal threshold
# would land somewhere near 0.5. Real runs this competition have repeatedly
# found the ACTUAL working threshold sits much lower (~0.15-0.20, see
# DESIGN.md's threshold-calibration lessons) -- with a fixed 0.35-0.65 band,
# the "uncertain" region re-scored here would sit nowhere near the real
# decision boundary, wasting the LLM pass on rows that barely matter. Made
# env-overridable so the band can be recentered around whatever TY_THRESHOLD
# actually gets used, e.g. TY_UNCERTAIN_BAND_LOW=0.10 TY_UNCERTAIN_BAND_HIGH=0.30
# for a ~0.17-0.20 working threshold.
UNCERTAIN_BAND_LOW = float(os.environ.get("TY_UNCERTAIN_BAND_LOW", "0.35"))
UNCERTAIN_BAND_HIGH = float(os.environ.get("TY_UNCERTAIN_BAND_HIGH", "0.65"))
# Added 2026-07-08: "override" (default, original behavior) replaces the GBDT's
# band predictions outright with the LLM's binary relevant/not-relevant call.
# "blend" is a new, softer combination -- the two relevancy signals (GBDT proba,
# LLM confidence) are weighted-averaged into one proba and re-thresholded,
# mirroring how 05_train.py's own lgb/xgb/cat ensemble is a weighted blend, not
# a hard vote. NOT yet validated against a real submission -- see DESIGN.md.
RESCORE_MODE = os.environ.get("TY_RESCORE_MODE", "override")  # "override" | "blend"
RESCORE_BLEND_LLM_WEIGHT = float(os.environ.get("TY_RESCORE_BLEND_LLM_WEIGHT", "0.5"))

# ---------------- stacking meta-model ----------------
# Nested validation split fraction used to score the meta-learner honestly instead
# of refitting-and-reporting-on-the-same-OOF-rows (mild optimism found in a
# teammate's stacking script -- see DESIGN.md).
META_NESTED_HOLDOUT_FRAC = 0.20

# ---------------- iterative hard-negative mining (optional, round 2) ----------------
# See 09_hard_negative_mining.py. Round-1 negatives (03_negative_sampling.py) are
# mined from ANN/lexical/category-sibling similarity to the QUERY -- they mimic the
# retriever's blind spots, not the round-1 MODEL's blind spots. This second, optional
# round scores a wider candidate pool per training term with each term's own
# out-of-fold (never-trained-on-this-term) fold model, and keeps candidates the
# model confidently (falsely) scores as positive -- these are the negatives the
# model is actually confusing right now, which round-1 mining structurally can't
# find. Off by default (needs round-1 models to already exist); enable by running
# 09_hard_negative_mining.py after 05_train.py's first pass, then re-running
# 04_build_features.py (pointed at TY_TRAIN_PAIRS_FILE=train_pairs_labeled_round2.parquet)
# and 05_train.py again for the round-2 model.
HARD_NEG_ANN_K = int(os.environ.get("TY_HARD_NEG_ANN_K", "150"))  # wider than round-1's ANN_SEARCH_K=80
HARD_NEG_SCORE_THRESHOLD = float(os.environ.get("TY_HARD_NEG_SCORE_THRESHOLD", "0.5"))
HARD_NEG_MAX_PER_TERM = int(os.environ.get("TY_HARD_NEG_MAX_PER_TERM", "30"))

# ---------------- hyperparameter search (optional) ----------------
# See hpo_search.py. Off the hot path entirely -- a standalone script that writes
# best_hyperparams.json, which 05_train.py loads INSTEAD OF its hardcoded
# hyperparameters if the file exists (falls back to the proven hardcoded values
# otherwise, so 05_train.py never breaks if this step is skipped).
HPO_N_TRIALS = int(os.environ.get("TY_HPO_N_TRIALS", "25"))
HPO_TIMEOUT_SECONDS = int(os.environ.get("TY_HPO_TIMEOUT_SECONDS", "3600"))

# ---------------- embedding fine-tuning (optional) ----------------
# See 00_finetune_embeddings.py. Off by default: 01_encode_embeddings.py uses
# MAIN_MODEL (the base HF hub id) unless TY_FINETUNED_MODEL_DIR is set, in which
# case it loads the fine-tuned checkpoint from that local path instead.
FINETUNE_OUTPUT_DIR = WORK_DIR / "finetuned_embed"
FINETUNED_MODEL_DIR = os.environ.get("TY_FINETUNED_MODEL_DIR", "")
FINETUNE_EPOCHS = int(os.environ.get("TY_FINETUNE_EPOCHS", "1"))
FINETUNE_BATCH_SIZE = int(os.environ.get("TY_FINETUNE_BATCH_SIZE", "64"))
FINETUNE_LR = float(os.environ.get("TY_FINETUNE_LR", "2e-5"))
# Held-out term_id fraction during fine-tuning, to catch overfitting to the
# training terms' specific vocabulary before it ever reaches 01_encode_embeddings.py --
# same cold-start rationale as GroupKFold(term_id) elsewhere in this pipeline.
FINETUNE_VAL_TERM_FRAC = 0.10
