"""
NewPipeline — shared config.

Text-based relevance pipeline built on three Trendyol models:
  - Trendyol/tyroberta                              -> cross-encoder relevance CLASSIFIER (center piece)
  - Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 -> bi-encoder embedding features
  - Trendyol/Trendyol-LLM-8B-T1                      -> generative query enrichment (Qwen3-8B based)

Design principle (learned from the Qwen override failure on the leaderboard):
NOTHING here overrides the final decision. Every text signal enters an existing
learned ensemble as a FEATURE (or an OOF ensemble member), so a weak signal is
automatically down-weighted by the meta-learner instead of blindly flipping rows.

Path auto-detection mirrors Claude-src but is FIXED: we only treat the env as
Kaggle when the competition data actually exists at the Kaggle path (the old
`os.path.exists('/kaggle/input')` check misfired on Colab). Every knob is a TY_*
env override so a script can be dropped into Kaggle/Colab unmodified.
"""
import os
from pathlib import Path

# ---------------- environment / paths ----------------
_KAGGLE_DATA = "/kaggle/input/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle"
# FIX: require the data to actually be there, not just the /kaggle/input mount.
IS_KAGGLE = os.path.exists(f"{_KAGGLE_DATA}/items.csv")

if IS_KAGGLE:
    _DEFAULT_DATA_DIR = _KAGGLE_DATA
    _DEFAULT_WORK_DIR = "/kaggle/working"
else:
    # Colab default; override with TY_DATA_DIR / TY_WORK_DIR anywhere else.
    _DEFAULT_DATA_DIR = "/content/data"
    _DEFAULT_WORK_DIR = "/content/work"

DATA_DIR = Path(os.environ.get("TY_DATA_DIR", _DEFAULT_DATA_DIR))
WORK_DIR = Path(os.environ.get("TY_WORK_DIR", _DEFAULT_WORK_DIR))

# Claude-src cache dir: where the existing 51-feature parquets + fold artifacts live.
# 40_merge_features.py reads train_features.parquet / submission_features.parquet
# from here to append our new columns. Defaults to WORK_DIR/cache (same as Claude-src).
CLAUDE_CACHE_DIR = Path(os.environ.get("TY_CLAUDE_CACHE_DIR", str(WORK_DIR / "cache")))

CACHE_DIR = WORK_DIR / "np_cache"       # NewPipeline's own artifacts
OUTPUT_DIR = WORK_DIR / "np_output"
MODEL_DIR = WORK_DIR / "np_models"
for _d in (CACHE_DIR, OUTPUT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------- models ----------------
CE_MODEL = os.environ.get("TY_CE_MODEL", "Trendyol/tyroberta")
EMBED_MODEL = os.environ.get("TY_EMBED_MODEL", "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0")
LLM_MODEL = os.environ.get("TY_LLM_ENRICH_MODEL", "Trendyol/Trendyol-LLM-8B-T1")

# ---------------- CV / determinism ----------------
RANDOM_SEED = 42
N_FOLDS = 5                              # MUST match Claude-src (GroupKFold by term_id)

# ---------------- cross-encoder (stage 20-22) ----------------
CE_MAX_LEN = int(os.environ.get("TY_CE_MAX_LEN", "192"))       # query + item text tokens
CE_EPOCHS = int(os.environ.get("TY_CE_EPOCHS", "1"))
CE_BATCH_SIZE = int(os.environ.get("TY_CE_BATCH_SIZE", "64"))
CE_EVAL_BATCH_SIZE = int(os.environ.get("TY_CE_EVAL_BATCH_SIZE", "256"))
CE_LR = float(os.environ.get("TY_CE_LR", "2e-5"))
CE_WARMUP_RATIO = float(os.environ.get("TY_CE_WARMUP_RATIO", "0.06"))
CE_FP16 = os.environ.get("TY_CE_FP16", "1") == "1"            # T4-friendly
# Positive class weight for the imbalanced (~18.7% pos) loss. ~ (1-p)/p.
CE_POS_WEIGHT = float(os.environ.get("TY_CE_POS_WEIGHT", "4.35"))
CE_ITEM_TITLE_CHARS = int(os.environ.get("TY_CE_ITEM_TITLE_CHARS", "160"))
CE_ITEM_ATTR_CHARS = int(os.environ.get("TY_CE_ITEM_ATTR_CHARS", "180"))
CE_SHARD_SIZE = int(os.environ.get("TY_CE_SHARD_SIZE", "300000"))   # submission scoring checkpoint

# ---------------- embedding features (stage 30) ----------------
EMBED_BATCH_SIZE = int(os.environ.get("TY_EMBED_BATCH_SIZE", "256"))
EMBED_MATRYOSHKA = [768, 512, 128]      # dims the model exposes (see model card)
KNN_K = int(os.environ.get("TY_KNN_K", "10"))   # query->query neighbours for label transfer

# ---------------- LLM query enrichment (stage 31) ----------------
LLM_TP = int(os.environ.get("TY_LLM_ENRICH_TP", "1"))        # 2 for 2xT4
LLM_MAX_LEN = int(os.environ.get("TY_LLM_ENRICH_MAX_LEN", "1024"))
LLM_DTYPE = os.environ.get("TY_LLM_ENRICH_DTYPE", "half")    # T4 has no bf16
LLM_QUANT = os.environ.get("TY_LLM_ENRICH_QUANT", "")        # e.g. "awq" if using a quantized 8B

# ---------------- output keys (kept in one place so writer/reader never disagree) ----------------
CE_COL = "ce_relevance_prob"
CE_TRAIN_PARQUET = CACHE_DIR / "ce_train_oof.parquet"          # [term_id, item_id, ce_relevance_prob]
CE_SUB_PARQUET = CACHE_DIR / "ce_submission.parquet"          # [id, ce_relevance_prob]
EMBED_TRAIN_PARQUET = CACHE_DIR / "embed_train_feats.parquet"
EMBED_SUB_PARQUET = CACHE_DIR / "embed_submission_feats.parquet"
LLM_QUERY_PARQUET = CACHE_DIR / "llm_query_enrichment.parquet"  # [term_id, ...enrichment cols]


def fold_parquet():
    """Per-row fold assignment for the labeled train set (stage 20).
    [term_id, item_id, label, fold] — GroupKFold(term_id), identical to Claude-src."""
    return CACHE_DIR / "ce_folds.parquet"
