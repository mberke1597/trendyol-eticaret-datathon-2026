"""
Shared paths/constants for the Trendyol search-relevance pipeline.
Auto-detects Kaggle vs local (WSL) environment so every script in src/
can be dropped into a Kaggle notebook unmodified.
"""
import os
from pathlib import Path

IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    # matches the path used in the organizer-provided baseline notebook
    _DEFAULT_DATA_DIR = "/kaggle/input/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle"
    _DEFAULT_WORK_DIR = "/kaggle/working"
else:
    _DEFAULT_DATA_DIR = "/mnt/c/Users/berke/Desktop/TrendyolE-Ticaret/trendyol-e-ticaret-yarismasi-2026-kaggle"
    _DEFAULT_WORK_DIR = "/mnt/c/Users/berke/Desktop/TrendyolE-Ticaret"

DATA_DIR = Path(os.environ.get("TY_DATA_DIR", _DEFAULT_DATA_DIR))
WORK_DIR = Path(os.environ.get("TY_WORK_DIR", _DEFAULT_WORK_DIR))

CACHE_DIR = WORK_DIR / "cache"
OUTPUT_DIR = WORK_DIR / "output"
MODEL_DIR = WORK_DIR / "models"
for _d in (CACHE_DIR, OUTPUT_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Embedding models (see DatasetDescription.txt / prompt.md section 3)
MAIN_MODEL = "Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0"  # 768d: query/title/category
TINY_MODEL = "atasoglu/turkish-tiny-bert-uncased-mean-nli-stsb-tr"  # 128d: query/attributes

RANDOM_SEED = 42
N_FOLDS = 5

# README: click-through density in the hidden test set is expected around 30%
TARGET_POS_RATE = 0.30

# negative sampling mix (see 02_negative_sampling.py)
# Mirrors the org's "BM25 + FAISS" retriever so training negatives match the
# hard-candidate distribution of submission_pairs.csv instead of easy random pairs.
NEG_PER_POS_MULTIPLIER = 6   # candidates mined per positive before capping
MIN_NEG_PER_TERM = 20
MAX_NEG_PER_TERM = 150
NEG_SOURCE_RATIOS = {
    "dense_ann": 0.35,     # FAISS-style: nearest items to the query embedding (semantic near-miss)
    "lexical": 0.25,       # BM25-style: inverted-index token overlap (lexical near-miss)
    "category_sibling": 0.20,  # same leaf category as a true positive, different item (fine-grained confusion)
    "popularity_random": 0.20,  # half uniform / half popularity-weighted, to counter popularity bias
}
