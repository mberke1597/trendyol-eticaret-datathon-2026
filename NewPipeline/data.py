"""
NewPipeline — shared data loading + text assembly.

Central place for:
  - loading items/terms/pairs from DATA_DIR (works on Kaggle/Colab),
  - building the canonical query string and item string used by BOTH the
    cross-encoder (stage 21/22) and any text model, so train and inference
    text is assembled identically (no train/serve skew),
  - assigning GroupKFold(term_id) folds IDENTICAL to Claude-src's 05_train.py,
    which is what makes the cross-encoder's OOF predictions leakage-free and
    stackable with the existing lgb/xgb/cat OOF.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from config import (
    DATA_DIR, CLAUDE_CACHE_DIR, N_FOLDS,
    CE_ITEM_TITLE_CHARS, CE_ITEM_ATTR_CHARS,
)


def _clip(s, n):
    s = "" if s is None else str(s)
    return s[:n]


def load_catalog():
    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    for c in ["title", "category", "brand", "gender", "age_group", "attributes"]:
        if c in items:
            items[c] = items[c].fillna("")
    terms["query"] = terms["query"].fillna("")
    return items, terms


def build_item_text(items, title_chars=CE_ITEM_TITLE_CHARS, attr_chars=CE_ITEM_ATTR_CHARS):
    """One compact Turkish string per item for the cross-encoder / LLM.
    Short by design: title dominates relevance, attributes are truncated so the
    query+item pair stays within the token budget and prefill stays cheap.
    Returns a pd.Series indexed by item_id."""
    def one(r):
        parts = [_clip(r["title"], title_chars)]
        if r.get("category"): parts.append(f"kategori: {r['category']}")
        if r.get("brand"):    parts.append(f"marka: {r['brand']}")
        g = r.get("gender", "")
        if g and g != "unknown": parts.append(f"cinsiyet: {g}")
        a = r.get("age_group", "")
        if a and a != "unknown": parts.append(f"yaş: {a}")
        at = _clip(r.get("attributes", ""), attr_chars)
        if at: parts.append(at)
        return " | ".join(parts)
    txt = items.apply(one, axis=1)
    return pd.Series(txt.values, index=items["item_id"].values)


def query_series(terms):
    return pd.Series(terms["query"].values, index=terms["term_id"].values)


def load_train_labeled():
    """Canonical labeled train set produced by Claude-src stage 03.
    Columns: term_id, item_id, label[, neg_source]. Row order is preserved so
    GroupKFold matches 05_train.py exactly.

    2026-07-15: prefers train_pairs_labeled_clean.parquet (stage 62's
    veto-cleaned labels) when it exists — the previous cross-encoder was
    trained on ~29% poisoned negatives (likely-relevant items labeled 0),
    which is why it underperformed (F1_rel 0.61); never train a CE on the
    dirty file again. Override with TY_CE_LABELS=<filename> if needed."""
    fname = os.environ.get("TY_CE_LABELS")
    if fname:
        p = CLAUDE_CACHE_DIR / fname
    else:
        p = CLAUDE_CACHE_DIR / "train_pairs_labeled_clean.parquet"
        if not p.exists():
            p = CLAUDE_CACHE_DIR / "train_pairs_labeled.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing. Run Claude-src stage 03_negative_sampling.py first "
            "(then NewPipeline/62_clean_negatives.py for the clean version)."
        )
    print(f"[data] labeled train set: {p.name}"
          + ("  (CLEAN, veto-filtered)" if "clean" in p.name else "  (DIRTY — run 62_clean_negatives.py!)"))
    df = pd.read_parquet(p)
    keep = [c for c in ["term_id", "item_id", "label", "neg_source"] if c in df.columns]
    return df[keep].reset_index(drop=True)


def load_submission_pairs():
    return pd.read_csv(f"{DATA_DIR}/submission_pairs.csv")   # id, term_id, item_id


def assign_folds(train_labeled):
    """GroupKFold(N_FOLDS) by term_id, deterministic, identical to Claude-src.
    Returns an int fold array aligned to train_labeled's row order."""
    y = train_labeled["label"].values
    groups = train_labeled["term_id"].values
    folds = np.full(len(train_labeled), -1, dtype=np.int8)
    gkf = GroupKFold(n_splits=N_FOLDS)
    for f, (_, val_idx) in enumerate(gkf.split(train_labeled, y, groups)):
        folds[val_idx] = f
    assert (folds >= 0).all(), "some rows unassigned to a fold"
    return folds
