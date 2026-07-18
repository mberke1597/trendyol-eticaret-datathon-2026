"""
Stage 41 — retrain the ensemble with the new text features + cross-encoder member.

Reuses Claude-src's proven, fold-safe 05_train.py / 07_predict.py verbatim rather
than reimplementing stacking. It does that by staging our enriched feature
matrices into Claude-src's cache under the exact filenames those scripts read,
with the cross-encoder OOF exposed as the `llm_rel_score` column so 05_train.py
picks it up BOTH ways:

  - as a GBDT feature (get_feature_cols auto-detects any numeric column), and
  - as the 4th stacking member "llm" when TY_LLM_REL_ENSEMBLE=1 (its train column
    is our leakage-free OOF, its submission column is the 5-fold bagged score —
    exactly the fixed-external-predictor contract 05_train.py documents).

The embed + LLM-query columns ride along purely as extra GBDT features.

Originals are backed up (.bak) and restorable. This script only stages files and
prints the two commands to run; it does not import the digit-prefixed modules.

Usage:
  python 41_train_ensemble.py            # stage enriched features into Claude-src cache
  python 41_train_ensemble.py --restore  # put the original feature parquets back
"""
import argparse, shutil, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from config import CLAUDE_CACHE_DIR, CACHE_DIR, CE_COL

TRAIN_DST = CLAUDE_CACHE_DIR / "train_features.parquet"
SUB_DST = CLAUDE_CACHE_DIR / "submission_features.parquet"
TRAIN_SRC = CACHE_DIR / "train_features_plus.parquet"
SUB_SRC = CACHE_DIR / "submission_features_plus.parquet"


def _alias_ce_as_member(df):
    """Expose the cross-encoder prob as `llm_rel_score` so 05_train.py uses it as
    the 4th ensemble member. Kept as the single member column (drop the duplicate
    ce_relevance_prob name) so it's one feature that is also the member."""
    if CE_COL in df.columns:
        df = df.rename(columns={CE_COL: "llm_rel_score"})
    return df


def restore():
    for dst in (TRAIN_DST, SUB_DST):
        bak = dst.with_suffix(".parquet.bak")
        if bak.exists():
            shutil.move(str(bak), str(dst))
            print(f"[41] restored {dst}")
        else:
            print(f"[41] no backup for {dst} -- nothing to restore")


def stage():
    for src, dst in ((TRAIN_SRC, TRAIN_DST), (SUB_SRC, SUB_DST)):
        if not src.exists():
            raise FileNotFoundError(f"{src} missing -- run 40_merge_features.py first")
        bak = dst.with_suffix(".parquet.bak")
        if dst.exists() and not bak.exists():
            shutil.copy(str(dst), str(bak))
            print(f"[41] backed up {dst} -> {bak}")
        df = _alias_ce_as_member(pd.read_parquet(src))
        df.to_parquet(dst, index=False)
        print(f"[41] staged {dst}  {df.shape}  (member='llm_rel_score' present={'llm_rel_score' in df.columns})")

    print("\n[41] Now run Claude-src's proven trainer + predictor:\n"
          "     cd <Claude-src>\n"
          "     TY_LLM_REL_ENSEMBLE=1 python 05_train.py\n"
          "     TY_LLM_REL_ENSEMBLE=1 python 07_predict.py\n"
          "  (TY_LLM_REL_ENSEMBLE=1 turns the cross-encoder into the 4th stacking member;\n"
          "   omit it to use the CE only as a GBDT feature.)\n"
          "  Restore the baseline features anytime with:  python 41_train_ensemble.py --restore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", action="store_true")
    args = ap.parse_args()
    restore() if args.restore else stage()


if __name__ == "__main__":
    main()
