"""
Stage 80 — LOCAL macro-F1 scorer + threshold optimizer.  STOP FLYING BLIND.

Competition metric (official): macro-averaged F1 = (F1_relevant + F1_irrelevant)/2,
computed GLOBALLY over all rows. Leaderboard is on a Public subset; final on Private.

This module lets us measure ANY model's macro-F1 locally on out-of-fold (OOF)
predictions, and pick the macro-F1-OPTIMAL threshold — instead of guessing on the
leaderboard. It also reports the per-class F1 so we see whether we're losing on the
relevant class (usually the bottleneck) or the irrelevant class.

CRITICAL CALIBRATION NOTE:
  Our OOF is built on MINED negatives (~18.7% positive), but the TEST set is
  ~28-31% positive. macro-F1 and its optimal threshold depend on class balance, so
  a threshold tuned on the 18.7% OOF will NOT be the test-optimal one. Two options
  are provided:
    (1) --oof-optimal : threshold that maximises macro-F1 on the OOF as-is.
    (2) --target-rate 0.29 : threshold giving ~29% predicted-positive (matches the
        empirically LB-best density; safer for the real test distribution).
  Use (2) for submissions; use (1) to compare MODELS at their own best operating
  point. Report BOTH so the gap is visible.

Usage:
  # tune on OOF (needs oof probabilities + true labels aligned)
  python 80_macrof1_scorer.py --oof-proba oof_proba.npy --oof-labels oof_labels.npy
  # apply a chosen threshold to a submission probability array -> submission.csv
  python 80_macrof1_scorer.py --apply submission_proba.npy --ids submission_pairs.csv \
      --threshold 0.31 --out submission.csv
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd


def macro_f1(y_true, y_pred):
    """Official metric: (F1_relevant + F1_irrelevant)/2, computed globally."""
    y_true = y_true.astype(np.int8); y_pred = y_pred.astype(np.int8)
    def f1(pos):
        yt = (y_true == pos); yp = (y_pred == pos)
        tp = int((yt & yp).sum()); fp = int((~yt & yp).sum()); fn = int((yt & ~yp).sum())
        if tp == 0:
            return 0.0
        p = tp / (tp + fp); r = tp / (tp + fn)
        return 2 * p * r / (p + r)
    f1_rel = f1(1); f1_irr = f1(0)
    return (f1_rel + f1_irr) / 2, f1_rel, f1_irr


def sweep(proba, labels, grid=None):
    """Return the macro-F1-optimal threshold on (proba, labels) and a table."""
    if grid is None:
        grid = np.unique(np.quantile(proba, np.linspace(0.02, 0.98, 97)))
    best = (-1, 0.5)
    rows = []
    for t in grid:
        m, fr, fi = macro_f1(labels, (proba >= t).astype(np.int8))
        rows.append((t, m, fr, fi, float((proba >= t).mean())))
        if m > best[0]:
            best = (m, t)
    return best[1], best[0], pd.DataFrame(rows, columns=["thr", "macroF1", "F1_rel", "F1_irr", "pos_rate"])


def threshold_for_rate(proba, rate):
    return float(np.quantile(proba, 1 - rate))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-proba"); ap.add_argument("--oof-labels")
    ap.add_argument("--apply"); ap.add_argument("--ids"); ap.add_argument("--threshold", type=float)
    ap.add_argument("--target-rate", type=float, default=0.29)
    ap.add_argument("--out")
    args = ap.parse_args()

    if args.oof_proba:
        p = np.load(args.oof_proba); y = np.load(args.oof_labels)
        thr, best, tbl = sweep(p, y)
        print(f"[80] OOF rows={len(y):,} pos_rate={y.mean():.3f}")
        print(f"[80] OOF-optimal threshold={thr:.4f}  macroF1={best:.4f}")
        r = tbl.iloc[(tbl.thr - thr).abs().values.argmin()]
        print(f"      -> F1_rel={r.F1_rel:.4f}  F1_irr={r.F1_irr:.4f}  pos_rate={r.pos_rate:.3f}")
        tr = threshold_for_rate(p, args.target_rate)
        m2, fr2, fi2 = macro_f1(y, (p >= tr).astype(np.int8))
        print(f"[80] threshold@{args.target_rate:.0%}-positive={tr:.4f}  macroF1={m2:.4f} "
              f"(F1_rel={fr2:.4f} F1_irr={fi2:.4f})")
        print("[80] NOTE: F1_rel is almost always the bottleneck -> improving the model's "
              "ranking of relevant items is what raises the score, not the threshold.")

    if args.apply:
        p = np.load(args.apply)
        ids = pd.read_csv(args.ids)["id"].values
        assert len(ids) == len(p)
        thr = args.threshold if args.threshold is not None else threshold_for_rate(p, args.target_rate)
        pred = (p >= thr).astype(np.int8)
        pd.DataFrame({"id": ids, "prediction": pred}).to_csv(args.out, index=False)
        print(f"[80] applied thr={thr:.4f} pos_rate={pred.mean():.3f} -> {args.out}")


if __name__ == "__main__":
    main()
