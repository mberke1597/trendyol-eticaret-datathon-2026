"""
Stage 50 — build submissions from the cross-encoder directly.

Your "classify, don't threshold" request in its purest form: the cross-encoder's
2-class head gives P(relevant) per pair, so the classification decision is
argmax == (prob >= 0.5). No hand-tuned threshold.

Writes two CE-only submissions for A/B on the leaderboard:
  ce_argmax.csv        prediction = 1 if ce_relevance_prob >= 0.5   (pure classifier)
  ce_density30.csv     threshold set so ~30% predicted positive      (matches the
                       competition's empirical positive rate; robust if the head
                       is mis-calibrated)

The stronger submission is expected to be the ENSEMBLE one (Claude-src 07_predict
after stage 41), which uses the CE as a feature + 4th member rather than alone —
this file is for isolating how much signal the cross-encoder carries by itself.

Run:  python 50_make_submission.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from config import CE_SUB_PARQUET, CE_COL, OUTPUT_DIR
from data import load_submission_pairs

TARGET_POS_RATE = 0.30


def main():
    if not Path(CE_SUB_PARQUET).exists():
        raise FileNotFoundError(f"{CE_SUB_PARQUET} missing -- run 22_score_crossencoder.py first")
    ce = pd.read_parquet(CE_SUB_PARQUET)                 # id, ce_relevance_prob
    sub = load_submission_pairs()[["id"]].merge(ce, on="id", how="left")
    sub[CE_COL] = sub[CE_COL].fillna(0.5)
    p = sub[CE_COL].values

    argmax = (p >= 0.5).astype(np.int8)
    pd.DataFrame({"id": sub["id"], "prediction": argmax}).to_csv(OUTPUT_DIR / "ce_argmax.csv", index=False)
    print(f"[50] ce_argmax.csv  pos_rate={argmax.mean()*100:.1f}%")

    thr = np.quantile(p, 1 - TARGET_POS_RATE)
    dens = (p >= thr).astype(np.int8)
    pd.DataFrame({"id": sub["id"], "prediction": dens}).to_csv(OUTPUT_DIR / "ce_density30.csv", index=False)
    print(f"[50] ce_density30.csv thr={thr:.3f} pos_rate={dens.mean()*100:.1f}%")
    print(f"[50] wrote -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
