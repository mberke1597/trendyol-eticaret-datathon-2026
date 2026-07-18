"""
Stage 14 — HOTFIX: drop the anti-learned nbq_item_click_sim column and retrain.

Fastest recovery path on Kaggle (no need to re-run 04/12/13):

  python Claude-src/14_drop_bad_feature.py     # removes the column from both parquets
  python Claude-src/05_train.py                # retrain (~same time as before)
  python Claude-src/07_predict.py --save-proba
  # then rate-match the threshold (target ~27.5% final positive rate):
  #   TY_THRESHOLD=<value printed by the sweep> python 07_predict.py --from-cached-proba

Why: see the warning block in 13_query_neighbor_features.py — the feature's
train-side LOO made it an INVERSE label proxy in training (positives ~0,
negatives keep click profiles), so the model anti-learned it; at test time the
semantics flip and the head of the ranking is poisoned (LB 0.698 at any rate).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CACHE_DIR  # noqa: E402

BAD = ["nbq_item_click_sim"]

for name in ["train_features.parquet", "submission_features.parquet"]:
    path = Path(f"{CACHE_DIR}/{name}")
    df = pd.read_parquet(path)
    present = [c for c in BAD if c in df.columns]
    if not present:
        print(f"[14] {name}: {BAD} not present, nothing to do")
        continue
    df = df.drop(columns=present)
    df.to_parquet(path, index=False)
    print(f"[14] {name}: dropped {present}, shape now {df.shape}")

print("[14] done. Retrain: python 05_train.py && python 07_predict.py --save-proba")
