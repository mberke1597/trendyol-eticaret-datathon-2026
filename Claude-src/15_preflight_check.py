"""
Stage 15 — PRE-FLIGHT CHECK: validate a new proba against the 0.894 anchor
BEFORE spending a leaderboard submission.

Anchor: submissions/891_corrected_v2.csv (LB 0.894, pos rate 27.42%,
md5 eda769d6adfb22e5ac26721364c796e4).

Checks (learned from the 0.698 incident, see DETECTIVE_FINDINGS_4):
  1. HEAD HEALTH — among your top-8% highest-proba rows, what fraction does
     the anchor also call relevant? A healthy sibling model: >= 0.80.
     The broken (anti-learned) model would have shown ~0.5 here. This is the
     single check that would have saved 5 submissions.
  2. MONOTONICITY — anchor-agreement of proba deciles must increase toward
     both extremes (top decile and bottom decile most aligned).
  3. RATE at the suggested threshold — final positive rate in [0.26, 0.30].
  4. AGREEMENT at rate-matched cut vs anchor in [0.80, 0.95] — lower means
     something structural changed; higher means you basically resubmitted the
     anchor.

Usage:
  python 15_preflight_check.py path/to/submission_proba.npy
Prints PASS/FAIL per check plus the rate-matched threshold to use.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, WORK_DIR  # noqa: E402

import os
ANCHOR = Path(os.environ.get(
    "TY_ANCHOR",
    Path(__file__).resolve().parent.parent / "submissions" / "891_corrected_v2.csv",
))
TARGET_RATE = 0.275


def main():
    proba_path = sys.argv[1] if len(sys.argv) > 1 else str(WORK_DIR / "output" / "submission_proba.npy")
    p = np.load(proba_path)
    anchor = pd.read_csv(ANCHOR)
    if len(anchor) == 1_048_576:
        raise SystemExit(
            f"ANCHOR FILE IS EXCEL-TRUNCATED: {ANCHOR} has exactly 1,048,576 rows "
            "(= 2^20, Excel's row limit). Someone opened this CSV in Excel and saved it, "
            "silently deleting 2.3M rows. Re-copy the ORIGINAL 891_corrected_v2.csv "
            "(3,359,679 rows, md5 eda769d6adfb22e5ac26721364c796e4) into the dataset and "
            "NEVER open submission CSVs with Excel. Verify with: "
            "python -c \"import pandas as pd; print(len(pd.read_csv('891_corrected_v2.csv')))\""
        )
    assert len(p) == len(anchor), f"proba rows {len(p)} != anchor rows {len(anchor)}"
    y = anchor["prediction"].values.astype(bool)
    n = len(p)
    ok = True

    # 1. head health
    k8 = int(0.08 * n)
    top8 = np.argpartition(-p, k8)[:k8]
    head = float(y[top8].mean())
    passed = head >= 0.80
    ok &= passed
    print(f"[1] head health: top-8% proba rows also 1 in anchor = {head:.3f} "
          f"(>=0.80) {'PASS' if passed else 'FAIL  <-- inverted head, DO NOT SUBMIT'}")

    # 2. decile monotonicity
    qs = np.quantile(p, np.linspace(0, 1, 11))
    rates = []
    for i in range(10):
        m = (p >= qs[i]) & (p <= qs[i + 1])
        rates.append(float(y[m].mean()))
    mono = all(rates[i] <= rates[i + 1] + 0.03 for i in range(9))
    ok &= mono
    print(f"[2] anchor-positive rate by proba decile: {[round(r,2) for r in rates]} "
          f"{'PASS' if mono else 'FAIL  <-- non-monotone ranking'}")

    # 3+4. rate-matched cut
    t = float(np.quantile(p, 1 - TARGET_RATE))
    pred = (p >= t)
    rate = float(pred.mean())
    agree = float((pred == y).mean())
    r_ok = 0.26 <= rate <= 0.30
    a_ok = 0.80 <= agree <= 0.95
    ok &= r_ok and a_ok
    print(f"[3] threshold {t:.5f} -> positive rate {rate:.4f} {'PASS' if r_ok else 'FAIL'}")
    print(f"[4] agreement vs anchor at that cut: {agree:.4f} (0.80-0.95) {'PASS' if a_ok else 'FAIL'}")

    # 5. click-bias (the 0.698 killer, proven 2026-07-16): among anchor-positive
    # rows, clicked items must NOT score systematically below unclicked ones.
    # The anti-learned nbq_item_click_sim run showed median 0.052 (clicked) vs
    # 0.177 (unclicked) = ratio 0.29 -> LB lost ~0.10 at every threshold.
    pairs = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv", usecols=["item_id"])
    clicked_items = set(pd.read_csv(f"{DATA_DIR}/training_pairs.csv", usecols=["item_id"])["item_id"])
    clicked = pairs["item_id"].isin(clicked_items).values
    med_c = float(np.median(p[y & clicked]))
    med_u = float(np.median(p[y & ~clicked]))
    ratio = med_c / (med_u + 1e-9)
    passed = ratio >= 0.60
    ok &= passed
    print(f"[5] click-bias: median proba on anchor-positives clicked={med_c:.4f} vs "
          f"unclicked={med_u:.4f} ratio={ratio:.2f} (>=0.60) "
          f"{'PASS' if passed else 'FAIL  <-- model penalizes click history (anti-learned behavioral feature)'}")

    print(f"\n{'ALL CHECKS PASSED — safe to submit.' if ok else 'CHECKS FAILED — investigate before spending a submission.'}")
    print(f"suggested: TY_THRESHOLD={t:.5f} python 07_predict.py --from-cached-proba")


if __name__ == "__main__":
    main()
