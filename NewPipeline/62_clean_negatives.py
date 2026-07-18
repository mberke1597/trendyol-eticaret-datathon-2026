"""
Stage 62 — NEGATIVE VETO: remove poisoned (likely-relevant) mined negatives.

SMOKING GUN (measured on this dataset, 2026-07-15 detective session):
  Full query-token containment (all query tokens present as TOKENS in item
  title+category+brand+attributes):
    positives                 74.6%
    dense_ann  "negatives"    45.0%   <-- almost as relevant as positives!
    lexical    "negatives"    15.6%
    category_sibling           9.1%
    popularity_random          0.1%
  dense_ann is ~35% of all negatives. The model is being TAUGHT "items that
  match the query well are irrelevant" -> caps AUC ~0.93, wrecked round-3
  hard-negative mining (LB 0.51), and sabotaged the cross-encoder (F1_rel 0.61
  was trained on these bad labels).

FIX: VETO (drop) any label=0 row that looks relevant:
  rule A: full query-token containment in item text (token-set membership,
          Turkish-normalized — NOT substring matching, "kot" must not match
          "dakota").
  rule B: query<->item-title embedding cosine >= auto threshold. The threshold
          is the P-th percentile of TRUE-POSITIVE cosines (default P=75): if a
          "negative" is as semantically close as the top quartile of real
          positives, we refuse to teach the model it's negative.
Vetoed rows are DROPPED, not relabeled (we don't assert they're positive).
Positives are untouched. Dropping also nudges pos_rate up toward the real test
density (~28-31%) — a side benefit.

Output: CLAUDE_CACHE_DIR/train_pairs_labeled_clean.parquet
Then retrain everything on it:
  TY_TRAIN_PAIRS_FILE=train_pairs_labeled_clean.parquet python Claude-src/04_build_features.py
  python Claude-src/05_train.py
  python Claude-src/07_predict.py
  # and rebuild the cross-encoder dataset from clean labels (20/21/22 — stage 20
  # reads the clean file automatically when it exists, see data.py).

Run:  python 62_clean_negatives.py                    # rules A+B (B auto-threshold)
      python 62_clean_negatives.py --no-sim-veto      # rule A only
      python 62_clean_negatives.py --sim-veto 0.82    # rule B manual threshold
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from config import DATA_DIR, CLAUDE_CACHE_DIR

_SPLIT = re.compile(r"[^0-9a-zçğıöşü]+")
# Words too generic to carry relevance signal on their own; ignored on the
# QUERY side so "kadın için mont" == "kadın mont".
_STOP = {"ve", "ile", "için", "bir", "the"}


def norm(s):
    """Turkish-aware casefold. Plain .lower() maps 'I'->'i' which is WRONG for
    Turkish (I->ı, İ->i) and silently breaks containment for any query typed
    with uppercase I (e.g. 'IŞIK')."""
    if not isinstance(s, str):
        return ""
    return s.replace("İ", "i").replace("I", "ı").lower().replace("i̇", "i")


def toks(s, min_len=2):
    return {t for t in _SPLIT.split(norm(s)) if len(t) >= min_len and t not in _STOP}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-file", default="train_pairs_labeled.parquet")
    ap.add_argument("--out-file", default="train_pairs_labeled_clean.parquet")
    ap.add_argument("--sim-veto", type=float, default=None,
                    help="manual cosine threshold for rule B (overrides auto)")
    ap.add_argument("--no-sim-veto", action="store_true", help="disable rule B entirely")
    ap.add_argument("--sim-veto-pos-pct", type=float, default=75.0,
                    help="auto rule-B threshold = this percentile of TRUE-POSITIVE cosines")
    args = ap.parse_args()

    lab = pd.read_parquet(CLAUDE_CACHE_DIR / args.in_file)
    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    tmap = dict(zip(terms.term_id, terms["query"].astype(str)))
    neg = (lab["label"] == 0).values

    # ---- rule A: token-set containment ----
    # Tokenize each unique item ONCE (the expensive part), then a cheap
    # set-subset check per row.
    need = set(lab.item_id)
    item_tok = {}
    for ch in pd.read_csv(f"{DATA_DIR}/items.csv",
                          usecols=["item_id", "title", "category", "brand", "attributes"],
                          chunksize=200_000):
        ch = ch[ch.item_id.isin(need)]
        txt = (ch.title.fillna("") + " " + ch.category.fillna("") + " "
               + ch.brand.fillna("") + " " + ch.attributes.fillna(""))
        for iid, t in zip(ch.item_id.values, txt.values):
            item_tok[iid] = toks(t)

    qtok_cache = {t: toks(q) for t, q in tmap.items()}
    contain = np.zeros(len(lab), dtype=bool)
    for k, (t, i) in enumerate(zip(lab.term_id.values, lab.item_id.values)):
        q = qtok_cache.get(t)
        if q and q <= item_tok.get(i, frozenset()):
            contain[k] = True
    veto = neg & contain
    print(f"[62] rule A (token containment) vetoes {int(veto.sum()):,} negatives "
          f"(containment rate: pos={contain[~neg].mean():.3f} neg={contain[neg].mean():.3f})")

    # ---- rule B: embedding cosine, auto-thresholded on positive sims ----
    if not args.no_sim_veto:
        try:
            qemb = np.load(f"{CLAUDE_CACHE_DIR}/query_emb_main.npy").astype(np.float32)
            iemb = np.load(f"{CLAUDE_CACHE_DIR}/item_title_emb.npy").astype(np.float32)
            tid = np.load(f"{CLAUDE_CACHE_DIR}/term_id.npy", allow_pickle=True).astype(str)
            iid = np.load(f"{CLAUDE_CACHE_DIR}/item_id.npy", allow_pickle=True).astype(str)
            qemb /= np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-9
            iemb /= np.linalg.norm(iemb, axis=1, keepdims=True) + 1e-9
            tpos = {t: k for k, t in enumerate(tid)}
            ipos = {i: k for k, i in enumerate(iid)}
            ti = np.fromiter((tpos[t] for t in lab.term_id.values), dtype=np.int64, count=len(lab))
            ii = np.fromiter((ipos[i] for i in lab.item_id.values), dtype=np.int64, count=len(lab))
            cos = np.empty(len(lab), dtype=np.float32)
            for s in range(0, len(lab), 500_000):
                e = min(s + 500_000, len(lab))
                cos[s:e] = np.einsum("ij,ij->i", qemb[ti[s:e]], iemb[ii[s:e]])
            if args.sim_veto is not None:
                thr = args.sim_veto
                print(f"[62] rule B manual threshold: {thr:.4f}")
            else:
                thr = float(np.percentile(cos[~neg], args.sim_veto_pos_pct))
                print(f"[62] rule B auto threshold: {thr:.4f} "
                      f"(P{args.sim_veto_pos_pct:g} of true-positive cosines)")
            simveto = neg & (cos >= thr) & ~veto
            veto |= simveto
            print(f"[62] rule B (cosine) vetoes {int(simveto.sum()):,} additional negatives")
        except FileNotFoundError:
            print("[62] cached embeddings not found -> skipping rule B")

    clean = lab[~veto].reset_index(drop=True)
    clean.to_parquet(CLAUDE_CACHE_DIR / args.out_file, index=False)
    n0b, n0a = int(neg.sum()), int((clean.label == 0).sum())
    print(f"[62] negatives {n0b:,} -> {n0a:,} (dropped {n0b - n0a:,}, {100*(n0b-n0a)/n0b:.1f}%)")
    print(f"[62] rows {len(lab):,} -> {len(clean):,} | pos_rate {lab.label.mean():.3f} -> {clean.label.mean():.3f}")
    if "neg_source" in clean.columns:
        print("[62] surviving negatives by source:")
        print(clean[clean.label == 0].neg_source.value_counts())
    print(f"[62] wrote -> {CLAUDE_CACHE_DIR / args.out_file}")
    print(f"[62] NEXT: TY_TRAIN_PAIRS_FILE={args.out_file} python Claude-src/04_build_features.py "
          "&& python Claude-src/05_train.py")


if __name__ == "__main__":
    main()
