"""
Stage 70 — product-TYPE contradiction filter, powered by LLM query understanding.

Consumes stage 31's output (LLM_QUERY_PARQUET: term_id, q_type, q_expanded).
For each pair currently predicted RELEVANT(1), it checks whether the query's
PRODUCT TYPE (intrinsic — "bot", "kablo", "pantolon") appears anywhere in the
item's title or category. If the item is not even the queried product type, the
pair is almost certainly a false positive -> flip 1->0.

Why this is the SAFE lane (unlike v5/v6):
  - Product type is INTRINSIC to the item (a cable is a cable), not a per-variant
    attribute like colour/size. Variant-attribute contradictions (v5 colour,
    v6 material) LOST on the LB because an item page spans many variants; TYPE
    does not vary.
  - Synonym safety: we accept the match if EITHER the LLM-normalized type OR any
    of the LLM's expanded synonyms is found in the item text, so "pantolon"
    vs an item titled "şalvar" is not wrongly flipped when the LLM expands
    pantolon -> {şalvar, ...}.
  - 1->0 only: can only raise precision, never explode recall.

Usage:
  python 70_type_contradiction_filter.py --in <sub.csv> --out <filtered.csv>
"""
import argparse, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from config import DATA_DIR, LLM_QUERY_PARQUET

STOP = set("ve ile için bir the li lı lu lük çok en de da modeli seti".split())


def _stem(w):
    return w[:-3] if len(w) > 6 else (w[:-2] if len(w) > 4 else w)


def _tokens(s):
    return [t for t in re.split(r"[^0-9a-zçğıöşü]+", s) if t and t not in STOP]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    if not Path(LLM_QUERY_PARQUET).exists():
        raise FileNotFoundError(f"{LLM_QUERY_PARQUET} missing -- run 31_llm_query_understanding.py first")

    sub = pd.read_csv(args.inp)
    qu = pd.read_parquet(LLM_QUERY_PARQUET)[["term_id", "q_type", "q_expanded"]]
    items = pd.read_csv(f"{DATA_DIR}/items.csv", usecols=["item_id", "title", "category"])
    pairs = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv")

    pos = sub.loc[sub.prediction == 1, "id"]
    d = pairs.merge(pos.to_frame(), on="id").merge(qu, on="term_id").merge(items, on="item_id")
    for c in ["q_type", "q_expanded", "title", "category"]:
        d[c] = d[c].fillna("").str.lower()

    itext = (d["title"] + " " + d["category"]).values
    qtype = d["q_type"].values
    qexp = d["q_expanded"].values

    flip = np.zeros(len(d), bool)
    for i in range(len(d)):
        t = qtype[i]
        if not t or len(t) < 3:
            continue
        # accept types words + synonyms; require query type head-word absent from item
        cands = _tokens(t) + _tokens(qexp[i])
        if not cands:
            continue
        text = itext[i]
        # match if any candidate (or its stem) appears as substring token in item text
        hit = any(c in text or _stem(c) in text for c in cands)
        if not hit:
            flip[i] = True

    flagged = set(d.loc[flip, "id"])
    print(f"[70] positives checked={len(d):,}  type-mismatch flipped 1->0={len(flagged):,} "
          f"({100*len(flagged)/max(len(d),1):.2f}%)")
    m = sub["id"].isin(flagged)
    sub.loc[m, "prediction"] = 0
    sub.to_csv(args.out, index=False)
    print(f"[70] wrote {args.out}  (pos {int((sub.prediction==1).sum()):,})")


if __name__ == "__main__":
    main()
