"""
Stage 94 — TIKLAMA GRAFİĞİNDEN SÖZLÜK MADENCİLİĞİ (eşanlam + typo).

Fikir: embedding uzayında ikiz olan (cos>=0.90) iki TRAIN sorgusu, token
kümeleri tek kelimede ayrışıyorsa ve tıkladıkları ürün kümeleri örtüşüyorsa,
ayrışan kelime çifti büyük olasılıkla eşanlamlıdır (kot=jean, bayan=kadın,
tshirt=tişört) ya da typo'dur (dayson=dyson). Bunu elle değil, grafikten
otomatik çıkarıyoruz.

Çıktılar (extra_data_path altına):
  synonyms_mined.csv   kolonlar: tok_a, tok_b, support, kind(synonym|typo)
93_graph_features.py bunu 'recall_syn' kanalında kullanır (sorgu token'ları
eşanlamlarla genişletilip recall yeniden hesaplanır).
"""
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, CLAUDE_CACHE_DIR

EXTRA = Path(os.environ.get("TY_EXTRA_DATA_PATH", CLAUDE_CACHE_DIR))
_SPLIT = re.compile(r"[^0-9a-zçğıöşü]+")
MIN_SUPPORT = 3
SIM_TWIN = 0.90


def norm(s):
    return str(s).replace("İ", "i").replace("I", "ı").lower()


def toks(s):
    return frozenset(t for t in _SPLIT.split(norm(s)) if len(t) >= 2)


def edit1or2(a, b):
    if abs(len(a) - len(b)) > 2:
        return False
    # hızlı yaklaşık: ortak 2-gram oranı
    A = {a[i:i + 2] for i in range(len(a) - 1)}
    B = {b[i:i + 2] for i in range(len(b) - 1)}
    return len(A & B) / max(len(A | B), 1) >= 0.5


def main():
    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    tid = pd.Index(np.load(f"{CLAUDE_CACHE_DIR}/term_id.npy", allow_pickle=True).astype(str))
    qemb = np.load(f"{CLAUDE_CACHE_DIR}/query_emb_main.npy").astype(np.float32)
    qemb /= np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-9
    tr = pd.read_csv(f"{DATA_DIR}/training_pairs.csv", usecols=["term_id", "item_id"])
    items_by = tr.groupby("term_id")["item_id"].apply(set)
    tr_terms = tr.term_id.unique()
    rows = tid.get_indexer(tr_terms)
    TRE = qemb[rows]
    tmap = dict(zip(terms.term_id, terms["query"].astype(str)))

    cand = Counter()
    K = 8
    for s in range(0, len(tr_terms), 3000):
        e = min(s + 3000, len(tr_terms))
        S = TRE[s:e] @ TRE.T
        for k in range(s, e):
            S[k - s, k] = -1.0
        nn = np.argpartition(-S, K, axis=1)[:, :K]
        nns = np.take_along_axis(S, nn, axis=1)
        for k in range(e - s):
            ta = tr_terms[s + k]
            A = toks(tmap[ta])
            for j, w in zip(nn[k], nns[k]):
                if w < SIM_TWIN:
                    continue
                tb = tr_terms[j]
                B = toks(tmap[tb])
                da, db = A - B, B - A
                if len(da) == 1 and len(db) == 1:
                    x, y = next(iter(da)), next(iter(db))
                    if x == y or len(x) < 3 or len(y) < 3:
                        continue
                    # tıklama örtüşmesi: ikizlerin ürünleri kesişiyor mu
                    ia, ib = items_by.get(ta, set()), items_by.get(tb, set())
                    if ia and ib and not (ia & ib):
                        continue
                    cand[tuple(sorted((x, y)))] += 1

    out = []
    for (a, b), sup in cand.items():
        if sup >= MIN_SUPPORT:
            out.append({"tok_a": a, "tok_b": b, "support": sup,
                        "kind": "typo" if edit1or2(a, b) else "synonym"})
    df = pd.DataFrame(out, columns=["tok_a", "tok_b", "support", "kind"])
    if len(df):
        df = df.sort_values("support", ascending=False)
    EXTRA.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXTRA / "synonyms_mined.csv", index=False)
    print(f"[94] {len(df)} sözlük girdisi -> {EXTRA/'synonyms_mined.csv'} "
          f"(synonym={int((df.kind=='synonym').sum()) if len(df) else 0}, typo={int((df.kind=='typo').sum()) if len(df) else 0})")
    if len(df):
        print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
