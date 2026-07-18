"""
Stage 90 — SKOR ARTIRAN KATMANIN KANALLARI (0.894 -> 0.904 katmanı, adım 1/3).

Verilen (term_id, item_id) çiftleri için 10 simetrik "kanal" üretir. Simetrik =
train ve test tarafında AYNI anlamı taşır (LOO/asimetri tuzağı yok — bkz.
DETECTIVE_FINDINGS_4). Kanallar:

  cos      query<->title embedding kosinüsü (TY-ecomm-embed cache'inden)
  recall   query token'larının item metninde bulunma oranı (Türkçe normalize)
  contain  tüm query token'ları item metninde mi (0/1)
  catw     komşu-transfer: en yakın 10 TRAIN sorgusunun tıkladığı kategorilerde
           bu item'ın kategorisinin sim-ağırlıklı payı (train tarafında self hariç)
  brandw   aynısı marka için
  top1     en yakın train sorgusuna kosinüs (train tarafında self hariç)
  gcon     cinsiyet çelişkisi (query kadın/erkek vs item gender+title fallback)
  acon     yaş çelişkisi (bebek/çocuk sorgu vs yetişkin item)
  qlen     sorgu kelime sayısı
  cf       co-click CF: item embedding'inin, komşu train sorgularının TIKLADIĞI
           itemlara maksimum benzerliği (collaborative filtering kanalı)

Kullanım:
  python 90_build_channels.py --pairs train_clean   # temiz eğitim çiftleri
  python 90_build_channels.py --pairs submission    # 3.36M test çifti
Çıktı: <extra_data_path>/channels_<pairs>.npz
Girdi cache'leri (Claude-src 01_encode üretir, offline çalışır):
  query_emb_main.npy, item_title_emb.npy, term_id.npy, item_id.npy
"""
import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, CLAUDE_CACHE_DIR

EXTRA = Path(os.environ.get("TY_EXTRA_DATA_PATH", CLAUDE_CACHE_DIR))
_SPLIT = re.compile(r"[^0-9a-zçğıöşü]+")
_STOP = {"ve", "ile", "için", "bir", "the"}
K_NEIGHBORS = 10
CF_NEIGHBORS = 6
CF_POS_CAP = 48


def norm(s):
    if not isinstance(s, str):
        return ""
    return s.replace("İ", "i").replace("I", "ı").lower().replace("i̇", "i")


def toks(s, ml=2):
    return {t for t in _SPLIT.split(norm(s)) if len(t) >= ml and t not in _STOP}


def load_pairs(which):
    if which == "submission":
        df = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv")
        return df[["term_id", "item_id"]], False
    p = EXTRA / "train_pairs_labeled_clean.parquet"
    if not p.exists():
        p = CLAUDE_CACHE_DIR / "train_pairs_labeled_clean.parquet"
    df = pd.read_parquet(p)
    return df[["term_id", "item_id"]], True   # True => self-excluded kNN (leakage guard)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", choices=["train_clean", "submission"], required=True)
    args = ap.parse_args()
    pairs, exclude_self = load_pairs(args.pairs)

    tid = pd.Index(np.load(f"{CLAUDE_CACHE_DIR}/term_id.npy", allow_pickle=True).astype(str))
    iid = pd.Index(np.load(f"{CLAUDE_CACHE_DIR}/item_id.npy", allow_pickle=True).astype(str))
    qemb = np.load(f"{CLAUDE_CACHE_DIR}/query_emb_main.npy").astype(np.float32)
    qemb /= np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-9
    iemb = np.load(f"{CLAUDE_CACHE_DIR}/item_title_emb.npy", mmap_mode="r")

    ti = tid.get_indexer(pairs.term_id.values)
    ii = iid.get_indexer(pairs.item_id.values)
    n = len(pairs)

    # ---- cos ----
    cos = np.empty(n, np.float32)
    for s in range(0, n, 100_000):
        e = min(s + 100_000, n)
        B = np.asarray(iemb[ii[s:e]], dtype=np.float32)
        B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
        cos[s:e] = np.einsum("ij,ij->i", qemb[ti[s:e]], B)

    # ---- lexical: recall / contain ----
    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    qtok = {t: toks(q) for t, q in zip(terms.term_id, terms["query"])}
    need = set(pairs.item_id)
    itok = {}
    for ch in pd.read_csv(f"{DATA_DIR}/items.csv",
                          usecols=["item_id", "title", "category", "brand", "attributes"],
                          chunksize=250_000):
        ch = ch[ch.item_id.isin(need)]
        txt = (ch.title.fillna("") + " " + ch.category.fillna("") + " "
               + ch.brand.fillna("") + " " + ch.attributes.fillna(""))
        for i, t in zip(ch.item_id.values, txt.values):
            itok[i] = toks(t)
    recall = np.zeros(n, np.float32)
    contain = np.zeros(n, bool)
    for k, (t, i) in enumerate(zip(pairs.term_id.values, pairs.item_id.values)):
        q = qtok.get(t)
        it = itok.get(i, frozenset())
        if q:
            inter = len(q & it)
            recall[k] = inter / len(q)
            contain[k] = inter == len(q)

    # ---- neighbor transfer: catw / brandw / top1  +  cf ----
    items = pd.read_csv(f"{DATA_DIR}/items.csv", usecols=["item_id", "category", "brand"])
    cat, _ = pd.factorize(items.category.fillna(""))
    brd, _ = pd.factorize(items.brand.fillna("").astype(str).str.lower())
    tr = pd.read_csv(f"{DATA_DIR}/training_pairs.csv", usecols=["term_id", "item_id"])
    tr_rows_all = tid.get_indexer(tr.term_id.values)
    tr_item_rows = iid.get_indexer(tr.item_id.values)
    cats_by, brands_by, pos_by = {}, {}, {}
    g = pd.DataFrame({"t": tr_rows_all, "c": cat[tr_item_rows],
                      "b": brd[tr_item_rows], "i": tr_item_rows}).groupby("t")
    for t, gr in g:
        cats_by[t] = set(gr.c)
        brands_by[t] = set(gr.b)
        pos_by[t] = gr.i.values[:60]
    tr_terms_rows = np.array(sorted(pos_by.keys()), dtype=np.int64)
    TRE = qemb[tr_terms_rows]
    col_of = {int(r): k for k, r in enumerate(tr_terms_rows)}

    uniq = np.unique(ti)
    K = K_NEIGHBORS + (1 if exclude_self else 0)
    top1_u = np.zeros(len(uniq), np.float32)
    cw_u, bw_u, nbpos_u = [], [], []
    for s in range(0, len(uniq), 3000):
        e = min(s + 3000, len(uniq))
        S = qemb[uniq[s:e]] @ TRE.T
        if exclude_self:
            for k in range(s, e):
                c = col_of.get(int(uniq[k]))
                if c is not None:
                    S[k - s, c] = -1.0
        nn = np.argpartition(-S, K_NEIGHBORS, axis=1)[:, :K_NEIGHBORS]
        nns = np.take_along_axis(S, nn, axis=1)
        top1_u[s:e] = nns.max(axis=1)
        for k in range(e - s):
            cw, bw, tot = defaultdict(float), defaultdict(float), 0.0
            acc = []
            order = np.argsort(-nns[k])
            for rank, j in enumerate(nn[k][order]):
                w = float(max(nns[k][order][rank], 0.0))
                tot += w
                trow = int(tr_terms_rows[j])
                for x in cats_by.get(trow, ()):
                    cw[x] += w
                for x in brands_by.get(trow, ()):
                    bw[x] += w
                if rank < CF_NEIGHBORS:
                    acc.append(pos_by.get(trow, np.empty(0, np.int64)))
            tot = tot or 1.0
            cw_u.append({x: v / tot for x, v in cw.items()})
            bw_u.append({x: v / tot for x, v in bw.items()})
            v = np.unique(np.concatenate(acc))[:CF_POS_CAP] if acc else np.empty(0, np.int64)
            nbpos_u.append(v)
    pos_of = {int(r): k for k, r in enumerate(uniq)}

    catw = np.zeros(n, np.float32)
    brandw = np.zeros(n, np.float32)
    top1 = np.zeros(n, np.float32)
    rc, rb = cat[ii], brd[ii]
    for k in range(n):
        p = pos_of[int(ti[k])]
        top1[k] = top1_u[p]
        catw[k] = cw_u[p].get(rc[k], 0.0)
        brandw[k] = bw_u[p].get(rb[k], 0.0)

    cf = np.zeros(n, np.float32)
    dfg = pd.DataFrame({"t": ti, "i": ii, "r": np.arange(n)}).groupby("t")
    for t, gr in dfg:
        P_ = nbpos_u[pos_of[int(t)]]
        if len(P_) == 0:
            continue
        C = np.asarray(iemb[P_], dtype=np.float32)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
        G = np.asarray(iemb[gr.i.values], dtype=np.float32)
        G /= np.linalg.norm(G, axis=1, keepdims=True) + 1e-9
        cf[gr.r.values] = (G @ C.T).max(axis=1)

    # ---- gcon / acon / qlen ----
    itg = pd.read_csv(f"{DATA_DIR}/items.csv", usecols=["item_id", "gender", "age_group", "title"])
    gg = itg.gender.fillna("unknown").map(norm)
    ig = np.where(gg == "kadın", 1, np.where(gg == "erkek", 2, np.where(gg == "unisex", 3, 0))).astype(np.int8)
    tl = itg.title.fillna("").map(norm)
    tf = np.where(tl.str.contains("kadın|bayan", regex=True), 1,
                  np.where(tl.str.contains("erkek"), 2, 0)).astype(np.int8)
    ig = np.where(ig == 0, tf, ig).astype(np.int8)
    aa = itg.age_group.fillna("unknown").map(norm)
    ia = np.where(aa.str.contains("bebek"), 1,
                  np.where(aa.str.contains("çocuk|genç"), 2,
                           np.where(aa == "yetişkin", 3, 0))).astype(np.int8)
    taf = np.where(tl.str.contains("bebek"), 1, np.where(tl.str.contains("çocuk"), 2, 0)).astype(np.int8)
    ia = np.where(ia == 0, taf, ia).astype(np.int8)
    q = terms["query"].fillna("").map(norm)
    qg = np.where(q.str.contains("kadın|bayan", regex=True), 1,
                  np.where(q.str.contains(r"\berkek", regex=True), 2, 0)).astype(np.int8)
    qa = np.where(q.str.contains("bebek"), 1, np.where(q.str.contains("çocuk"), 2, 0)).astype(np.int8)
    qlen_t = q.str.split().str.len().astype(np.int8).values
    tg_map = dict(zip(tid, range(len(tid))))  # term emb row already = index order
    g_q, a_q = qg[ti], qa[ti]
    g_i, a_i = ig[ii], ia[ii]
    gcon = (((g_q == 1) & (g_i == 2)) | ((g_q == 2) & (g_i == 1))).astype(np.float32)
    acon = ((((a_q == 1) | (a_q == 2)) & (a_i == 3))).astype(np.float32)
    qlen = qlen_t[ti].astype(np.float32)

    out = EXTRA / f"channels_{args.pairs}.npz"
    np.savez_compressed(out, cos=cos, recall=recall, contain=contain.astype(np.float32),
                        catw=catw, brandw=brandw, top1=top1, gcon=gcon, acon=acon,
                        qlen=qlen, cf=cf)
    print(f"[90] {args.pairs}: {n:,} satır, 10 kanal -> {out}")


if __name__ == "__main__":
    main()
