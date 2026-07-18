"""
Stage 93 — TIKLAMA GRAFİĞİ TAM MADENCİLİĞİ (nihai feature seti).

Stage 90'ın 10 temel kanalının ÜZERİNE, sorgu-ürün tıklama grafiğinden
çıkarılabilecek kalan her şeyi ekler ve tek dosyada birleştirir:
  channels_full_<pairs>.npz  (90'ın çıktısı + aşağıdaki 14 yeni kanal)

YENİ KANALLAR
  Transfer (komşu tıklamalarından aday hakkında):
    catw_l1, catw_l2   kategori yolunun 1./2. seviyesinde sim-ağırlıklı eşleşme
                       (leaf tutmasa da üst kategori tutuyorsa gri ton)
    color_prior        adayın rengi, komşuların tıkladığı renk dağılımında mı
    material_prior     aynısı materyal için
    gender_prior       komşu tıklamalarının cinsiyet dağılımında adayın payı
    cf2                2-adımlı co-click: komşu ürünler -> onları tıklayan başka
                       sorgular -> ONLARIN ürünleri; aday embedding'inin bu
                       genişletilmiş kümeye maks. benzerliği
  Sorgu-karakteri (transferin güvenilirliği):
    nb_entropy         komşu tıklamalarının kategori entropisi (dar/dağınık niyet)
    twin_density       cos>=0.90 train ikizi sayısı (0-10)
    nb_mean            top-10 komşu benzerliğinin ortalaması
    exp_pos            komşuların train pozitif sayısının sim-ağırlıklı ort. (log1p)
  Ürün-karakteri:
    item_breadth       adayı tıklamış FARKLI train sorgusu sayısı (log1p)
    item_maxsim        adayı tıklamış sorgulardan sorguya en yakınının benzerliği
                       (test tarafında LOO yok; train tarafında self hariç —
                       eski nbq_item_click_sim'in SİMETRİK ve güvenli hali:
                       train tarafında da top-10 komşu HARİCİ kısıtlaması yok,
                       her iki tarafta aynı tanım)
  Sözlük (94'ün çıktısıyla):
    recall_syn         eşanlam/typo sözlüğüyle genişletilmiş token recall
    contain_syn        genişletilmiş tam kapsama (0/1)

Kullanım (90'dan SONRA):
  python 93_graph_features.py --pairs train_clean
  python 93_graph_features.py --pairs submission
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
K = 10
CF2_Q_CAP = 20      # 2-hop: ara sorgu sayısı üst sınırı
CF2_I_CAP = 80      # 2-hop: genişletilmiş ürün kümesi üst sınırı
ITEM_CLICKQ_CAP = 50


def norm(s):
    if not isinstance(s, str):
        return ""
    return s.replace("İ", "i").replace("I", "ı").lower().replace("i̇", "i")


def toks(s, ml=2):
    return {t for t in _SPLIT.split(norm(s)) if len(t) >= ml and t not in _STOP}


ATTR_RE = re.compile(r"(?:^|, )(renk|materyal): ([^,]{2,30})")


def parse_attr(a):
    out = {}
    for k, v in ATTR_RE.findall(norm(a)):
        out.setdefault(k, v.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", choices=["train_clean", "submission"], required=True)
    args = ap.parse_args()
    exclude_self = args.pairs == "train_clean"

    base = np.load(EXTRA / f"channels_{args.pairs}.npz")
    if args.pairs == "submission":
        pairs = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv")[["term_id", "item_id"]]
    else:
        p = EXTRA / "train_pairs_labeled_clean.parquet"
        if not p.exists():
            p = CLAUDE_CACHE_DIR / "train_pairs_labeled_clean.parquet"
        pairs = pd.read_parquet(p)[["term_id", "item_id"]]
    n = len(pairs)
    assert len(base["cos"]) == n, "90 çıktısı ile çift sayısı uyuşmuyor"

    tid = pd.Index(np.load(f"{CLAUDE_CACHE_DIR}/term_id.npy", allow_pickle=True).astype(str))
    iid = pd.Index(np.load(f"{CLAUDE_CACHE_DIR}/item_id.npy", allow_pickle=True).astype(str))
    qemb = np.load(f"{CLAUDE_CACHE_DIR}/query_emb_main.npy").astype(np.float32)
    qemb /= np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-9
    iemb = np.load(f"{CLAUDE_CACHE_DIR}/item_title_emb.npy", mmap_mode="r")
    ti = tid.get_indexer(pairs.term_id.values)
    ii = iid.get_indexer(pairs.item_id.values)

    # ---- katalog yapıları ----
    items = pd.read_csv(f"{DATA_DIR}/items.csv",
                        usecols=["item_id", "category", "gender", "attributes"])
    catpath = items.category.fillna("").map(norm).str.split("/")
    l1, _ = pd.factorize(catpath.str[0].fillna(""))
    l2, _ = pd.factorize((catpath.str[0].fillna("") + "/" + catpath.str[1].fillna("")))
    leaf, _ = pd.factorize(items.category.fillna(""))
    gg = items.gender.fillna("unknown").map(norm)
    ig = np.where(gg == "kadın", 1, np.where(gg == "erkek", 2, np.where(gg == "unisex", 3, 0))).astype(np.int8)
    attrs = [parse_attr(a) for a in items.attributes.fillna("")]
    color = np.array([a.get("renk", "") for a in attrs], dtype=object)
    material = np.array([a.get("materyal", "") for a in attrs], dtype=object)

    # ---- tıklama grafiği ----
    tr = pd.read_csv(f"{DATA_DIR}/training_pairs.csv", usecols=["term_id", "item_id"])
    tr_trow = tid.get_indexer(tr.term_id.values)
    tr_irow = iid.get_indexer(tr.item_id.values)
    pos_by = defaultdict(list)          # term emb row -> clicked item rows
    clickers = defaultdict(list)        # item row -> clicking term rows
    for t, i in zip(tr_trow, tr_irow):
        pos_by[t].append(i)
        clickers[i].append(t)
    pos_by = {t: np.unique(v) for t, v in pos_by.items()}
    clickers = {i: np.unique(v)[:ITEM_CLICKQ_CAP] for i, v in clickers.items()}
    n_pos = {t: len(v) for t, v in pos_by.items()}
    tr_terms_rows = np.array(sorted(pos_by.keys()), dtype=np.int64)
    TRE = qemb[tr_terms_rows]
    col_of = {int(r): k for k, r in enumerate(tr_terms_rows)}

    # ---- terim başına komşu istatistikleri ----
    uniq = np.unique(ti)
    U = len(uniq)
    stats = {k: np.zeros(U, np.float32) for k in
             ["nb_entropy", "twin_density", "nb_mean", "exp_pos"]}
    l1w, l2w, colw, matw, genw, hop2 = [], [], [], [], [], []
    for s in range(0, U, 3000):
        e = min(s + 3000, U)
        S = qemb[uniq[s:e]] @ TRE.T
        if exclude_self:
            for k in range(s, e):
                c = col_of.get(int(uniq[k]))
                if c is not None:
                    S[k - s, c] = -1.0
        nn = np.argpartition(-S, K, axis=1)[:, :K]
        nns = np.take_along_axis(S, nn, axis=1)
        for k in range(e - s):
            order = np.argsort(-nns[k])
            sims = nns[k][order]
            neigh = [int(tr_terms_rows[j]) for j in nn[k][order]]
            stats["twin_density"][s + k] = float((sims >= 0.90).sum())
            stats["nb_mean"][s + k] = float(np.mean(np.maximum(sims, 0)))
            w = np.maximum(sims, 0.0)
            tot = w.sum() or 1.0
            stats["exp_pos"][s + k] = float(np.log1p((w * [n_pos.get(t, 0) for t in neigh]).sum() / tot))
            cd, d1, d2, cw_, mw_, gw_ = (defaultdict(float) for _ in range(6))
            hop_items = []
            for rank, (t, wi) in enumerate(zip(neigh, w)):
                P = pos_by.get(t, np.empty(0, np.int64))
                for i2 in P:
                    cd[leaf[i2]] += wi
                    d1[l1[i2]] += wi
                    d2[l2[i2]] += wi
                    if color[i2]:
                        cw_[color[i2]] += wi
                    if material[i2]:
                        mw_[material[i2]] += wi
                    gw_[int(ig[i2])] += wi
                if rank < 4:                       # 2-hop yalnız en yakın 4 komşudan
                    for i2 in P[:12]:
                        for t2 in clickers.get(int(i2), ())[:CF2_Q_CAP]:
                            hop_items.append(pos_by.get(int(t2), np.empty(0, np.int64))[:8])
            tot2 = sum(cd.values()) or 1.0
            probs = np.array(list(cd.values())) / tot2
            stats["nb_entropy"][s + k] = float(-(probs * np.log(probs + 1e-12)).sum())
            nrm = lambda d: {x: v / tot2 for x, v in d.items()}
            l1w.append(nrm(d1)); l2w.append(nrm(d2))
            colw.append(nrm(cw_)); matw.append(nrm(mw_)); genw.append(nrm(gw_))
            hv = np.unique(np.concatenate(hop_items))[:CF2_I_CAP] if hop_items else np.empty(0, np.int64)
            hop2.append(hv)
    pos_of = {int(r): k for k, r in enumerate(uniq)}

    # ---- satır bazına indirgeme ----
    NEW = {k: np.zeros(n, np.float32) for k in
           ["catw_l1", "catw_l2", "color_prior", "material_prior", "gender_prior",
            "cf2", "nb_entropy", "twin_density", "nb_mean", "exp_pos",
            "item_breadth", "item_maxsim", "recall_syn", "contain_syn"]}
    rl1, rl2, rleaf = l1[ii], l2[ii], leaf[ii]
    rcol, rmat, rgen = color[ii], material[ii], ig[ii]
    for k in range(n):
        p = pos_of[int(ti[k])]
        NEW["catw_l1"][k] = l1w[p].get(rl1[k], 0.0)
        NEW["catw_l2"][k] = l2w[p].get(rl2[k], 0.0)
        if rcol[k]:
            NEW["color_prior"][k] = colw[p].get(rcol[k], 0.0)
        if rmat[k]:
            NEW["material_prior"][k] = matw[p].get(rmat[k], 0.0)
        NEW["gender_prior"][k] = genw[p].get(int(rgen[k]), 0.0)
        for st in ["nb_entropy", "twin_density", "nb_mean", "exp_pos"]:
            NEW[st][k] = stats[st][p]

    # cf2: 2-hop item kümesine maks. benzerlik (terim gruplu, vektörize)
    dfg = pd.DataFrame({"t": ti, "i": ii, "r": np.arange(n)}).groupby("t")
    for t, gr in dfg:
        H = hop2[pos_of[int(t)]]
        if len(H) == 0:
            continue
        C = np.asarray(iemb[H], dtype=np.float32)
        C /= np.linalg.norm(C, axis=1, keepdims=True) + 1e-9
        Gm = np.asarray(iemb[gr.i.values], dtype=np.float32)
        Gm /= np.linalg.norm(Gm, axis=1, keepdims=True) + 1e-9
        NEW["cf2"][gr.r.values] = (Gm @ C.T).max(axis=1)

    # item_breadth + item_maxsim (simetrik: train tarafında self-term hariç)
    breadth = np.zeros(n, np.float32)
    imax = np.zeros(n, np.float32)
    for k in range(n):
        cq = clickers.get(int(ii[k]))
        if cq is None:
            continue
        if exclude_self:
            cq = cq[cq != ti[k]]
            if len(cq) == 0:
                continue
        breadth[k] = np.log1p(len(cq))
        imax[k] = float(np.max(qemb[cq] @ qemb[ti[k]]))
    NEW["item_breadth"] = breadth
    NEW["item_maxsim"] = imax

    # sözlükle genişletilmiş recall (94'ün çıktısı varsa)
    syn_p = EXTRA / "synonyms_mined.csv"
    syn = defaultdict(set)
    if syn_p.exists():
        for r in pd.read_csv(syn_p).itertuples(index=False):
            syn[r.tok_a].add(r.tok_b)
            syn[r.tok_b].add(r.tok_a)
    terms_df = pd.read_csv(f"{DATA_DIR}/terms.csv")
    qtok = {t: toks(q) for t, q in zip(terms_df.term_id, terms_df["query"])}
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
    for k, (t, i) in enumerate(zip(pairs.term_id.values, pairs.item_id.values)):
        q = qtok.get(t)
        it = itok.get(i, frozenset())
        if not q:
            continue
        hit = sum(1 for w in q if w in it or (syn[w] & it))
        NEW["recall_syn"][k] = hit / len(q)
        NEW["contain_syn"][k] = 1.0 if hit == len(q) else 0.0

    out = {f: base[f] for f in base.files}
    out.update(NEW)
    dst = EXTRA / f"channels_full_{args.pairs}.npz"
    np.savez_compressed(dst, **out)
    print(f"[93] {args.pairs}: {len(out)} kanal ({len(base.files)} temel + {len(NEW)} graf) -> {dst}")


if __name__ == "__main__":
    main()
