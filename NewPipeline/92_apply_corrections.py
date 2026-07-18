"""
Stage 92 — SKOR ARTIRAN KATMAN, adım 3/3: taban submission'ın düzeltilmesi.
İNTERNETSİZ çalışır (yalnız cache + model dosyaları okur). Deterministiktir
(tüm seed'ler sabit) — gerçek LB'de adım adım doğrulanmış zincir (her aşama
ayrı submission ile ölçüldü):

  taban (Claude-src GBDT + spec filtre)                     0.894
  T1  üçlü-kanıt flipleri (cos+recall+catw, guard'lı)      +0.002
  T2  marka-kanalı + zero-pos rescue + tier-2              +0.001
  T3  title click-transfer (>=0.90 twin) + tutarlılık      +0.001
  T4  item-transfer + typo köprüleri + hedefli çıkarımlar  +0.001
  T6a corrector p<=0.02 çıkarımları (typo-guard'lı)        +0.001
  T6b corrector p>=0.97 & catw>=0.5 eklemeleri (cap 3)     +0.001
  T8  distilasyon-uyuşmazlığı flipleri (çift model onayı)  +0.003
  TOPLAM                                                    0.904

Guard hattı (her flip'ten önce): sayı eşleşmesi (beden/model/GB), ders-konusu,
marka (title-df filtreli özgün markalar), cinsiyet, char-3gram typo kalkanı,
Türkçe ek koruması (prefix/substring).

Kullanım:
  python 92_apply_corrections.py --base <taban_submission.csv> --out <final.csv>
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, CLAUDE_CACHE_DIR

EXTRA = Path(os.environ.get("TY_EXTRA_DATA_PATH", CLAUDE_CACHE_DIR))
MODELS = Path(os.environ.get("TY_MODEL_DUMP_PATH", CLAUDE_CACHE_DIR.parent / "models"))
SEED = 0
_SPLIT = re.compile(r"[^0-9a-zçğıöşü]+")
NUM = re.compile(r"\d+")
SUBJ = {"fizik", "kimya", "biyoloji", "matematik", "türkçe", "tarih", "coğrafya",
        "edebiyat", "ingilizce", "felsefe", "geometri", "fen"}


def norm(s):
    if not isinstance(s, str):
        return ""
    return s.replace("İ", "i").replace("I", "ı").lower().replace("i̇", "i")


def c3(s):
    s = re.sub(r"\s+", " ", s).strip()
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else {s}


class Guards:
    """Sayı / ders / marka / cinsiyet / typo guard hattı (raporlardaki halleriyle)."""

    def __init__(self, sub, tmap):
        self.sub, self.tmap = sub, tmap
        bvc = pd.read_csv(f"{DATA_DIR}/items.csv", usecols=["brand"])\
            .brand.fillna("").astype(str).str.lower().value_counts()
        raw = {b for b, c in bvc.items() if c >= 20 and " " not in b and len(b) >= 4}
        # jenerik "marka" kelimelerini (bohem, modern...) title-frekansıyla ele
        from collections import Counter
        df = Counter()
        for ch in pd.read_csv(f"{DATA_DIR}/items.csv", usecols=["title"], chunksize=300_000):
            for t in ch.title.fillna("").astype(str):
                df.update(set(_SPLIT.split(norm(t))))
        self.bv = {b for b in raw if df.get(b, 0) < 2000}
        self.itxt, self.ittl = {}, {}

    def load_items(self, rows):
        need = set(self.sub.item_id.values[rows]) - set(self.itxt)
        if not need:
            return
        it = pd.read_csv(f"{DATA_DIR}/items.csv",
                         usecols=["item_id", "title", "category", "brand", "attributes"])
        it = it[it.item_id.isin(need)]
        for r in it.itertuples(index=False):
            self.itxt[r.item_id] = norm(f"{r.title} {r.category} {r.brand} {r.attributes}")
            self.ittl[r.item_id] = norm(str(r.title))

    def add_ok(self, k, use_brand=True):
        q = norm(self.tmap[self.sub.term_id.values[k]])
        t = self.itxt.get(self.sub.item_id.values[k], "")
        qn = set(NUM.findall(q))
        if qn and not qn <= set(NUM.findall(t)):
            return False
        qs = {w for w in SUBJ if w in q}
        ts = {w for w in SUBJ if w in t}
        if qs and ts and not qs & ts:
            return False
        if use_brand:
            qb = [w for w in _SPLIT.split(q) if len(w) >= 4 and w in self.bv]
            if qb and not all(b in t for b in qb):
                return False
        ttl = self.ittl.get(self.sub.item_id.values[k], "")
        if "kadın" in q and "erkek" in ttl and "kadın" not in t:
            return False
        if "erkek" in q and "kadın" in ttl and "erkek" not in t:
            return False
        return True

    def rem_ok(self, k, thr=0.12, substring=False):
        q = norm(self.tmap[self.sub.term_id.values[k]])
        ttl = self.ittl.get(self.sub.item_id.values[k], "")
        if substring:
            t = self.itxt.get(self.sub.item_id.values[k], "")
            for w in _SPLIT.split(q):
                if len(w) >= 4 and w in t:
                    return False
        a, b = c3(q), c3(ttl)
        return len(a & b) / max(len(a | b), 1) < thr


def title_transfer_sim(sub, ti, ii, qemb, tid, iid):
    """Item'ın BİREBİR başlığını train'de tıklamış sorgularla maks. kosinüs."""
    items = pd.read_csv(f"{DATA_DIR}/items.csv", usecols=["item_id", "title"])
    tn = items.title.fillna("").astype(str).map(norm).str.replace(r"\s+", " ", regex=True).str.strip()
    tcode, _ = pd.factorize(tn)
    ipos = pd.Series(np.arange(len(items)), index=items.item_id)
    tr = pd.read_csv(f"{DATA_DIR}/training_pairs.csv", usecols=["term_id", "item_id"])
    from collections import defaultdict
    prof = defaultdict(list)
    trow = tid.get_indexer(tr.term_id.values)
    tt = tcode[ipos.reindex(tr.item_id.values).values.astype(int)]
    for t, c in zip(trow, tt):
        prof[c].append(t)
    prof = {k: np.unique(v)[:50] for k, v in prof.items()}
    rowt = tcode[ii]
    tts = np.zeros(len(ii), np.float32)
    for k in range(len(ii)):
        rows = prof.get(rowt[k])
        if rows is not None:
            tts[k] = float(np.max(qemb[rows] @ qemb[ti[k]]))
    return tts, tcode[ii]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="taban pipeline submission csv (0.894 düzeyi)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sub = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv")
    base = pd.read_csv(args.base).set_index("id").reindex(sub.id.values).reset_index()
    y = base.prediction.values.astype(bool).copy()
    anchor0 = y.copy()

    import json
    meta = json.load(open(MODELS / "corrector_meta.json"))
    FEATS = meta["features"]                      # 91 ile birebir aynı liste/sıra
    src = EXTRA / f"channels_full_submission.npz"
    if not src.exists():
        src = EXTRA / "channels_submission.npz"
    ch = np.load(src)
    C = {f: ch[f] for f in ch.files}
    cos, recall, contain = C["cos"], C["recall"], C["contain"].astype(bool)
    catw, brandw, top1 = C["catw"], C["brandw"], C["top1"]
    X = np.column_stack([C[f] for f in FEATS]).astype(np.float32)
    print(f"[92] kanal kaynağı: {src.name} ({len(FEATS)} feature)")

    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    tmap = dict(zip(terms.term_id, terms["query"].astype(str)))
    G = Guards(sub, tmap)

    tid = pd.Index(np.load(f"{CLAUDE_CACHE_DIR}/term_id.npy", allow_pickle=True).astype(str))
    iid = pd.Index(np.load(f"{CLAUDE_CACHE_DIR}/item_id.npy", allow_pickle=True).astype(str))
    qemb = np.load(f"{CLAUDE_CACHE_DIR}/query_emb_main.npy").astype(np.float32)
    qemb /= np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-9
    ti = tid.get_indexer(sub.term_id.values)
    ii = iid.get_indexer(sub.item_id.values)

    def apply(rem, add, tag, rem_kw=None, add_kw=None):
        rem_kw = rem_kw or {}
        add_kw = add_kw or {}
        G.load_items(np.concatenate([rem, add]).astype(int) if len(rem) + len(add) else np.array([], int))
        remf = [k for k in rem if G.rem_ok(int(k), **rem_kw)]
        addf = [k for k in add if G.add_ok(int(k), **add_kw)]
        if remf:
            y[np.array(remf)] = False
        if addf:
            y[np.array(addf)] = True
        print(f"[92:{tag}] -{len(remf)} +{len(addf)} (guard blokladı {len(rem)-len(remf)}/{len(add)-len(addf)})")

    # ---- T1: üçlü-kanıt flipleri ----
    rem = np.where(y & (cos < 0.45) & (recall == 0) & (catw == 0))[0]
    add = np.where((~y) & contain & (cos >= 0.80) & (catw >= 0.40))[0]
    apply(rem, add, "T1")

    # ---- T2: tier-2 + zero-pos rescue ----
    rem = np.where(y & (cos < 0.50) & (recall == 0) & (catw <= 0.01) & (brandw <= 0.01))[0]
    add = np.where((~y) & contain & (cos >= 0.75) & (catw >= 0.30))[0]
    apply(rem, add, "T2a")
    tpos = pd.Series(y.astype(int)).groupby(sub.term_id.values).transform("sum").values
    zc = np.where((tpos == 0) & (~y) & (cos >= 0.65) & ((catw >= 0.20) | contain))[0]
    zdf = pd.DataFrame({"k": zc, "t": sub.term_id.values[zc], "c": cos[zc]})
    zc = zdf.sort_values("c", ascending=False).groupby("t").head(2)["k"].values
    apply(np.array([], int), zc, "T2b-zeropos")

    # ---- T3: title click-transfer + duplicate-title tutarlılığı ----
    tts, rowtitle = title_transfer_sim(sub, ti, ii, qemb, tid, iid)
    add = np.where((~y) & (tts >= 0.90))[0]
    apply(np.array([], int), add, "T3a-transfer")
    df = pd.DataFrame({"t": ti, "ttl": rowtitle, "y": y, "c": cos, "r": np.arange(len(y))})
    g = df.groupby(["t", "ttl"])
    size = g["y"].transform("size"); s = g["y"].transform("sum")
    mixed = (size > 1) & (s > 0) & (s < size)
    y[df.r.values[(mixed & (df.c >= 0.55) & (~df.y)).values]] = True
    y[df.r.values[(mixed & (df.c < 0.55) & (df.y)).values]] = False
    print(f"[92:T3b-consistency] karışık gruplar çözüldü ({int(mixed.sum())} satır)")

    # ---- T4: relax transfer + typo köprüsü + hedefli çıkarımlar ----
    add = np.where((~y) & (tts >= 0.86) & (tts < 0.90) & (catw >= 0.30))[0]
    apply(np.array([], int), add, "T4a")
    add = np.where((~y) & (recall == 0) & (cos >= 0.75) & (catw >= 0.30))[0]
    apply(np.array([], int), add, "T4b-typo")
    rem = np.where(y & (top1 >= 0.95) & (catw == 0) & (brandw == 0) & (recall < 0.5) & (cos < 0.60))[0]
    apply(rem, np.array([], int), "T4c", rem_kw={"substring": True})
    add = np.where((~y) & (tts >= 0.93) & (cos >= 0.60))[0]
    apply(np.array([], int), add, "T4d-itemtransfer", add_kw={"use_brand": False})

    # ---- T6: corrector zonları ----
    m = lgb.Booster(model_file=str(MODELS / "corrector_lgb.txt"))
    bi = meta["best_iteration"]
    p = np.empty(len(X), np.float32)
    for s0 in range(0, len(X), 700_000):
        e = min(s0 + 700_000, len(X))
        p[s0:e] = m.predict(X[s0:e], num_iteration=bi)
    protected = y & ~anchor0
    rem = np.where(y & (p <= 0.02) & ~protected)[0]
    apply(rem, np.array([], int), "T6a")
    add = np.where((~y) & (p >= 0.97) & (catw >= 0.50))[0]
    adf = pd.DataFrame({"k": add, "t": sub.term_id.values[add], "p": p[add]})
    add = adf.sort_values("p", ascending=False).groupby("t").head(3)["k"].values
    apply(np.array([], int), add, "T6b")

    # ---- T8: distilasyon-uyuşmazlığı (test-içi, etiketsiz, seed sabit) ----
    tts_f = tts.astype(np.float32)
    Xd = np.column_stack([X, tts_f]).astype(np.float32)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), min(1_500_000, len(y)), replace=False)
    dm = lgb.train(dict(objective="binary", learning_rate=0.1, num_leaves=127,
                        verbose=-1, seed=SEED),
                   lgb.Dataset(Xd[idx], label=y[idx].astype(np.int8)), num_boost_round=120)
    pdist = np.empty(len(Xd), np.float32)
    for s0 in range(0, len(Xd), 600_000):
        e = min(s0 + 600_000, len(Xd))
        pdist[s0:e] = dm.predict(Xd[s0:e])
    add = np.where((~y) & (pdist >= 0.85) & (p >= 0.80))[0]
    rem = np.where(y & (pdist <= 0.15) & (p <= 0.20))[0]
    apply(rem, add, "T8", rem_kw={"thr": 0.15})

    # ---- T9: terim-bazlı beklenen-yoğunluk kalibrasyonu (graf kanalları varsa) ----
    # exp_pos: komşu train sorgularının pozitif sayısı (log1p, sim-ağırlıklı).
    # Güçlü ikizi olan (twin_density>=2), beklenen yoğunluğu yüksek (exp_pos>=log1p(15))
    # ama tahmin edilen pozitifi <=2 kalan terimlerde corrector'ın en güvendiği
    # adayları 3'e tamamla — v8 sonrası ölçülen "recall açığı" düzeltmesi,
    # min-pos'un veri-güdümlü hali.
    if "exp_pos" in C and "twin_density" in C:
        expp, twin = C["exp_pos"], C["twin_density"]
        tpos = pd.Series(y.astype(int)).groupby(sub.term_id.values).transform("sum").values
        needy = (tpos <= 2) & (twin >= 2) & (expp >= np.log1p(15)) & (~y) & (p >= 0.5)
        zdf = pd.DataFrame({"k": np.where(needy)[0],
                            "t": sub.term_id.values[needy], "p": p[needy]})
        zc = zdf.sort_values("p", ascending=False).groupby("t").head(3)["k"].values
        apply(np.array([], int), zc, "T9-density")

    out = pd.DataFrame({"id": sub.id.values, "prediction": y.astype(np.int8)})
    out.to_csv(args.out, index=False)
    print(f"[92] FINAL: pozitif oranı {y.mean():.4f}, değişen satır {int((y!=anchor0).sum()):,} -> {args.out}")


if __name__ == "__main__":
    main()
