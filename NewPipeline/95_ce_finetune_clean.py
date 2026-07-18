"""
Stage 95 — CROSS-ENCODER'I TEMİZ ETİKETLERLE FINE-TUNE ET (GPU, opsiyonel).

Takımın CE hattının en iyi checkpoint'inden (bge-reranker-v2-m3 tabanlı
`ce_bgeattrmm43`, tekil Public LB 0.88 / seed-ort. 0.89) başlar ve bizim üç
kanıtlı iyileştirmemizi uygular:

  1. TEMİZ NEGATİFLER: orijinal FT'ler FAISS hard-negative'lerle eğitilmişti;
     ölçümümüz o negatiflerin ~%45'inin aslında alakalı olduğunu gösterdi
     (zehirli etiket tavanı). Burada 62'nin veto'lu seti kullanılır.
  2. TIKLANAN-SORGU ZENGİNLEŞTİRMESİ (doc2query): ürün metnine, aynı başlığın
     train'de tıklandığı sorgular eklenir -> davranış sinyali CE'nin metnine girer.
  3. SÖZLÜK GENİŞLETMESİ: sorguya 94'ün madenlediği eşanlamlar eklenir
     (kot=jean, tshirt=tişört) -> token örtüşmesi CE için normalize olur.

NOT: metin şablonu orijinal 'attr' modunun birebir kopyası değildir; başlangıç
AĞIRLIKLARI transfer edilir, reçete bizimdir. Girdi checkpoint step1'de HF'den
indirilir (TY_CE_HF_REPO / TY_CE_CKPT).

Çıktı: <model_dump_path>/ce_clean/  (AutoModelForSequenceClassification)
Süre: ~2-4 saat (L4/T4, 400k çift, 1 epoch, bf16).
"""
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, CLAUDE_CACHE_DIR

EXTRA = Path(os.environ.get("TY_EXTRA_DATA_PATH", CLAUDE_CACHE_DIR))
MODELS = Path(os.environ.get("TY_MODEL_DUMP_PATH", CLAUDE_CACHE_DIR.parent / "models"))
CKPT_DIR = Path(os.environ.get("TY_CE_CKPT_DIR", str(MODELS / "ce_bgeattrmm43")))
MAX_LEN = int(os.environ.get("TY_CE_MAX_LEN", "128"))
N_PAIRS = int(os.environ.get("TY_CE_N_PAIRS", "400000"))
EPOCHS = float(os.environ.get("TY_CE_EPOCHS", "1"))
BATCH = int(os.environ.get("TY_CE_BATCH", "32"))
LR = float(os.environ.get("TY_CE_LR", "1e-5"))
SEED = 42


def norm(s):
    return str(s).replace("İ", "i").replace("I", "ı").lower()


def build_texts():
    """(term_id,item_id) -> (query_text_genişletilmiş, item_text_zenginleştirilmiş)."""
    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    items = pd.read_csv(f"{DATA_DIR}/items.csv",
                        usecols=["item_id", "title", "category", "brand", "gender", "attributes"])
    # sözlük (94)
    syn = {}
    sp = EXTRA / "synonyms_mined.csv"
    if sp.exists():
        for r in pd.read_csv(sp).itertuples(index=False):
            syn.setdefault(r.tok_a, set()).add(r.tok_b)
            syn.setdefault(r.tok_b, set()).add(r.tok_a)
    qtext = {}
    for t, q in zip(terms.term_id, terms["query"].astype(str)):
        qn = norm(q)
        extra = sorted({s for w in qn.split() for s in syn.get(w, ())} - set(qn.split()))[:3]
        qtext[t] = qn + (" | " + " ".join(extra) if extra else "")
    # tıklanan-sorgu zenginleştirmesi: başlık -> train'de o başlığı tıklayan sorgular
    tr = pd.read_csv(f"{DATA_DIR}/training_pairs.csv", usecols=["term_id", "item_id"])
    tmap = dict(zip(terms.term_id, terms["query"].astype(str)))
    title_norm = items.title.fillna("").map(norm).str.replace(r"\s+", " ", regex=True).str.strip()
    t_of_item = dict(zip(items.item_id, title_norm))
    clicked = {}
    for t, i in zip(tr.term_id.values, tr.item_id.values):
        ttl = t_of_item.get(i, "")
        clicked.setdefault(ttl, [])
        if len(clicked[ttl]) < 3:
            clicked[ttl].append(norm(tmap[t]))
    itext = {}
    for r in items.itertuples(index=False):
        ttl = t_of_item.get(r.item_id, "")
        clk = clicked.get(ttl, [])
        attrs = norm(str(r.attributes))[:180]
        itext[r.item_id] = (f"{norm(str(r.title))} | {norm(str(r.category))} | "
                            f"{norm(str(r.brand))} | {norm(str(r.gender))} | {attrs}"
                            + (f" | aranan: {' ; '.join(clk)}" if clk else ""))
    return qtext, itext


def main():
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    assert CKPT_DIR.exists(), (
        f"{CKPT_DIR} yok. step1.sh CE checkpoint'ini indirir; ya da: "
        "huggingface-cli download efeyol11/trendyol-eticaret-2026-models "
        f"--include 'ce_bgeattrmm43/*' --local-dir {MODELS}")
    lab_p = EXTRA / "train_pairs_labeled_clean.parquet"
    if not lab_p.exists():
        lab_p = CLAUDE_CACHE_DIR / "train_pairs_labeled_clean.parquet"
    lab = pd.read_parquet(lab_p)
    rng = np.random.default_rng(SEED)
    if len(lab) > N_PAIRS:  # terim bütünlüğünü koru: terim örnekle
        terms_u = pd.unique(lab.term_id)
        keep = set(rng.choice(terms_u, int(len(terms_u) * N_PAIRS / len(lab)), replace=False))
        lab = lab[lab.term_id.isin(keep)].reset_index(drop=True)
    print(f"[95] eğitim çifti: {len(lab):,} (pozitif %{100*lab.label.mean():.1f})")

    qtext, itext = build_texts()
    A = [qtext[t] for t in lab.term_id]
    B = [itext[i] for i in lab.item_id]
    y = lab.label.values.astype(np.float32)

    tok = AutoTokenizer.from_pretrained(CKPT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR, num_labels=1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    if dev == "cuda":
        model = model.to(torch.bfloat16) if torch.cuda.is_bf16_supported() else model
    model.gradient_checkpointing_enable()

    idx = np.arange(len(y))
    rng.shuffle(idx)
    steps = int(len(idx) / BATCH * EPOCHS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    model.train()
    step = 0
    for ep in range(int(np.ceil(EPOCHS))):
        for s in range(0, len(idx), BATCH):
            if step >= steps:
                break
            b = idx[s:s + BATCH]
            enc = tok([A[i] for i in b], [B[i] for i in b], truncation=True,
                      max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
            out = model(**enc).logits.squeeze(-1).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out, torch.tensor(y[b], device=dev))
            loss.backward()
            opt.step(); sch.step(); opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"  step {step}/{steps} loss {loss.item():.4f}")
    out_dir = MODELS / "ce_clean"
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[95] kaydedildi -> {out_dir}")


if __name__ == "__main__":
    main()
