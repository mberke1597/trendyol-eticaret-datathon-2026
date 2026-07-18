"""
Stage 96 — CE SKORUNU 25. KANAL OLARAK EKLE (GPU, opsiyonel).

95'in ürettiği ce_clean modelini (yoksa hazır ce_bgeattrmm43 checkpoint'ini)
train_clean + submission çiftlerinde koşar ve `ce_score` kanalını
channels_full_*.npz dosyalarına ekler. Sonrasında:
  python 91_train_corrector.py   # feature listesi dinamik -> ce_score otomatik girer
  python 92_apply_corrections.py --base ... --out ...

İki bağımsız ailenin birleşimi budur: davranış-grafiği kanalları (90/93) +
derin metin etkileşimi (CE). Corrector hakemlik eder, guard hattı korur.

Süre: 3.36M satır, 568M model, bs=256, fp16 -> T4x2'de ~2.5-4 saat.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, CLAUDE_CACHE_DIR

EXTRA = Path(os.environ.get("TY_EXTRA_DATA_PATH", CLAUDE_CACHE_DIR))
MODELS = Path(os.environ.get("TY_MODEL_DUMP_PATH", CLAUDE_CACHE_DIR.parent / "models"))
MAX_LEN = int(os.environ.get("TY_CE_MAX_LEN", "128"))
BATCH = int(os.environ.get("TY_CE_INFER_BATCH", "256"))


def pick_model_dir():
    for name in ["ce_clean", "ce_bgeattrmm43", "ce_bge_attr"]:
        p = MODELS / name
        if p.exists():
            return p
    raise SystemExit("CE modeli yok: önce 95'i koşun ya da step1 ile checkpoint indirin")


def main():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    # build_texts'i 95'ten yeniden kullan (metin şablonu eğitim/inference AYNI olmalı)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "s95", str(Path(__file__).resolve().parent / "95_ce_finetune_clean.py"))
    s95 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s95)
    qtext, itext = s95.build_texts()

    mdir = pick_model_dir()
    print(f"[96] model: {mdir.name}")
    tok = AutoTokenizer.from_pretrained(mdir)
    model = AutoModelForSequenceClassification.from_pretrained(
        mdir, num_labels=1, torch_dtype="auto")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()

    def score(pairs_df, tag):
        A = [qtext[t] for t in pairs_df.term_id]
        B = [itext[i] for i in pairs_df.item_id]
        out = np.empty(len(A), np.float32)
        with torch.no_grad():
            for s in range(0, len(A), BATCH):
                e = min(s + BATCH, len(A))
                enc = tok(A[s:e], B[s:e], truncation=True, max_length=MAX_LEN,
                          padding=True, return_tensors="pt").to(dev)
                logit = model(**enc).logits.squeeze(-1).float()
                out[s:e] = torch.sigmoid(logit).cpu().numpy()
                if (s // BATCH) % 200 == 0:
                    print(f"  [{tag}] {e:,}/{len(A):,}")
        return out

    # train_clean
    lab_p = EXTRA / "train_pairs_labeled_clean.parquet"
    if not lab_p.exists():
        lab_p = CLAUDE_CACHE_DIR / "train_pairs_labeled_clean.parquet"
    lab = pd.read_parquet(lab_p)
    for name, pairs in [("train_clean", lab[["term_id", "item_id"]]),
                        ("submission", pd.read_csv(f"{DATA_DIR}/submission_pairs.csv")[["term_id", "item_id"]])]:
        src = EXTRA / f"channels_full_{name}.npz"
        base = np.load(src)
        assert len(base["cos"]) == len(pairs)
        ce = score(pairs, name)
        out = {f: base[f] for f in base.files}
        out["ce_score"] = ce
        np.savez_compressed(src, **out)
        print(f"[96] {name}: ce_score eklendi ({len(out)} kanal) -> {src}")
    print("[96] bitti. Sırada: 91 (corrector yeniden) + 92.")


if __name__ == "__main__":
    main()
