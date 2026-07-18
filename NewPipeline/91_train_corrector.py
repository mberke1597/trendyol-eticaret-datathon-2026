"""
Stage 91 — SKOR ARTIRAN KATMAN, adım 2/3: düzeltici (corrector) LightGBM eğitimi.

TEMİZ etiketler (62_clean_negatives çıktısı) + 10 simetrik kanal (stage 90) ile
tek bir LightGBM eğitir. Gerçek LB'de kanıtlanmış sonuç: bu modelin
yüksek-güven bölgeleri, taban GBDT submission'ını düzeltmek için kullanıldı
(0.899 -> 0.904 sıçramasının motoru).

Doğrulama: terim-bazlı %20 held-out (GroupShuffle mantığı, seed=42).
Val AUC ~0.981. DİKKAT: val pozitif ayrışması veto-yapısı nedeniyle iyimserdir;
bu model tek başına submission ÜRETMEZ, yalnızca taban tahmini düzeltir
(bkz. 92_apply_corrections.py).

Çıktı: <model_dump_path>/corrector_lgb.txt + corrector_meta.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLAUDE_CACHE_DIR

EXTRA = Path(os.environ.get("TY_EXTRA_DATA_PATH", CLAUDE_CACHE_DIR))
MODELS = Path(os.environ.get("TY_MODEL_DUMP_PATH", CLAUDE_CACHE_DIR.parent / "models"))
SEED = 42


def load_channels(name):
    """channels_full_* (93, tam graf seti) varsa onu, yoksa channels_* (90) kullan.
    Feature listesi dosyadan DİNAMİK okunur ve meta'ya yazılır — 92 aynı listeyi
    kullanır, eğitim/inference uyumu garanti."""
    p = EXTRA / f"channels_full_{name}.npz"
    if not p.exists():
        p = EXTRA / f"channels_{name}.npz"
    ch = np.load(p)
    feats = sorted(ch.files)
    return np.column_stack([ch[f] for f in feats]).astype(np.float32), feats, p.name


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    X, FEATS, src = load_channels("train_clean")
    print(f"[91] kanal kaynağı: {src} ({len(FEATS)} feature)")
    lab_p = EXTRA / "train_pairs_labeled_clean.parquet"
    if not lab_p.exists():
        lab_p = CLAUDE_CACHE_DIR / "train_pairs_labeled_clean.parquet"
    lab = pd.read_parquet(lab_p)
    y = lab.label.values.astype(np.int8)
    assert len(y) == len(X), "kanal/etiket satır uyuşmazlığı — 90'ı temiz parquet ile koşun"

    rng = np.random.default_rng(SEED)
    uniq = pd.unique(lab.term_id.values)
    vt = set(rng.choice(uniq, int(0.2 * len(uniq)), replace=False))
    val = np.fromiter((t in vt for t in lab.term_id.values), bool, count=len(y))

    dtr = lgb.Dataset(X[~val], label=y[~val])
    dva = lgb.Dataset(X[val], label=y[val], reference=dtr)
    m = lgb.train(dict(objective="binary", learning_rate=0.08, num_leaves=63,
                       metric="auc", verbose=-1, seed=SEED),
                  dtr, num_boost_round=250, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    m.save_model(str(MODELS / "corrector_lgb.txt"))
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y[val], m.predict(X[val], num_iteration=m.best_iteration))
    meta = {"features": FEATS, "channel_source": src, "best_iteration": m.best_iteration,
            "val_auc_termwise_holdout": round(float(auc), 4), "seed": SEED,
            "importance": {f: int(v) for f, v in zip(FEATS, m.feature_importance())},
            "note": "val AUC veto-yapisi nedeniyle iyimser; model yalniz duzeltici olarak kullanilir"}
    with open(MODELS / "corrector_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[91] corrector eğitildi: val AUC {auc:.4f}, iter {m.best_iteration} -> {MODELS/'corrector_lgb.txt'}")


if __name__ == "__main__":
    main()
