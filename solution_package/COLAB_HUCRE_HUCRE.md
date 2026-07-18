# COLAB'DA HÜCRE HÜCRE ÇALIŞTIRMA (GPU: L4 veya T4 seç)

> Kural #1: env değişkenleri HER ZAMAN `os.environ` ile (bu projede `!export`
> yüzünden bir retrain boşa gitti). Kural #2: her hücrenin sonundaki
> **KONTROL** satırını görmeden sonraki hücreye geçme.

---

**HÜCRE 1 — Drive + repo + veri yolları**
```python
from google.colab import drive
drive.mount('/content/drive')

# Repo'yu Drive'a bir kez kopyalamış olmalısın (TrendyolE-Ticaret klasörünün tamamı)
ROOT = "/content/drive/MyDrive/TrendyolE-Ticaret"
COMP = f"{ROOT}/trendyol-e-ticaret-yarismasi-2026-kaggle"   # yarışma CSV'leri
import os
assert os.path.exists(f"{COMP}/submission_pairs.csv"), "veri yolu yanlış!"
print("OK:", os.listdir(ROOT)[:10])
```
KONTROL: `OK: [...]` ve assert hatası yok.

---

**HÜCRE 2 — Paketler (~3 dk)**
```python
!pip -q install "transformers==4.51.3" "sentence-transformers==3.4.1" \
  lightgbm==4.5.0 xgboost catboost faiss-cpu pyarrow scikit-learn
import lightgbm, transformers; print("OK", lightgbm.__version__, transformers.__version__)
```
KONTROL: `OK 4.5.0 4.51.3`.

---

**HÜCRE 3 — Env değişkenleri (KRİTİK HÜCRE — os.environ!)**
```python
import os
os.environ["TY_DATA_DIR"] = COMP
os.environ["TY_WORK_DIR"] = "/content/drive/MyDrive/trendyol_work"    # cache+output buraya
os.environ["TY_EXTRA_DATA_PATH"] = os.environ["TY_WORK_DIR"] + "/cache"
os.environ["TY_MODEL_DUMP_PATH"] = os.environ["TY_WORK_DIR"] + "/models"
os.environ["TY_TRAIN_PAIRS_FILE"] = "train_pairs_labeled_clean.parquet"  # temiz etiketler
os.environ["TY_NEG_VETO"] = "1"
os.environ["TY_DROP_CLICK_FEATURES"] = "1"
os.environ["TY_USE_CE"] = "1"        # CE istemiyorsan "0" yap (Hücre 11-12 atlanır)
for d in [os.environ["TY_EXTRA_DATA_PATH"], os.environ["TY_MODEL_DUMP_PATH"]]:
    os.makedirs(d, exist_ok=True)
print({k:v for k,v in os.environ.items() if k.startswith("TY_")})
```
KONTROL: yazdırılan sözlükte 7 TY_ değişkeni.

---

**HÜCRE 4 — Backbone + CE checkpoint indirme (~10 dk, internetli)**
```python
from huggingface_hub import snapshot_download
snapshot_download("Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0")
snapshot_download("atasoglu/turkish-tiny-bert-uncased-mean-nli-stsb-tr")
if os.environ["TY_USE_CE"] == "1":
    for repo in ["efeyol11/trendyol-eticaret-2026-models","bvrtuu/trendyol-eticaret-2026-models"]:
        try:
            snapshot_download(repo, allow_patterns=["ce_bgeattrmm43/*"],
                              local_dir=os.environ["TY_MODEL_DUMP_PATH"]); print("CE OK:",repo); break
        except Exception as e: print(repo,"olmadı:",e)
```
KONTROL: `CE OK: efeyol11/...` (veya bvrtuu).

---

**HÜCRE 5 — Embedding cache (01) (~40 dk GPU)**
```python
!python {ROOT}/Claude-src/01_encode_embeddings.py
import numpy as np
q = np.load(os.environ["TY_EXTRA_DATA_PATH"]+"/query_emb_main.npy")
print("KONTROL:", q.shape)   # (50153, 768) görmelisin
```

---

**HÜCRE 6 — Negatif madenciliği + temizlik (03 + 62) (~30 dk)**
```python
!python {ROOT}/Claude-src/03_negative_sampling.py
!python {ROOT}/NewPipeline/62_clean_negatives.py
```
KONTROL: 62 çıktısında `negatives 1,0XX,XXX -> ~77X,XXX (dropped ~2X%)` ve
`pos_rate 0.18x -> 0.24x` satırları. (Veto açık olduğu için sayılar rapordakinden
biraz farklı olabilir — düşen oran %15-30 bandında olmalı.)

---

**HÜCRE 7 — Taban feature'lar (04 + 12 + 13) (~40 dk)**
```python
!python {ROOT}/Claude-src/04_build_features.py
!python {ROOT}/Claude-src/12_add_group_features.py
!python {ROOT}/Claude-src/13_query_neighbor_features.py
```
KONTROL: 04 logunda **`reading labeled train pairs from train_pairs_labeled_clean.parquet`**.
Bu satır YOKSA dur — Hücre 3'ü tekrar koş.

---

**HÜCRE 8 — Taban GBDT eğitimi (05) (~1-2 saat GPU)**
```python
!python {ROOT}/Claude-src/05_train.py
```
KONTROL (üçü birden): `dropped ['item_click_log', 'item_click_cat_rel', 'brand_click_log']`
· `~1,0XX,XXX rows` · `%24.x positive`. `1,34X,XXX / 18.7%` görürsen temiz set
kullanılmamış demektir — DUR, Hücre 3+7'yi kontrol et.

---

**HÜCRE 9 — Sözlük + kanallar (94 + 90 + 93) (~1.5 saat CPU)**
```python
!python {ROOT}/NewPipeline/94_mine_synonyms.py
!python {ROOT}/NewPipeline/90_build_channels.py --pairs train_clean
!python {ROOT}/NewPipeline/90_build_channels.py --pairs submission
!python {ROOT}/NewPipeline/93_graph_features.py --pairs train_clean
!python {ROOT}/NewPipeline/93_graph_features.py --pairs submission
```
KONTROL: 94'te `N sözlük girdisi` (N>100 beklenir); 93'te iki kez `24 kanal`.

---

**HÜCRE 10 — (CE yolu) Fine-tune (95) (~2-4 saat GPU)** — `TY_USE_CE=0` ise atla
```python
!python {ROOT}/NewPipeline/95_ce_finetune_clean.py
```
KONTROL: `kaydedildi -> .../models/ce_clean` ve loss'un ~0.6'dan aşağı inmesi.

---

**HÜCRE 11 — (CE yolu) CE skorlama (96) (~2.5-4 saat GPU)** — `TY_USE_CE=0` ise atla
```python
!python {ROOT}/NewPipeline/96_add_ce_channel.py
```
KONTROL: iki kez `ce_score eklendi (25 kanal)`.

---

**HÜCRE 12 — Corrector eğitimi (91) (~10 dk)**
```python
!python {ROOT}/NewPipeline/91_train_corrector.py
import json
m = json.load(open(os.environ["TY_MODEL_DUMP_PATH"]+"/corrector_meta.json"))
print("feature sayısı:", len(m["features"]), "| val AUC:", m["val_auc_termwise_holdout"])
print("top-8 importance:", sorted(m["importance"].items(), key=lambda x:-x[1])[:8])
```
KONTROL: feature sayısı 24 (CE'li ise 25); val AUC ≥ 0.975; CE'li koşuda
`ce_score` top-8 importance'ta görünmeli.

---

**HÜCRE 13 — Taban tahmin (07) + preflight (~40 dk)**
```python
!python {ROOT}/Claude-src/07_predict.py --save-proba
!python {ROOT}/Claude-src/15_preflight_check.py
```
KONTROL: preflight **5/5 PASS**. FAIL varsa SUBMIT ETME — hangi check'in
düştüğünü söyle, ona göre teşhis koyarız. Pozitif oranı %26-28 dışındaysa
preflight'ın önerdiği `TY_THRESHOLD` ile 07'yi `--from-cached-proba` ile tekrar koş.

---

**HÜCRE 14 — Düzeltme zinciri (92) → FİNAL (~30 dk)**
```python
BASE = os.environ["TY_WORK_DIR"] + "/output/submission.csv"
OUT  = os.environ["TY_WORK_DIR"] + "/output/final_submission.csv"
!python {ROOT}/NewPipeline/92_apply_corrections.py --base {BASE} --out {OUT}
import pandas as pd
s = pd.read_csv(OUT)
print("KONTROL: satır", len(s), "| pozitif oranı", round(s.prediction.mean(),4))
```
KONTROL: satır 3,359,679 · pozitif oranı 0.26-0.28 · 92 logunda T1..T9 satırları.

---

**HÜCRE 15 — İndir**
```python
from google.colab import files
files.download(OUT)
```

## Kaggle notebook farkları
- Hücre 1 yerine: veri `/kaggle/input/...` altında; `COMP` onu göstersin,
  `TY_WORK_DIR="/kaggle/working/trendyol_work"`.
- Drive yok; uzun koşularda checkpoint'ler kaybolmasın diye önemli çıktıları
  (models/, cache/*.parquet) Kaggle Dataset'e kaydet.
- Oturum ~9-12 saatte ölür: Hücre 5-8'i bir oturumda, 9-14'ü ikinci oturumda
  koşmak güvenli (cache Drive/Dataset'te olduğu için kaldığın yerden devam eder).
