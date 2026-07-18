# SIFIRDAN ÇALIŞTIRMA REHBERİ

## 0. Neye ihtiyacın var
- Linux ortam (Colab / Kaggle / WSL2 / herhangi bir Ubuntu). GPU **şart değil**,
  sadece embedding üretimini (01) ve taban GBDT'yi hızlandırır (T4 yeterli).
- ~40GB boş disk, 16GB+ RAM.
- Yarışma verisi bir klasörde: `competition_data/` içinde items.csv, terms.csv,
  training_pairs.csv, submission_pairs.csv, sample_submission.csv.

## 1. Klasör düzeni
Repo'yu (bu klasörün tamamını) makineye kopyala. Kökten görünüm:
```
TrendyolE-Ticaret/
├── Claude-src/            # taban pipeline (01..13)
├── NewPipeline/           # düzeltici katman (62, 90, 91, 92, 93, 94)
├── solution_package/      # step1.sh, step2.sh, step3.sh
└── competition_data/      # yarışma CSV'lerini BURAYA koy (adı sen verirsin)
```

## 2. Üç komut
```bash
cd TrendyolE-Ticaret/solution_package

bash step1.sh
# ~10 dk. Conda env (ty2026) kurar, pip paketlerini ve iki backbone modeli
# (TY-ecomm-embed, turkish-tiny-bert) indirir. TEK internetli aşama.

bash step2.sh ../competition_data/ ../extra_generated_data/ ../models/
# Veri üretimi + eğitim. Sırasıyla koşan aşamalar ve yaklaşık süreler (T4 GPU / CPU):
#   01 embedding cache        ~40dk GPU (CPU'da ~4-5 saat)
#   03 negatif madenciliği    ~20dk   (TY_NEG_VETO=1 otomatik açık)
#   62 temiz etiketler        ~10dk   -> train_pairs_labeled_clean.parquet
#   04+12+13 taban feature    ~40dk
#   05 taban 3'lü GBDT        ~1-2sa GPU
#   94 sözlük madenciliği     ~5dk    -> synonyms_mined.csv
#   90+93 kanallar (24)       ~1-1.5sa CPU (train + submission)
#   91 corrector              ~5dk
# Bittiğinde: extra_generated_data/ dolu, models/ içinde taban GBDT + corrector.

bash step3.sh ../models/ ../competition_data/ ../output/
# İNTERNETSİZ inference: taban tahmin (07) + düzeltme zinciri (92, T1..T9).
# ~30-45dk. Sonuç: ../output/submission.csv
```

## 3. Doğrulama kontrol noktaları (her aşamada BUNU gör)
| Aşama | Görmen gereken satır |
|---|---|
| 62 | `negatives 1,090,200 -> ~775,000 (dropped ~29%)` ve `pos_rate 0.187 -> 0.244` |
| 04 | `reading labeled train pairs from train_pairs_labeled_clean.parquet` |
| 05 | `~1.02M rows` ve `%24.x positive` (1.34M/%18.7 görürsen temiz set KULLANILMAMIŞ) |
| 93 | `24 kanal (10 temel + 14 graf)` |
| 91 | `kanal kaynağı: channels_full_train_clean.npz (24 feature)` |
| 92 | Her T aşamasında `-X +Y (guard blokladı ...)` satırları ve son pozitif oranı 0.26-0.28 |

## 4. Hızlı duman testi (tam koşudan ÖNCE, 5 dakika)
Küçük veriyle zinciri doğrula (repo'daki data_smoke ile):
```bash
export TY_DATA_DIR=$PWD/../data_smoke TY_WORK_DIR=$PWD/../work_smoke
export TY_EXTRA_DATA_PATH=/tmp/smoke_extra TY_MODEL_DUMP_PATH=/tmp/smoke_models
mkdir -p /tmp/smoke_extra /tmp/smoke_models
python ../NewPipeline/62_clean_negatives.py --no-sim-veto
python ../NewPipeline/94_mine_synonyms.py
python ../NewPipeline/90_build_channels.py --pairs train_clean
python ../NewPipeline/90_build_channels.py --pairs submission
python ../NewPipeline/93_graph_features.py --pairs train_clean
python ../NewPipeline/93_graph_features.py --pairs submission
python ../NewPipeline/91_train_corrector.py
# hata yoksa gerçek koşuya geç
```

## 5. Bilinen tuzaklar (hepsi bu projede gerçekten yaşandı)
- **Notebook'ta `!export` ÇALIŞMAZ.** Colab/Kaggle'da env değişkenlerini
  `os.environ["TY_..."]="..."` ile AYNI hücrede set et, sonra `!bash step2.sh ...`.
  (step-scriptleri kendi env'lerini kurar; elle koşuyorsan bu kural kritik.)
- **CSV'leri asla Excel ile açma.** Excel 1.048.576 satırda sessizce keser
  (3.36M satırlık dosyaların %70'i gider).
- **Threshold'u asla eski koşudan kopyalama.** 07 çıktısındaki oran %26-28
  bandında değilse `TY_THRESHOLD` ile rate-match yap (bkz. Claude-src/15_preflight_check.py).
- **transformers<5 ve sentence-transformers==3.4.1 sabit** — 01 aşaması yeni
  sürümlerle kırılır (requirements bunu zaten pinliyor).
- step3'ü koşmadan önce step2'nin cache'i (`extra_generated_data/`) duruyor
  olmalı; step3 hiçbir şey indirmez (`HF_HUB_OFFLINE=1`).

## 6. Bir şey patlarsa
Aşamalar bağımsız script'ler — patlayan aşamayı tek başına yeniden koş
(env değişkenleri step2.sh'daki export bloğuyla aynı olsun). Her script
idempotent: çıktısını yeniden üretir, öncekini ezer.
