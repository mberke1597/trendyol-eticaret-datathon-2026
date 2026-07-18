# Kaggle'da Sıfırdan Çalıştırma — Güncel En İyi Doğrulanmış Sonuç (0.891)

Bu dosya, `Claude-src/` pipeline'ının **şu anki en iyi gerçek-LB-doğrulanmış sonucu
(0.891)** üretmek için TAM komut sırasını içerir. Son güncelleme: 2026-07-08 —
HPO'nun ve Türkçe morfolojinin gerçek LB'de İŞE YARAMADIĞI doğrulandıktan sonra
yazıldı (bkz. DESIGN.md lesson 11-12). Bu yüzden aşağıdaki "ana yol" bunları
İÇERMİYOR — ana yoldan ayrı, "opsiyonel/riskli" bölümünde duruyorlar.

`/kaggle/input/datasets/.../Claude-src/` yolunu HER SESSION'DA kendi güncel
dataset adına göre değiştir (Kaggle dataset'i her re-upload'ta yol değişebilir).

**Her `!` ile başlayan satırı ayrı bir Kaggle hücresine koy, `!` işaretini VE
`python` kelimesini ASLA atlama** — bu ikisi bu oturumda defalarca hataya yol açtı
(`SyntaxError: invalid decimal literal` / `Permission denied`).

## 0) Ortam kurulumu (HER YENİ Kaggle session'da zorunlu — session paketleri sıfırlar)

```bash
!pip install -q -r /kaggle/input/datasets/<SENIN-DATASET-YOLUN>/Claude-src/requirements.txt
```

`requirements.txt` artık koşulsuz `faiss-cpu` kuruyor (2026-07-08 düzeltmesi —
`faiss-gpu-cu12` bu Kaggle imajıyla SWIG/binary uyuşmazlığı veriyordu ve zaten
`03_negative_sampling.py` GPU FAISS kullanmıyor).

**Bu adımdan sonra Kaggle'da "Restart Session" yap, sonra devam et.** transformers
zaten import edilmişse pip install tek başına yeterli olmaz.

Restart sonrası doğrula, DEVAM ETMEDEN ÖNCE:
```python
import sentence_transformers, transformers
print(sentence_transformers.__version__, transformers.__version__)
# MUTLAKA 3.4.1  4.51.3 olmalı -- başka bir şey basıyorsa dur, devam etme
# (Alibaba-NLP/new-impl remote code transformers>=5 ile CUDA "index out of
# bounds" assert'iyle çöküyor -- bkz. DESIGN.md)
```

## 1) Embedding fine-tuning (~45 dk, opsiyonel ama 0.891 yoluna dahil)

```bash
!python /kaggle/input/datasets/<...>/Claude-src/00_finetune_embeddings.py
```

Çıktıda `fine-tuned scores` satırının `base scores`'u TÜM metriklerde geçtiğini
doğrula (geçmezse `WARNING` basar).

## 2) Embeddingleri encode et (fine-tuned checkpoint'i otomatik bulur/kullanır)

```bash
!python /kaggle/input/datasets/<...>/Claude-src/01_encode_embeddings.py
```

## 3) Negatif örnekleme + round-1 özellik çıkarımı

```bash
!python /kaggle/input/datasets/<...>/Claude-src/03_negative_sampling.py
!python /kaggle/input/datasets/<...>/Claude-src/04_build_features.py
```

`features.py` artık beden/numara eşleşmesini (`size_match`/`size_conflict`)
koşulsuz üretiyor (kod zaten pipeline'da, flag gerekmiyor). Türkçe morfoloji
(`root_overlap_n`/`root_recall`) VARSAYILAN OLARAK KAPALI kalıyor — hiçbir env
var set ETME, bu adımda hiçbir şey yapmana gerek yok, sadece `TY_USE_TURKISH_MORPHOLOGY=1`
YAZMA (gerçek LB'de 0.891→0.879 regresyona yol açtığı doğrulandı).

## 4) Round-1 eğitim (HPO KULLANMA — aşağıya bak)

```bash
!python /kaggle/input/datasets/<...>/Claude-src/05_train.py
```

**HPO/Optuna adımını ATLA.** Gerçek bir Kaggle koşusunda test edildi:
threshold=0.20'de 0.891 baseline'a karşı 0.882 verdi — HPO şu anki haliyle net
bir regresyon (bkz. DESIGN.md lesson 12). `models/best_hyperparams.json`
dosyası bu session'da hiç OLUŞMAYACAK (hpo_search.py hiç çalıştırılmadığı için),
bu yüzden `05_train.py` otomatik olarak kanıtlanmış hardcoded hiperparametreleri
kullanacak — ekstra bir şey yapmana gerek yok.

## 5) Round-1 modellerini yedekle, round-2 hard-negative mining yap

```bash
!cp -r /kaggle/working/models /kaggle/working/models_round1
!python /kaggle/input/datasets/<...>/Claude-src/09_hard_negative_mining.py
```

(Varsayılan env değişkenleriyle bu otomatik olarak `train_pairs_labeled.parquet`'i
okuyup `train_pairs_labeled_round2.parquet`'e yazar — round 2 için ekstra env var
GEREKMEZ.)

## 6) Round-2 pairs ile özellik çıkarımı + eğitim (0.891'i üreten model)

```bash
!TY_TRAIN_PAIRS_FILE=train_pairs_labeled_round2.parquet python /kaggle/input/datasets/<...>/Claude-src/04_build_features.py
!python /kaggle/input/datasets/<...>/Claude-src/05_train.py
!cp -r /kaggle/working/models /kaggle/working/models_round2
```

**`TY_TRAIN_PAIRS_FILE` ORTAM DEĞİŞKENİNİ UNUTMA** — bunu unutmak sessizce
round-1 verisine düşmeye ve 0.846 gibi bir regresyona yol açtı, gerçek bir olay.

**Round-2'yi yedeklemeyi ASLA atlama** — round-3 veya HPO denemek istersen ve
işe yaramazsa (ikisi de zaten denendi ve başarısız oldu, bkz. aşağı) buraya geri
dönebilmen gerekir.

## 7) Tahmin — doğru eşikle

```bash
!python /kaggle/input/datasets/<...>/Claude-src/07_predict.py --save-proba
!TY_THRESHOLD=0.20 python /kaggle/input/datasets/<...>/Claude-src/07_predict.py --from-cached-proba
```

`meta.json`'daki varsayılan (OOF-kalibreli) eşiği KULLANMA — her retrain'de
farklı bir değer olabilir (0.51-0.59 arası görüldü) ve gerçek LB'de çok düşük
pozitif oran (~%10) verip skoru çökertir. Her zaman `TY_THRESHOLD=0.20` ile
`--from-cached-proba` çalıştır. Çıktıda `predicted positive rate` ~%27-29
aralığında olmalı.

0.20'nin üzeri (0.22, 0.25) test edildi ve HEPSİ 0.891'den kötü çıktı (0.884,
sırasıyla) — 0.20 şu ana kadarki en iyi eşik, başka bir eşik denemene gerek yok.

## 8) Açıklanabilirlik (SHAP)

```bash
!python /kaggle/input/datasets/<...>/Claude-src/08_explainability.py
```

---

## Opsiyonel / Riskli Ek Adımlar (dikkatli kullan, hepsi ayrı test edilmeli)

### HPO/Optuna — DENENDİ, BAŞARISIZ (0.891→0.882 aynı eşikte), ÖNERİLMEZ
```bash
!pip install -q optuna
!python /kaggle/input/datasets/<...>/Claude-src/hpo_search.py
```
Bunu çalıştırırsan `models/best_hyperparams.json` oluşur ve `05_train.py` onu
OTOMATİK kullanır — round-2'yi yeniden eğitmeden önce bu dosyanın olmadığından
emin ol (`!rm -f /kaggle/working/models/best_hyperparams.json`) eğer HPO'suz
temiz baseline istiyorsan.

### Türkçe morfoloji — DENENDİ, BAŞARISIZ (0.891→0.879), ÖNERİLMEZ
```bash
!pip install -q zeyrek symspellpy
!TY_USE_TURKISH_MORPHOLOGY=1 TY_TRAIN_PAIRS_FILE=train_pairs_labeled_round2.parquet python /kaggle/input/datasets/<...>/Claude-src/04_build_features.py
```
Varsayılan zaten kapalı, bu env var'ı set ETME.

### Round-3 hard-negative mining — DENENDİ, BAŞARISIZ (0.891→0.51), KESİNLİKLE ÖNERİLMEZ
```bash
!TY_INPUT_PAIRS_FILE=train_pairs_labeled_round2.parquet TY_OUTPUT_PAIRS_FILE=train_pairs_labeled_round3.parquet python /kaggle/input/datasets/<...>/Claude-src/09_hard_negative_mining.py
```

### LLM zenginleştirme + rescore (Trendyol-LLM-Asure-12B) — kod düzeltildi, GERÇEK GPU'da henüz denenmedi
Model yükleme bug'ı düzeltildi (2026-07-08, `AutoModelForImageTextToText`).
Önce mutlaka dry-run ile plumbing'i doğrula, SONRA küçük gerçek bir test yap
(hız/rows-per-sec ölçmeden tam koşuya girme — geçmişte bu yüzden bir koşu
~0.3 satır/sn çıkıp günler sürecekti):
```bash
!python /kaggle/input/datasets/<...>/Claude-src/02_llm_enrichment.py --dry-run
!python /kaggle/input/datasets/<...>/Claude-src/06_rescore_uncertain_band.py --dry-run
```
Sonra gerçek model ile (GPU + internet gerekir):
```bash
!TY_USE_LLM_ENRICHMENT=1 python /kaggle/input/datasets/<...>/Claude-src/02_llm_enrichment.py --force
```
Çıktıdaki `rows/s` ve ETA'yı bana getir, tam koşuya değip değmeyeceğine oradan
karar veririz. Rescore'u denemek istersen (belirsiz bant, sınırlı satır sayısı):
```bash
!TY_THRESHOLD=0.20 TY_UNCERTAIN_BAND_LOW=0.10 TY_UNCERTAIN_BAND_HIGH=0.30 TY_LLM_BATCH_SIZE=4 TY_USE_RESCORE=1 TY_RESCORE_MODE=blend python /kaggle/input/datasets/<...>/Claude-src/06_rescore_uncertain_band.py --force
```
Ayrı `submission_rescored.csv` yazar, orijinal `submission.csv`'yi bozmaz.
