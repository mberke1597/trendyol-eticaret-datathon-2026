# Kaggle'da Çalıştırma Rehberi

`src/` altındaki pipeline, bu bilgisayardaki sandbox'ta pratik değil (GPU yok, 2 CPU,
uzun-süren arka plan işlemlerini desteklemiyor). `config.py` zaten `/kaggle/input`
varlığını otomatik algılayıp doğru veri/çalışma yollarını seçtiği için (`IS_KAGGLE`),
kodda hiçbir değişiklik yapmadan Kaggle'da çalıştırabilirsin.

## 1. Kod dosyalarını Kaggle'a taşı

En kolay yol — `src/`, `requirements.txt`, `DESIGN.md` klasörünü bir Kaggle **Dataset**
olarak yükle:

1. kaggle.com → **Datasets** → **New Dataset**.
2. `src/` klasörünü (tüm .py dosyalarıyla) ve `requirements.txt`'i sürükle-bırak.
3. **Private** olarak işaretle (yarışma kuralı gereği kod paylaşımı serbest değil),
   isim ver (örn. `trendyol-src`) ve yayınla (Create).

## 2. Yarışma notebook'unu aç

1. Yarışmanın Kaggle sayfasında **Data** veya **Code** sekmesi → **New Notebook**.
   Bu, yarışma verisini otomatik olarak
   `/kaggle/input/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle/` altına bağlar
   (README'deki örnek notebook'un kullandığı yol ile aynı, `config.py` bunu bekliyor).
2. Sağ panelden **Add Input** → **Your Datasets** → 1. adımda yüklediğin `trendyol-src`
   dataset'ini ekle. Artık kodun `/kaggle/input/trendyol-src/src/...` altında.

## 3. Notebook ayarları (sağ panel, ⋮ / Settings)

- **Accelerator**: GPU (P100 veya GPU T4 x2) — embedding aşaması (01) için şart,
  yoksa CPU'da çok yavaş olur.
- **Internet**: **On** — Hugging Face'ten `TY-ecomm-embed-multilingual-base-v1.2.0`
  ve `turkish-tiny-bert-...` modellerini indirmek için gerekli.
  - Eğer yarışma kuralları internet'i kapalı zorunlu kılıyorsa: modelleri önceden
    `huggingface_hub` ile indirip ayrı bir Dataset olarak yükle, sonra
    `src/config.py`'deki `MAIN_MODEL` / `TINY_MODEL` değişkenlerini o dataset'in
    yerel yoluna çevir (`01_encode_embeddings.py`'nin başındaki not bunu anlatıyor).

## 4. Kurulum + çalıştırma hücreleri

İlk hücre:
```bash
!pip install -q transformers==4.51.3 "sentence-transformers>=5,<6" faiss-gpu-cu12 lightgbm xgboost catboost
```
(Kaggle image'ında pandas/numpy/sklearn zaten kurulu; sadece pinlenmiş/eksik paketleri kur.)

Sonraki hücrelerde sırayla (her biri bitmeden diğerine geçme):
```bash
!python /kaggle/input/trendyol-src/src/01_encode_embeddings.py
!python /kaggle/input/trendyol-src/src/02_negative_sampling.py
!python /kaggle/input/trendyol-src/src/03_build_features.py
!python /kaggle/input/trendyol-src/src/04_train.py
!python /kaggle/input/trendyol-src/src/05_predict.py
```
`config.py` çıktıları otomatik olarak `/kaggle/working/cache`, `/kaggle/working/models`,
`/kaggle/working/output` altına yazar — hiçbir ortam değişkeni ayarlamana gerek yok.

## 5. Submission

Adım 5 bitince `/kaggle/working/output/submission.csv` oluşur.
- Notebook'un **Output** sekmesinden indirip yarışma sayfasından elle submit edebilirsin, veya
- Kaggle CLI ile: `kaggle competitions submit -c trendyol-e-ticaret-yarismasi-2026-kaggle -f output/submission.csv -m "ilk deneme"`

## Dikkat Edilecekler

- **Süre**: En uzun adımlar 01 (embedding, ~1M satır) ve 04 (5 fold × 3 GBDT). GPU
  oturum limiti genelde ~9-12 saat, haftalık GPU kotası ~30 saat civarı — büyük
  ihtimalle tek oturumda bitmeyecek, adımları interaktif modda (Edit) parça parça
  çalıştırıp ara çıktıları (`cache/`, `models/`) kontrol ede ede ilerlemek daha güvenli.
- **Oturum kalıcılığı**: `/kaggle/working` bir sonraki oturumda sıfırlanır. Ara
  ilerlemeyi korumak istersen notebook'u **Save & Run All (Commit)** ile versiyonla;
  commit çıktısı (cache/models) sonraki notebook'a Dataset gibi eklenebilir.
- **Gizlilik**: Yarışma "private competition" kuralları gereği notebook'u public
  yapma veya kodu Code sekmesinde paylaşma (README'de belirtilmiş).
- **04_train.py** artık popülerlik özelliklerini her fold için o fold'un train
  teriminden yeniden hesaplıyor (bkz. `DESIGN.md` — 2026-07-03 düzeltmesi); ekstra
  bir işlem gerekmiyor, otomatik çalışıyor.
