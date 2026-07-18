<div align="center">

# 🛍️ Trendyol E-Ticaret Datathon 2026
### Arama Terimi × Ürün Alaka Tahmini (Query–Product Relevance)

[![Kaggle](https://img.shields.io/badge/Kaggle-Private%20Competition-20BEFF?logo=kaggle&logoColor=white)](#)
[![Metric](https://img.shields.io/badge/Metric-Macro--F1-orange)](#-değerlendirme-metriği)
[![Final Rank](https://img.shields.io/badge/Final%20Rank-57-blue)](#-sonuç)
[![Best Rank](https://img.shields.io/badge/Peak%20Rank-34-brightgreen)](#-sonuç)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](#-tech-stack)
[![LightGBM](https://img.shields.io/badge/LightGBM-GBDT-9ACD32)](#-tech-stack)

Bir (arama terimi, ürün) çiftinin **alakalı (1)** mi **alakasız (0)** mı olduğunu tahmin eden,
uçtan uca bir arama-alaka pipeline'ı.

</div>

---

## 🏁 Sonuç

| | |
|---|---|
| 🎯 **Final sıralama** | **57.** |
| 📈 **En yüksek çıkılan sıra** | **34.** |
| 📏 **Metrik** | Macro-averaged F1 (0 ve 1 sınıfının F1'inin eşit ağırlıklı ortalaması) |
| ⏱️ **Süre** | 3 hafta |
| 🧩 **Düzeltici katmanın LB katkısı** | 0.894 → 0.904 |

---

## 📌 Problem

Trendyol arama motorunun temel sorusu: kullanıcı bir şey aradığında, gösterilecek ürün gerçekten
alakalı mı? Yarışmada bize verilen `(arama terimi, ürün)` çiftleri için binary bir alaka etiketi
(1 = relevant, 0 = irrelevant) tahmin eden bir model kurmamız istendi.

Zorluk şuydu: **eğitim verisi yalnızca pozitif (relevant) çiftlerden oluşuyordu** — negatifleri
üretmek, temizlemek ve modele doğru şekilde öğretmek tamamen bize bırakılmıştı. Test setinde ise
hem pozitif hem negatif çiftler vardı ve leaderboard **macro-F1** ile ölçülüyordu (iki sınıfın da
eşit ağırlıkla önemsendiği bir metrik — sadece çoğunluk sınıfı yakalamak yetmiyor).

---

## 🧠 Çözüm Mimarisi

İki katmanlı bir yaklaşım kurduk: **taban GBDT ensemble'ı** + LB'de aşama aşama doğrulanmış bir
**düzeltici katman**.

```
┌─────────────────────────────────────────────────────────────────────┐
│  1) EMBEDDING & VERİ HAZIRLIĞI                                       │
│     TY-ecomm-embed + turkish-tiny-bert ile query/item embedding      │
│     cache'i, FAISS tabanlı hard-negative madenciliği                 │
├─────────────────────────────────────────────────────────────────────┤
│  2) NEGATİF TEMİZLİĞİ  (zehirli negatif problemi)                    │
│     Madenlenmiş negatiflerin ~%29'u aslında alakalıydı → token       │
│     kapsaması + pozitif-kosinüs eşiğiyle temizlik (%18.7 → %24.4     │
│     pozitif oranı)                                                   │
├─────────────────────────────────────────────────────────────────────┤
│  3) TABAN MODEL — 3'lü GBDT Ensemble                                 │
│     LightGBM + XGBoost + CatBoost, grup/komşu-transfer feature'ları  │
├─────────────────────────────────────────────────────────────────────┤
│  4) DÜZELTİCİ KATMAN — Tıklama-Grafiği Madenciliği (24 kanal)        │
│     Sorgu-ürün tıklama grafiğinden komşu-transfer, co-click,         │
│     kategori/renk/materyal önseli, otomatik eşanlam & typo sözlüğü   │
├─────────────────────────────────────────────────────────────────────┤
│  5) (Opsiyonel) CROSS-ENCODER — 25. kanal                            │
│     bge-reranker tabanlı CE, temiz etiketlerle fine-tune edilip      │
│     corrector'a bir kanal olarak eklenir                             │
├─────────────────────────────────────────────────────────────────────┤
│  6) DÜZELTME ZİNCİRİ + GUARD HATTI                                   │
│     Kanıt-tabanlı flip kuralları (T1–T9), her flip'ten önce sayı/    │
│     marka/cinsiyet/typo guard'larından geçer                         │
└─────────────────────────────────────────────────────────────────────┘
```

Kilit teşhisler:
- **Sıfır sorgu kesişimi duvarı sahteydi** — test sorgularının train'e ortalama top-1 kosinüs
  benzerliği 0.81; tıklama davranışı komşu sorgular üzerinden transfer edilebiliyordu.
- **Davranışsal asimetri tuzağı** — tıklama-türevi feature'lar train'de etiket vekiliydi ama
  testte %79 satırda yoktu; bu yüzden tüm kanallar train/test simetrik tasarlandı.
- Detaylı teşhis ve LB doğrulama tabloları: [`solution_package/SOLUTION_README.md`](solution_package/SOLUTION_README.md)

---

## 🗂️ Repo Yapısı

```
TrendyolE-Ticaret/
├── Claude-src/            taban pipeline (embedding, negatif madenciliği, feature, 3'lü GBDT)
├── NewPipeline/            skor artıran düzeltici katman (kanallar, corrector, düzeltme zinciri, CE)
├── New-Pipeline/           erken deney sürümü (typo tolerance, morfoloji)
├── src/                    ilk baseline pipeline
├── solution_package/       teslim edilen çözüm paketi (step1/2/3, README, çalıştırma rehberleri)
├── DETECTIVE_FINDINGS*.md  veri kalitesi teşhis notları
├── SUREC_RAPORU.md         süreç raporu
└── DatasetDescription.txt  yarışma veri seti açıklaması
```

> Ham yarışma verisi, embedding cache, model checkpoint'leri ve submission arşivleri repoya
> dahil edilmedi (`.gitignore`) — hem boyut hem de yarışma veri paylaşım kuralları gereği.

---

## 🚀 Çalıştırma

```bash
bash solution_package/step1.sh                                                  # ortam + backbone indirme
bash solution_package/step2.sh competition_data/ extra_generated_data/ models/  # veri üretimi + eğitim
bash solution_package/step3.sh models/ competition_data/ output/                # inference (internetsiz)
```

Colab/Kaggle'da hücre hücre çalıştırma rehberi: [`solution_package/COLAB_HUCRE_HUCRE.md`](solution_package/COLAB_HUCRE_HUCRE.md)
Donanım/süre notları: [`solution_package/HARDWARE.md`](solution_package/HARDWARE.md)

---

## 📏 Değerlendirme Metriği

Macro-F1: her sınıf (0 = irrelevant, 1 = relevant) için ayrı ayrı F1 hesaplanır, ikisinin eşit
ağırlıklı ortalaması alınır. Leaderboard public subset üzerinde canlı; nihai sıralama private
subset'e göre belirlendi.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/-LightGBM-9ACD32)
![XGBoost](https://img.shields.io/badge/-XGBoost-EB0028)
![CatBoost](https://img.shields.io/badge/-CatBoost-FFCC00)
![Transformers](https://img.shields.io/badge/-🤗%20Transformers-FFD21E)
![SentenceTransformers](https://img.shields.io/badge/-Sentence--Transformers-blueviolet)
![FAISS](https://img.shields.io/badge/-FAISS-0467DF)
![Pandas](https://img.shields.io/badge/-pandas-150458?logo=pandas&logoColor=white)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white)

---

## 👥 Takım

3 kişilik bir takımla, 3 haftalık yoğun bir sürecin sonunda 57. sırada bitirdik — yol boyunca
34. sıraya kadar da yükseldik. Gecelerce süren denemeler, sürekli hipotez testi ve her adımı
gerçek leaderboard'da doğrulama disipliniyle ilerledik.

---

<div align="center">

**Trendyol E-Ticaret Datathon 2026** — verilerle yaşamayı öğrendiğimiz 3 hafta 🚀

</div>
