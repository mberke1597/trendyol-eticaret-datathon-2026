# Trendyol E-Ticaret Relevance — Tam Süreç Raporu

> Yarışma: query↔item alaka tahmini (0=alakasız, 1=alakalı).
> Metrik: **macro-F1 = (F1_relevant + F1_irrelevant) / 2** (global).
> Leaderboard: #1 = 0.977, #25 = 0.930, benchmark Level-5 = 0.960.
> **Bizim en iyi doğrulanmış submission: 0.894** (891 baseline → +0.003).
> Bu rapor tüm denenen yaklaşımları, hataları ve bulguları içerir.

---

## 1. VERİ VE PROBLEM

- **items.csv** — 962.873 ürün. Kolonlar: item_id, title, category (zincir,
  ort. 3.8 derinlik), brand, gender, age_group, **attributes** (anahtar:değer,
  ör. "renk: siyah, materyal: deri, desen: düz, kol boyu: uzun...").
- **terms.csv** — 50.153 benzersiz sorgu. Ort. 2.6 kelime; **%50.7'si 1-2 kelime**.
- **training_pairs.csv** — 250.000 **pozitif** (label=1) çift. Negatif yok →
  yarışmacı üretmeli.
- **submission_pairs.csv** — 3.359.679 test çifti (id, term_id, item_id).
- **15 üst kategori ailesi:** ev&mobilya %26, giyim %18, aksesuar %10,
  ayakkabı %9, elektronik %8, kozmetik %5, süpermarket %5, otomobil %4,
  bebek %2.5, banyo %2.5, hobi %2.4, kırtasiye %2.3, kitap %1.9, spor %1.7, bahçe %1.6.
- **Test yoğunluğu:** ~%28-31 pozitif (LB deneyleriyle bulundu). En iyi
  submission (891) %27.5 pozitif.

---

## 2. PIPELINE MİMARİSİ

### Claude-src/ (ana GBDT pipeline — 0.891 üreten)
| Aşama | Ne yapar |
|---|---|
| 00_finetune_embeddings | (ops) embedding fine-tune |
| 01_encode_embeddings | query/item embedding (TY-ecomm-embed) |
| 02_llm_enrichment | (ops) offline attribute extraction |
| 03_negative_sampling | hard-negatif madenleme → train_pairs_labeled (1.34M, %18.7 poz) |
| 04_build_features | 51 feature (leksik+embedding+davranış+çelişki) |
| 05_train | lgb+xgb+cat **stacking**, GroupKFold(term_id), 5-fold |
| 06_rescore_uncertain_band | (ops) belirsiz bandı LLM ile yeniden puanla |
| 07_predict | submission + eşik + gender/age hard-override |
| 08_explainability | SHAP feature importance |
| 09_hard_negative_mining | (ops) round-2 negatifler — HİÇ KOŞULMADI |
| 10_llm_relevance | (ops) Qwen ile pairwise relevance |
| hpo_search | (ops) hiperparametre — HİÇ KOŞULMADI |

**Model (models/meta.json):** stacking, meta_coef lgb=+3.27 xgb=+3.72 cat=-0.74,
eşik 0.425, **AUC 0.928, OOF macro-F1 0.807**. 51 feature; çelişki feature'ları
zaten var (color_conflict, material_conflict, size_conflict, gender/brand_contradiction).

### NewPipeline/ (bu çalışmada eklenen)
config, data · 20-22 cross-encoder · 30 embed feature · 31 LLM query ·
40-41 merge+ensemble · 50 submission · **60 attribute-çelişki filtresi (KAZANAN)** ·
61 spec_conflict feature · 70 tip-çelişki (başarısız) · **80 macro-F1 skorer** · PLAN.py

---

## 3. DENENEN YAKLAŞIMLAR (kronolojik, sonuçlarıyla)

### 3.1 Ortam/teknik engeller (çok zaman aldı)
- **Yol tespiti:** config.py `IS_KAGGLE = exists("/kaggle/input")` Colab'da yanlış
  tetikleniyordu → `TY_DATA_DIR` env ile çözüldü; NewPipeline/config'de düzeltildi
  (verinin gerçekten orada olmasını şart koştu).
- **`DATA_DIR="..."` tuzağı:** düz Python değişkeni `!python` alt-sürecine geçmez;
  `os.environ["TY_DATA_DIR"]` gerekir (defalarca tekrarlandı).
- **transformers 5.0 vs vLLM:** `Qwen2Tokenizer has no attribute
  all_special_tokens_extended` → `pip install "transformers<5.0"`.
- **`pip install vllm`** CUDA13 wheel'i çekti (`libcudart.so.13`) → Colab CUDA12;
  `vllm==0.10.1.1`'e sabitlendi.
- **sentence-transformers 4.x** torchcodec/ffmpeg hatası → `==3.4.1`.
- **hf_transfer askıda kaldı** (13M'de) → kapatıp `huggingface-cli download` +
  HF_TOKEN ile 16GB indirildi.
- **`/content` ephemeral** — CE modelleri birkaç kez silindi → **Google Drive**'a
  yönlendirildi (WORK_DIR=Drive).
- **catboost kurulu değildi** → 05/07 patladı; `pip install catboost`.

### 3.2 Çoklu-submission analizi (başlangıç)
11 eski submission (skor-isimli: 624…891) karşılaştırıldı. **726.804 satır (%21.6)**
submission'lar arası değişiyor ("kararsız"), 2.63M satır hemfikir. Amaç: pahalı
işlemleri sadece kararsız satırlarda yapmak.

### 3.3 Qwen LLM relevance (override) — **BAŞARISIZ**
Qwen2.5-3B ile her çifte P(relevant) → değişen satırları çevir.
| Deneme | Kural | LB |
|---|---|---|
| 891 baseline | — | **0.891** |
| calibrated (%24 çevirme) | Qwen~0→0 | 0.867 |
| conservative (%37) | Qwen>0.8→1,<0.2→0 | 0.822 |
**Ders:** monoton düşüş — sıfır-atış LLM, tuned ensemble'dan zayıf; override felaket.

### 3.4 Cross-encoder (tyroberta) — **BAŞARISIZ**
- 110M, 1 epoch, 5-fold OOF AUC ~0.89. Override → LB **0.86**. 4. ensemble üyesi
  (%27.5 eşik) → LB **0.861**. İkisi de base 0.891'in altında.
- **Güçlendirilmiş CE** (3 epoch, 256 token, zengin attributes): fold-0 OOF
  **macro-F1 0.755, F1_rel 0.61** → base'in (0.807) ALTINDA. Text-only,
  davranış sinyali yok → zayıf. 5-fold (~29h) koşmaya değmedi.

### 3.5 Attribute-çelişki filtresi — **KAZANAN (+0.003)**
Fikir: model "benziyor" diye alakalı diyor ama tek spec çeliş­iyor → sert 1→0.
| Versiyon | Eklenen | LB |
|---|---|---|
| 891 | — | 0.891 |
| 891_corrected | kitap konusu, lastik ebadı, sezon, kol boyu, cinsiyet (2431 flip) | **0.893** |
| v2 | + sınıf, GB kapasite, iphone modeli (3855) | **0.894** |
| v3 | + galaxy, numara, ml | 0.894 (nötr) |
| v4 | + 601 yanlış-negatif (0→1) | 0.894 (nötr) |
| v8 | + kişilik, watt (train-doğrulanmış) | 0.894 (nötr, güvenli) |
**Neden çalıştı:** model çelişki feature'larına sahip ama YUMUŞAK (düşük ağırlık);
sert kural net vakalarda modeli geçiyor.

### 3.6 Renk / materyal / tip çelişkisi — **BAŞARISIZ**
| Deneme | LB |
|---|---|
| v5 (renk çelişkisi, 2732 flip) | ~ |
| v6 (+ materyal/desen) | **0.892** (düştü!) |
| v7 (LLM tip-çelişkisi, 45.886 flip) | atılmadı — %4.98 agresif, marka felaketi |
**Neden battı:** item attributes'ındaki renk/materyal **varyant-düzeyi** — ürün o
renkte de gelebilir. v7 tip: "prada→güneş gözlüğü"nü kesti (marka sorgusunda tip serbest).

### 3.7 LLM query-understanding (v6-güvenli) — kuruldu, test edilmedi
31_llm_query_understanding.py: Trendyol-8B ile sorgu normalize/tip/eşanlam
(renk/materyal ÇIKARMADAN). 70 tip-filtresi bunu tüketiyordu ama tip-çelişkisi
battı (marka + eşleşme kırılganlığı).

---

## 4. TRENDYOL CANLI DENEYLERİ (Grup A-F) + TRAIN DOĞRULAMASI

Canlı Trendyol araması + gerçek train etiketleri ile 7 tez test edildi.

| Grup | Bulgu | Train doğrulaması |
|---|---|---|
| **A Uzunluk** | "mont"(1kel)=kaos (montaj/montessori substring); "kadın mont"=temiz; uzun=dar | kategori tutarlılığı 1kel 0.67→4+kel 0.91 ✔ |
| **B Marka** | nike/prada/"santa barbara..."(5kel) → %100 marka-exact, her tip | saf marka çiftinde model %99.93 doğru ✔ |
| **C Renk** | "siyah mont"→gri/beyaz/kırmızı geliyor; %40 çantada renk-sızıntısı | alakalıda renk uyuşmama %21.4 → renk-flip YANLIŞ ✔ |
| **D Belirsiz** | cat→Caterpillar, mi→Xiaomi (davranışla çözülür); avize=temiz | metin kuralı tehlikeli, tıklama çözer |
| **E Spec** | Trendyol GEVŞEK (iphone13→15, fizik→kimya gösteriyor) AMA yarışma SIKI | kitap %0, lastik %0, iphone %3 uyuşmama → flip GÜVENLİ ✔ |
| **F Eşanlam** | jean=kot, bayan=kadın birleşiyor | leksik feature'a eşanlam sözlüğü faydalı |

**KRİTİK PRENSİP (train ground-truth'ta ölçüldü):**
> Bir çelişki kuralı ekle **ANCAK** o özniteliğin ALAKALI çiftlerdeki uyuşmama
> oranı **< %5** ise. Öz-nitelikler (model/ebat/konu/beden/kapasite/cinsiyet) SIKI
> (<%5) → güvenli. Varyant nitelikleri (renk/materyal/ölçü/adet) GEVŞEK (>%15) → yasak.

**Kural doğrulama tablosu (alakalı çiftte uyuşmama %):**
GÜVENLİ: cinsiyet 0.2 · kol_boyu 0.0 · sezon 0.7 · lastik ~0 · kitap 0.0 ·
iphone 3.0 · galaxy 0.0 · numara 0.0 · GB 0.0 · **kişilik 1.2** · watt 0.0 · ml 0.0
SINIR: sinif 7.5
YASAK: **RENK 21.4** · ekran-inç 29.0 · litre 7.8 · adet 6.5 · cm 6.7

**Aile-bazlı renk (yeni bulgu):** renk giyim/elektronik'te GEVŞEK (%18) ama
ayakkabı/kozmetik/otomobil/bebek/süpermarket'te SIKI (%0-1) → **aile-kapılı renk
kuralı** denenebilir (global renk battı çünkü giyimde ateşliyordu).

---

## 5. FEATURE ÖNEMİ & MODEL İNCELEMESİ

**Feature önem (label korelasyonu):** leksik örtüşme baskın (word_recall 0.29,
stem_overlap 0.25); embedding (sim_max_title_cat 0.22, sim_title 0.21); davranışsal
(item_click_cat_rel 0.15); **attributes zayıf kullanılıyor** (sim_attr 0.05).

**Çıkarımlar:** (a) Türkçe morfoloji/lemmatizer en önemli feature'ı (word/stem_recall)
büyütür — en ucuz kazanç; (b) attributes az kullanılıyor → attribute-MATCH feature'ı
fırsat; (c) cross-encoder tek yeni sinyal (token-etkileşimi) ama tyroberta 110M zayıf.

**spec_conflict feature (61):** doğrulanmış-güvenli çelişkileri tek binary feature;
train'de ateşlediğinde **pozitif oranı %0.5** (genel %18.6) → ~37x temiz negatif
sinyal, modelin gürültülü color_conflict'inden çok daha keskin.

**Marka analizi:** marka güçlü daraltıcı — marka geçen alakalı çiftlerde item ~%95+
o marka ("samsung→iphone" neredeyse hiç olmuyor). Ama hard-kural riskli (çok-kelimeli
marka, alt-marka mango/mango kids, renk-marka çakışması mavi/cam) → model soft
brand_contradiction feature'ı doğru yaklaşım.

**Ölçüm altyapısı (80):** yarışma macro-F1'ini birebir kuran + eşik tarayan skorer
(sklearn ile doğrulandı). Base OOF 0.807, LB 0.891 arası fark → OOF madenlenmiş
negatiflerle %18.7 poz, test %28-31 → **CV, LB'yi yansıtmıyor (kör uçuş).**

---

## 6. NEDEN 0.93 GÖREMİYORUZ (ana teşhis)

1. **Post-processing tavanlı.** Filtre/eşik en fazla +0.003-0.005. Base modeli
   hiç iyileştirmedik — sadece çıktısını cilaladık.
2. **Base ayrımı yetersiz.** AUC 0.928 → macro-F1 ~0.89 tavan. 0.93 için AUC ~0.96
   gerekir. Bunu tek veren güçlü text-relevance çekirdeği (üst sıra muhtemelen bunu
   kullanıyor) — bizim tyroberta CE F1_rel 0.61 verip base'in altında kaldı.
3. **Kör uçuş.** CV↔LB uyuşmuyor; her fikri LB'ye atarak deniyoruz.

---

## 7. NE İŞE YARADI / NE YARAMADI (özet)

**YARADI:** train-doğrulanmış öz-nitelik çelişki filtresi (0.891→0.894).
**YARAMADI:** Qwen override (0.86), cross-encoder (0.86/0.755), renk/materyal
çelişkisi (0.892), LLM tip-çelişkisi (marka felaketi), 0→1 yanlış-negatif (nötr),
eşik kovalama (0.86'lık modelde nötr).

**Genel ders:** Model + Trendyol büyük segmentlerde (marka/renk/tip/belirsiz) zaten
iyi — sert kural oralarda zarar. Tek kazanç, modelin YUMUŞAK bıraktığı NET
öz-nitelik çelişkilerini kesmek. Ama bu +0.003'lük dar bir kazanç; 0.93 için
fundamentally daha güçlü model gerekiyor.

---

## 8. ELİMDEKİ DOSYALAR

**En iyi submission:** `submissions/891_validated_v8.csv` (0.894, train-güvenli).
**Kod (NewPipeline):** 60 (filtre, train-doğrulanmış kurallar), 61 (spec_conflict
feature), 80 (macro-F1 skorer), 20-22 (CE), 40-41 (merge/ensemble), PLAN.py (yol haritası).
**Raporlar:** arastirma_raporu.md, submission_analiz_raporu.txt,
trendyol_sorgu_deney_plani.md, bu SUREC_RAPORU.md.

---

## 9. KALAN SEÇENEKLER (öncelik sırasıyla)

1. **Ölçmeyi düzelt:** base OOF'u 80 ile ölç; test yoğunluğuna (~%29) kalibre —
   bundan sonra her şeyi ölçerek yap.
2. **Base'i büyüt:** hard-negatif round-2 (`09`) + HPO (`hpo_search`) — proven
   ensemble'a birkaç puan, ~yarım gün.
3. **Daha büyük transformer** (XLM-R large / e-ticaret büyük model) tek-fold
   prototip — F1_rel 0.61'i aşarsa 0.93 yolu açılır; aşmazsa dur.
4. **Aile-kapılı renk** + morfoloji + attribute-MATCH feature'ları (incremental).
5. **0.894'ü kabul et** — temiz, savunulabilir, açıklanabilir sonuç.

**Not:** 0.894 → 0.930 (#25) boşluğu +0.036; mevcut araç seti (GBDT + tyroberta CE)
bunu vermiyor. 0.93 gerçekçi hedefse, çekirdek model değişmeli (büyük fine-tune
transformer), ki bu ciddi GPU (L4'te 5-fold ~günler) ister.
