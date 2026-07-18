# Görev Promptu: Trendyol Arama-Alaka Modeli — Sıfırdan Yeniden Tasarım (Claude-src)

## Rolün

Sen deneyimli bir **Data Scientist / ML Developer / LLM Developer**'sın. Elindeki
görev bir Kaggle yarışması + hackathon'un ML çekirdeğini, önceki turlarda çıkarılan
gerçek bulgular ve sektör araştırması ışığında, `Claude-src/` klasörü altında
**sıfırdan** ama **kör değil, kanıta dayalı** şekilde inşa etmek. Kod yazmadan önce
aşağıdaki bağlamı oku ve her mimari kararını buradaki bulgulardan birine dayandır —
"büyük şirketler böyle yapıyor" ya da "önceki denemede şu bulundu" diyebilmelisin.

Çalışırken production kalitesinde, test edilmiş, açıklanabilir kod yaz. Kısayol
alma (örn. hard override'ı sessizce kaldırma, feature'ı hesaplayıp DataFrame'e
eklemeyi unutma, encoding'i doğrulamadan bırakma) — bunların hepsi bu projede daha
önce gerçekten yaşanmış, tespit edilmiş hatalar (aşağıda detaylı).

## Zorunlu ön-okuma (kod yazmadan önce)

Sıfırdan yazacaksın ama körü körüne değil — önce şu dosyaları oku, bağlamı özümse:

1. `../prompt.md` — orijinal mimari tasarım brief'i (yarışmanın istediği çözüm).
2. `../README.txt` — yarışma kuralları, macro-F1 tanımı, değerlendirme ağırlıkları.
3. `../DatasetDescription.txt` — veri sözlüğü.
4. `../DESIGN.md` — mevcut `src/` pipeline'ının mimarisi ve **2026-07-03 tarihli
   cross-fold popülerlik sızıntısı düzeltmesi** (bu düzeltmenin mantığını anlamadan
   popülerlik feature'larına dokunma).
5. `../Trendyol_EDA_Raporu.docx` ve `../Trendyol_Cozum_Mimarisi.docx` — EDA ve
   mimari kararların gerekçeleri.
6. `../src/` — mevcut çalışan pipeline (01-05 script'leri, `features.py`,
   `item_meta.py`, `config.py`). Bunu kopyalama, ama neyin işe yaradığını
   (GroupKFold, embedding seçimi, threshold arama mantığı) ve neyin eksik
   olduğunu (aşağıda) buradan öğren.

**Not**: `Claude-src/` bağımsız, kendi kendine yeten bir klasör olacak (kendi
`config.py`, kendi `requirements.txt`). Mevcut `src/`'e yazma/silme — o, referans
ve fallback olarak kalıyor.

## Yarışmanın teknik özeti

- Görev: arama sorgusu (`query`) ↔ ürün (`item`) ikilisinin alakalı olup
  olmadığını tahmin etmek (binary classification), **macro-F1** ile
  değerlendiriliyor (`(F1_relevant + F1_irrelevant) / 2`).
- **Kritik EDA bulgusu**: train/test arasında `term_id` (sorgu) örtüşmesi **%0**
  — tam cold-start. Item örtüşmesi %21.1. Bu yüzden `term_id` memorization işe
  yaramaz, content-based modelleme zorunlu, ve CV **mutlaka** `GroupKFold(term_id)`
  ile yapılmalı (rastgele satır bölmesi CV skorunu yapay şekilde şişirir).
- `submission_pairs.csv` bir **BM25+FAISS retriever'ın hard-candidate çıktısı**
  (term başına min/medyan ≈100 aday) — yani negatif örnekleme rastgele değil,
  bu retriever'ı taklit eden "zor negatif" mantığıyla yapılmalı.
- Skor ağırlıkları: Kaggle LB %40, Hackathon final-set %20, sunum %10, model hızı
  %10, açıklanabilirlik %10, final rapor %10. **Kaggle skoru toplamın sadece
  %40'ı** — açıklanabilirlik ve rapor bölümlerini atlama.

## Önceki turdan çıkarılan gerçek dersler (kanıtlanmış, varsayım değil)

### 1. Cross-fold popülerlik sızıntısı (bulundu ve düzeltildi)
`item_click_count`/`brand_click_count`/`category` popülerlik feature'ları
**global** (tüm `training_pairs.csv`'den) hesaplanırsa, bir item'ın A terimi için
train fold'unda, B terimi için val fold'unda pozitif olması durumunda A'nın
click'i B'nin val satırına sızar (item'ların ~%7.8'i 2+ terim için pozitif).
Çözüm: popülerlik istatistiklerini **her outer fold için, sadece o fold'un train
teriminden** yeniden hesapla, satır bazlı LOO (leave-one-out) düzeltmesiyle
birlikte. Bu projede synthetic-catalog unit testiyle doğrulandı — sen de benzer
bir test yaz.

### 2. Eşik (threshold) kalibrasyonu OOF'ta güvenilir değil
OOF üzerinde F1-optimal bulunan eşik (~%14 pozitif oranı) gerçek public
leaderboard'da 0.68 verdi; 7 gerçek submission sonrası gerçek optimum ~%28-31
pozitif oranında (0.83 macro-F1, platoda) bulundu. **Ders**: OOF kalibrasyonunu
tek gerçek olarak görme, density-matching eşiğini de logla, ve gerçek LB
feedback'ine göre yeniden kalibre edilebilir bir mekanizma (`env var` override
gibi) bırak. Eşik taramasını **O(n log n) tam tarama** ile yap (sıralı OOF
üzerinde kümülatif TP/FP/FN/TN) — 0.05 adımlı kaba grid (bir takım arkadaşının
scriptinde görüldü) gerçek optimumu kolayca kaçırır.

### 3. Takım arkadaşının `train_stacking.py`/`utils.py` incelemesinden çıkanlar

**Kaçınılacak hatalar:**
- Türkçe karakter haritalarında (`str.maketrans`) mojibake / çift-encode hatası
  vardı — dosya `import` anında `ValueError` ile çöküyordu (gerçekten çalıştırıp
  doğrulandı). **Her Türkçe string literalini yazarken dosyanın gerçek UTF-8
  olduğunu doğrula**, ideal olarak bir round-trip unit testiyle ("kadın",
  "çocuk", "tişört" gibi gerçek örnekleri `clean_text()`'ten geçirip beklenen
  çıktıyı aldığını kontrol et).
- Eğitim setini yapay olarak **%50-50 dengelemek** (pos=neg) gerçek test
  yoğunluğuyla (~%28-31) uyuşmuyor; eşik taraması da sadece 0.10-0.60 arasında
  yapılmıştı — daha yüksek eşikleri hiç denemedi. Negatif örnekleme, gerçek
  test yoğunluğuna yakın bir oran hedeflemeli (dengeli değil).
- Gender/age hard override'ı tamamen kaldırmak — `prompt.md`'nin **mutlak kural**
  olarak tanımladığı şeyi ihlal ediyor. Soft feature olarak modele bırakmak
  yetmez, inference'ta hard override **mutlaka** kalmalı.
- Hesaplanan ama kullanılmayan feature'lar (`brand_click_feat` DataFrame'e hiç
  eklenmemiş, sonra var olmayan bir kolon adını silmeye çalışan ölü kod) —
  feature ekleme/çıkarma adımlarını uçtan uca test et, sessiz no-op'lara izin verme.

**Benimsenecek iyi fikirler:**
- Türkçe e-ticaret domain-synonym sözlüğü (`ayakkabı↔sneaker↔bot`,
  `mont↔kaban↔ceket`, `kot↔denim`) + bigram-merge + stem-check ile genişletilmiş
  kelime örtüşmesi — mevcut `features.py`'nin basit word/stem/char-3gram
  örtüşmesinden daha zengin.
- Renk/materyal eşanlamlıları (`altın↔sarı`, `bordo↔kırmızı`, `lacivert↔mavi`)
  + collocation istisnaları ("beyaz eşya" renk sayılmamalı).
- Lojistik regresyon **meta-model stacking** (OOF tahminleri üzerinde) — grid
  search blend ağırlıklarından daha ilkeli bir yaklaşım. Ama meta-learner için
  nested/ikinci seviye bir holdout kullan (aynı OOF'ta hem fit hem skorlama
  yapıp iyimser sonuç raporlama).
- `is_generic_query` flag'i (hediye/dekor/aksesuar gibi belirsiz sorgular).
- `BrandMatcher` mantığı (stop-word filtreli, iki-kelimeli/birleşik marka
  eşleştirme) — iyi tasarlanmış ama hiç bağlanmamıştı, sen bağla.

### 4. Kategori taksonomisi — az kullanılan güçlü sinyal
`items.csv`'deki `category` alanı gerçek Trendyol site navigasyonuyla aynı
hiyerarşik yapıda (`/` ile ayrılmış, 2-6 seviye derinlik). 30K'lık pozitif
örneklemde ölçüldü: sorgu kelimeleri kategori yolunun **herhangi bir
seviyesiyle %71.8**, **en spesifik (son) seviyeyle %64.7** oranında örtüşüyor.
Şu anki pipeline'da sadece embedding-tabanlı `sim_category` var, **literal
per-level token-overlap feature'ı yok** — bunu ekle (title'daki gibi
word/stem overlap, ama kategori yolunun her seviyesi için ayrı ayrı).
Not: üst-seviye cinsiyet facet'i (Kadın/Erkek/Anne&Çocuk) `category` yolunda
değil, ayrı `gender` kolonunda — ikisini karıştırma.

### 5. Büyük şirket araştırmasından somut, uygulanabilir fikirler
(Kaynaklar: Amazon Science, Walmart/ACM Web Conf 2025, Alibaba/Taobao TaoSR1,
IKEA.com dense retrieval negative mining — tam liste için önceki sohbet
mesajındaki "Sources" bölümüne bak.)

- **LLM'i asla 3.36M satırlık hot path'e sokma** — Walmart/Amazon'un yaptığı
  gibi LLM'i sadece offline/nearline aşamalarda kullan, sonucu küçük modele
  damıt. (Amazon'un bulgusu: 110M parametrelik model 7B LLM'i %1'den az farkla,
  50x daha hızlı yakalayabiliyor.)
- **LLM ile öznitelik çıkarımı**: gürültülü `attributes` metninden
  renk/materyal/cinsiyet gibi alanları regex yerine (ya da regex'e ek olarak)
  LLM ile çıkar — sadece training+candidate setindeki item alt-kümesi üzerinde,
  tek seferlik offline pass (tüm 962K item'da değil, Kaggle GPU kotası kısıtlı).
- **Belirsiz-bant re-scoring**: OOF/test tahmini eşiğe yakın (örn. 0.35-0.65)
  olan dar bir bantta, sadece o satırları LLM veya küçük bir cross-encoder ile
  yeniden puanla — bounded compute, coarse-to-fine (Taobao/Alibaba deseni).
- **LLM üretimli hard negative**: az pozitifi olan (long-tail) terimler için
  LLM'e "bu sorguya yakın ama alakasız ürün" ürettirip negatif setini
  zenginleştir (SyNeg/IKEA deseni).
- Kullanılabilecek gerçek modeller: `Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0`
  (768d, query/title/category), `atasoglu/turkish-tiny-bert-uncased-mean-nli-stsb-tr`
  (128d, query/attributes — doğrulanmış gerçek model), `Trendyol/Trendyol-LLM-Asure-12B`
  (Gemma3-12B tabanlı, e-ticaret relevance görevleri için HF kartında açıkça
  listelenmiş — offline distillation/label üretimi için, **inference'ta değil**).

## Yeni mimarinin (Claude-src) gereksinimleri

Aşağıdaki aşamaları, numaralı script'ler halinde (`01_...py`, `02_...py`, ...)
mevcut `src/`'in run-order konvansiyonuna benzer şekilde kur:

1. **`config.py`** — Kaggle/local otomatik path algılama (`IS_KAGGLE = os.path.exists("/kaggle/input")`
   deseni), tüm sabitler, negatif örnekleme oranları (gerçek test yoğunluğuna
   yakın hedeflensin, %50-50 değil).
2. **Embedding çıkarımı** — TY-ecomm-embed (768d) + Turkish TinyBERT (128d),
   float16 cache.
3. **LLM destekli offline zenginleştirme (YENİ)** — query intent + item
   attribute çıkarımı, bounded subset üzerinde, Aşure-12B veya Kaggle'da
   sığmıyorsa daha hafif bir açık model ile. Bu aşamayı **açıkça opsiyonel/
   feature-flag'li** yap (`USE_LLM_ENRICHMENT`), çünkü Kaggle GPU kotası
   (~30 saat/hafta) kısıtlı — LLM olmadan da pipeline tam çalışmalı.
4. **Negatif örnekleme** — 4 kaynaklı mining (dense_ann/lexical/category_sibling/
   popularity_random), gerçek yoğunluğa kalibre edilmiş, opsiyonel LLM hard
   negative eklentisi long-tail terimler için.
5. **Özellik mühendisliği** — dense semantic (title/category/attr) + YENİ
   kategori-seviye literal overlap + genişletilmiş synonym/bigram lexical
   overlap + constraint (gender/age/brand, title-fallback ile) + is_generic_query
   + fold-safe (LOO düzeltmeli, asla global olmayan) popülerlik + opsiyonel LLM
   soft-relevance feature'ı.
6. **Modelleme** — `GroupKFold(term_id)`, 3 model GBDT ensemble (LightGBM CPU,
   XGBoost/CatBoost GPU-aware `device`/`task_type` ayarlı) + lojistik regresyon
   meta-model stacking (nested validasyonlu).
7. **Eşik kalibrasyonu** — O(n log n) tam tarama, density-matching eşiğini de
   logla, `TY_THRESHOLD` gibi bir env var ile gerçek LB feedback'ine göre
   yeniden kalibre edilebilsin.
8. **Belirsiz-bant re-scoring (YENİ, opsiyonel)** — eşiğe yakın dar bant için
   LLM/cross-encoder re-scoring.
9. **Inference** — chunked, hard gender/age/brand override **mutlaka kalacak**
   (prompt.md'nin mutlak kural gereksinimi), submission_pairs.csv ile id-set/
   row-count sanity assert.
10. **Açıklanabilirlik (SHAP)** — hackathon'un %10'luk açıklanabilirlik
    kriterini karşılayan bir arayüz/rapor (bu proje boyunca hiç başlanmadı,
    şimdi ele al).
11. **Test/doğrulama** — her yeni feature ve her düzeltme için (özellikle
    popülerlik sızıntısı ve encoding gibi daha önce gerçek hata çıkmış
    noktalarda) unit test yaz.

## Ortam kısıtları (Kaggle)

- GPU: **T4x2 kullan, P100 kullanma** (P100'ün sm_60 mimarisi güncel PyTorch
  ile uyumsuz — bu projede gerçek hata olarak yaşandı).
  `torch.cuda.is_available()` ile `device="cuda"`/`task_type="GPU"` ayarlarını
  koşullu yap (LightGBM GPU wheel desteklemiyor, CPU'da kalsın).
- Internet: HuggingFace modellerini indirmek için **On** olmalı.
- `requirements.txt`'te `transformers==4.51.3` pinle (embedding modelinin
  `trust_remote_code`/RoPE cache implementasyonu `transformers>=5` ile kırılıyor).
- Kod dosya olarak çalıştırılmalı (`!python script.py`), notebook hücresine
  yapıştırılmamalı — `sys.path.insert(0, str(Path(__file__).resolve().parent))`
  deseni `__file__`'a bağımlı.
- `pip install config` gibi paket adı çakışmalarına dikkat (bu proje bunu
  gerçekten yaşadı — local `config.py`'yi gölgeledi).

## Skill kullanımı beklentisi

Görev boyunca ilgili yerlerde mevcut skill'leri kullan, elle yeniden icat etme:

- Mimari kararları (örn. LLM enrichment açık/kapalı, stacking vs. blend,
  cross-encoder re-scoring maliyeti) için **`engineering:architecture`** ile
  kısa bir ADR yaz — trade-off'ları ve gerekçeyi kayıt altına al.
- Yeni feature'lar ve pipeline aşamaları için **`engineering:testing-strategy`**
  ile bir test planı çıkar (özellikle sızıntı/encoding gibi geçmişte gerçek
  hata olmuş noktalar için).
- README/runbook (Kaggle'da nasıl çalıştırılır, mevcut `KAGGLE_SETUP.md`'ye
  benzer ama yeni pipeline için) için **`engineering:documentation`**.
- Final hackathon raporu (%10'luk kriter) için **`docx`** skill'i.
- Deney/metrik takibi (fold bazlı AUC/F1, eşik tarama tablosu, ensemble
  ağırlıkları) gerekiyorsa **`xlsx`** skill'i.
- Sunum gerekiyorsa **`pptx`** skill'i.
- Kod incelemesi/PR öncesi kontrol için **`engineering:code-review`**.

Skill'i çağırmadan önce ilgili araştırma/veri toplama adımını bitir (örn. önce
gerçek deney sonuçlarını üret, sonra raporu yaz) — format skill'lerini içerik
hazır olmadan açma.

## Teslim / kabul kriterleri

- `Claude-src/` bağımsız çalışabilir olmalı (kendi `config.py`,
  `requirements.txt`, run-order script'leri).
- Popülerlik feature'larının fold-safe olduğunu doğrulayan bir unit test.
- Türkçe metin işleme fonksiyonlarının gerçek UTF-8 round-trip testi
  (`clean_text("Kadın Ayakkabı")` gibi gerçek örneklerle).
- Eşik taramasının O(n log n) tam tarama olduğunu ve density-matching eşiğinin
  de raporlandığını göster.
- Gender/age/brand hard override'ının inference'ta aktif olduğunu bir test
  satırıyla kanıtla (örn. "kadın ayakkabı" sorgusu + erkek item → prediction=0
  garantisi).
- Kısa bir özet (prose, madde işareti değil) ile: hangi kararın hangi bulguya
  dayandığını, ve mevcut `src/` pipeline'ına göre somut farkları anlat.
