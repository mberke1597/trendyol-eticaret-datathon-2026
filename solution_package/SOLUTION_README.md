# TAKIMISMI_Trendyol2026_Solution — Çözüm Paketi

## Akışlar
```bash
bash step1.sh                                              # ortam + backbone indirme (internet)
bash step2.sh competition_data/ extra_generated_data/ models/   # veri üretimi + eğitim
bash step3.sh models/ competition_data/ output/            # inference (İNTERNETSİZ)
```

## Üretilen ek veri ↔ üretici eşlemesi
| Üretici | Çıktı |
|---|---|
| `Claude-src/01_encode_embeddings.py` | `extra_generated_data/cache/*.npy` (query/item embedding cache) |
| `Claude-src/03_negative_sampling.py` | `extra_generated_data/train_pairs_labeled.parquet` (madenlenmiş negatifler, veto'lu) |
| `NewPipeline/62_clean_negatives.py` | `extra_generated_data/train_pairs_labeled_clean.parquet` (temiz eğitim seti) |
| `NewPipeline/90_build_channels.py` | `extra_generated_data/channels_train_clean.npz`, `channels_submission.npz` |

Eğitilen modeller: `models/lgb_fold*.txt, xgb_fold*.json, cat_fold*.cbm, meta.json`
(taban üçlü GBDT) + `models/corrector_lgb.txt, corrector_meta.json` (düzeltici katman).
LORA/backbone finetune yoktur; backbone'lar (TY-ecomm-embed, turkish-tiny-bert)
step1'de HF'den indirilir ve step3 tamamen offline'dır (`HF_HUB_OFFLINE=1`).

## Donanım (HARDWARE.md'de ayrıntılı)
Geliştirme: Kaggle T4x2 / Colab L4 (16GB VRAM), 2x Intel Xeon vCPU ~4 çekirdek,
29GB RAM. Taban eğitim ~15dk GPU; düzeltici katman CPU-only ~25dk; inference ~12dk.

---

# SKOR ARTIRAN BÖLÜM — NewPipeline Düzeltici Katmanı (0.894 → 0.904)

Bu bölüm, LB'de submission-submission ölçülerek doğrulanmış katmandır. Üç yeni
dosya: `90_build_channels.py`, `91_train_corrector.py`, `92_apply_corrections.py`
(+ mevcut `62_clean_negatives.py`).

## 1. Teşhisler (neden çalışıyor)
- **Zehirli negatifler:** Madenlenmiş "negatiflerin" %29'u aslında muhtemel-alakalıydı
  (dense_ann negatiflerinin %45'i tam token kapsaması taşıyor; gerçek pozitiflerde %75).
  `62` iki kuralla (token kapsaması + pozitif-kosinüs P75 eşiği) 315k satırı düşürür;
  pozitif oranı %18.7→%24.4 olur (gerçek test yoğunluğu ~%28-31'e yaklaşır).
- **"Sıfır sorgu kesişimi" duvarı sahte:** Test sorgularının train'e ortalama top-1
  kosinüsü **0.81**; %27.6'sının ≥0.90 ikizi var. Tıklama davranışı komşular
  üzerinden transfer edilebilir → `catw/brandw/top1/cf` kanalları.
- **Davranışsal asimetri tuzağı:** Tıklama-türevi feature'lar train'de etiket
  vekiliyken testte %79 satırda yok; LOO'lu sürümler ise modele TERS öğretiliyor
  (0.698 vakası). Bu yüzden 10 kanalın tamamı train/test SİMETRİK tasarlandı;
  title-transfer (`tts`) modele sokulmaz, yalnız kural kanalı olarak kullanılır.

## 2. Kanallar — tam tıklama-grafiği madenciliği (`90` + `93` + `94`)
Toplam 24 kanal; hepsi train/test SİMETRİK, train tarafında self-excluded kNN.

**Temel 10** (`90_build_channels.py`): cos, recall, contain (leksik/embedding) ·
catw, brandw, top1 (komşu-transfer) · cf (co-click item benzerliği) ·
gcon, acon (cinsiyet/yaş çelişkisi) · qlen.

**Graf 14** (`93_graph_features.py`, sözlük: `94_mine_synonyms.py`):
- *Transfer:* catw_l1/catw_l2 (kategori yolunda seviye-bazlı kısmi eşleşme),
  color_prior/material_prior (komşuların tıkladığı renk/materyal dağılımında
  adayın payı — global renk kuralının sorgu-başına öğrenilmiş hali),
  gender_prior (soft cinsiyet dağılımı), cf2 (2-adımlı co-click genişletmesi).
- *Sorgu karakteri:* nb_entropy (komşu tıklamalarının kategori entropisi —
  dar/kaotik niyet), twin_density (≥0.90 ikiz sayısı — transfer güvenilirliği),
  nb_mean, exp_pos (komşuların pozitif sayısı → beklenen yoğunluk önseli).
- *Ürün karakteri:* item_breadth (kaç farklı sorgudan tıklandı — jenerik/niş),
  item_maxsim (adayı tıklamış en yakın sorgunun benzerliği; her iki tarafta
  aynı tanım, eski asimetrik click-sim tuzağının güvenli hali).
- *Sözlük:* recall_syn/contain_syn — 94'ün grafikten otomatik çıkardığı
  eşanlam/typo sözlüğüyle (kot=jean, tshirt=tişört, dayson=dyson) genişletilmiş
  leksik kapsama.

91/92 feature listesini npz'den DİNAMİK okur (corrector_meta.json'a yazılır);
93 koşulmazsa sistem 10 kanalla, koşulursa 24 kanalla çalışır — geriye uyumlu.

**25. kanal — CE entegrasyonu (`95`+`96`, opsiyonel GPU, `TY_USE_CE=1`):**
Takımın CE hattının en iyi checkpoint'i (bge-reranker-v2-m3 `ce_bgeattrmm43`,
tekil LB 0.88 / seed-ort. 0.89; HF: efeyol11/trendyol-eticaret-2026-models,
ayna: bvrtuu/...) `95` ile TEMİZ etiketler + tıklanan-sorgu zenginleştirmesi
(doc2query) + 94 sözlüğüyle fine-tune edilir; `96` skorunu `ce_score` kanalı
olarak ekler ve corrector otomatik alır. Gerekçe: orijinal CE'ler zehirli
FAISS negatifleriyle eğitilmişti (ölçüm: ~%45'i aslında alakalı) ve %24 pozitif
oranında kesiliyordu (ölçülen optimum %27-28). İki bağımsız aile — davranış
grafiği + derin metin etkileşimi — corrector'da birleşir, guard hattı hakemdir.
Ek olarak `92`'ye T9 eklendi: güçlü ikizi olan ve beklenen yoğunluğu yüksek
(exp_pos) ama tahmini ≤2 pozitifte kalan terimlerde corrector'ın en güvendiği
adaylarla 3'e tamamlama (min-pos'un veri-güdümlü hali).

## 3. Düzeltici model (`91_train_corrector.py`)
Temiz etiketler + 10 kanal ile tek LightGBM (seed=42, terim-bazlı %20 holdout,
val AUC ~0.981 — veto yapısı nedeniyle iyimserdir ve model tek başına submission
üretmek için DEĞİL, yalnız yüksek-güven bölgelerinde taban tahmini düzeltmek
için kullanılır).

## 4. Düzeltme zinciri (`92_apply_corrections.py`) — LB'de aşama aşama ölçüldü
| Aşama | Mekanizma | LB katkısı |
|---|---|---|
| T1 | Üçlü-kanıt flipleri: taban=1 & (cos<0.45, recall=0, catw=0) → 0; taban=0 & (contain, cos≥0.80, catw≥0.40) → 1 | +0.002 |
| T2 | Marka kanalı + tier-2 eşikler + sıfır-pozitifli terimlere top-2 rescue | +0.001 |
| T3 | Title click-transfer (birebir başlığı ≥0.90 ikiz sorgu tıklamış → 1) + aynı-başlık tutarlılığı | +0.001 |
| T4 | Item-transfer (≥0.93 ikizin tıkladığı item) + typo köprüleri + hedefli çıkarımlar | +0.001 |
| T6a/b | Corrector bölgeleri: p≤0.02 → 0 (typo-guard'lı), p≥0.97 & catw≥0.5 → 1 (terim başı ≤3) | +0.002 |
| T8 | Distilasyon-uyuşmazlığı: test-içi ikinci model (etiketsiz, seed=0) taban tahminine distile edilir; İKİ modelin birden karşı çıktığı satırlar fliplenir | +0.003 |

**Guard hattı** (her flip'ten önce, LB'de yanlış-flip vakalarından türetildi):
sayı eşleşmesi (beden/model/GB/sınıf), ders-konusu, marka (title-frekansı<2000
olan özgün markalar; jenerik "bohem/modern" tuzağı elenir), cinsiyet, char-3gram
typo kalkanı ("kolyw"→kolye korunur), Türkçe ek koruması ("farbela"→"farbelalı").

Zincir deterministiktir (tüm seed'ler sabit), internetsiz çalışır ve yalnız
`--base` (taban submission) + kanal/model dosyalarını okur.
