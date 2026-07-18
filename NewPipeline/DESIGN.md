# NewPipeline — Text-based relevance pipeline (Trendyol models)

**Amaç:** query↔item relevance'ı, üç Trendyol modelinin *metin* yeteneklerini
kullanarak öğrenilmiş bir **sınıflandırıcı** (threshold heuristiği değil) etrafında
yeniden kurmak; çıktıları hem güçlü feature'lar hem de bağımsız bir tahminci olarak
mevcut 0.891 ensemble'a eklemek.

> Durum: TASARIM. Henüz kod yok, henüz LB'de test edilmedi. Bu projenin kuralı:
> her iddia gerçek bir Kaggle submission'ıyla doğrulanana kadar "hipotez"dir.

---

## 0. Neden bu pipeline

Qwen-3B'yi *sert override* olarak denedik ve LB'de kaybettik:

| Deneme | Değişen satır kuralı | Public |
|---|---|---|
| 891 baseline | — | **0.891** |
| calibrated (%24 çevirme) | Qwen skoru ~0 → 0 | 0.867 |
| conservative (%37 çevirme) | Qwen >0.8→1, <0.2→0 | 0.822 |

Monotonik: Qwen'e ne kadar karar bıraktıysak o kadar kaybettik. **Ders:** sıfır-atış
generatif bir LLM'in ham relevance yargısı, 51-feature'lık tuned ensemble'dan zayıf.
LLM/embedding sinyalleri **override** olarak değil, **öğrenilen bir modele feature**
olarak girmeli — model onlara ne kadar güveneceğini kendisi öğrensin. Bütün bu
tasarımın ana ilkesi bu.

---

## 1. Mevcut pipeline (0.891) — kısa analiz

- **Veri:** 50.153 benzersiz query, 966.445 item (zengin `attributes` alanıyla),
  250k ham train pair → mining ile 1.34M etiketli (18.7% pozitif), 3.36M submission pair.
- **Model:** lgb + xgb + cat, 51 feature, stacking ensemble, OOF macro-F1 ≈ 0.806,
  AUC ≈ 0.928, F1-optimal threshold 0.42.
- **En güçlü feature'lar:** `stem_recall`, `sim_mean_title_cat`, `brand_click_log`,
  `sim_title`, `brand_contradiction` — yani leksik örtüşme + bi-encoder cosine + davranış.
- **Boşluklar (bizim dolduracağımız):**
  1. Semantik eşleşme yalnızca **bi-encoder cosine** (query ve item ayrı ayrı encode).
     Cross-encoder (query+item birlikte attend) IR literatüründe belirgin biçimde daha güçlü.
  2. Query anlama sığ — kısa/eşsesli/kısaltmalı query'ler normalize/expand edilmiyor.
  3. Karar bir **threshold** ile veriliyor; öğrenilmiş bir relevance classifier yok.
  4. Query'ler arası bilgi transferi yok (benzer query → benzer item davranışı).

---

## 2. Model envanteri — hangisi ne için

| Model | Boyut | Rol |
|---|---|---|
| **tyroberta** | 110M | **Cross-encoder relevance classifier** (merkez parça). query+item metnini birlikte okuyup P(relevant) üretir. 1.34M etiketle fine-tune. Ucuz inference (3.36M ~1–2 saat / T4). |
| **TY-ecomm-embed v1.2.0** | 0.3B | Bi-encoder semantic search. Query-kNN etiket transferi, paraphrase/expansion, çok-alanlı max-sim, Matryoshka (768/512/128) çok-ölçekli benzerlik. |
| **Trendyol-LLM-8B-T1** | 8B | Generatif zenginleştirme: query normalize/expand/summarize, attribute/key-value extraction, kategori tahmini. **Sadece 50k benzersiz query** üzerinde → ucuz. 8B T4'e ancak AWQ/GPTQ veya 2×T4 (TP=2) ile sığar. |

---

## 3. Yeni bileşenler

### A. tyroberta cross-encoder relevance classifier — **merkez parça**

- **Girdi:** `"[query] [SEP] [item title | category | brand | gender | age | attrs(truncated)]"`.
- **Baş:** `AutoModelForSequenceClassification`, 2 sınıf (relevant / not).
- **Eğitim:** 1.34M mined pair (pozitif + hard negative), stratified 5-fold OOF
  (Claude-src ile aynı fold'lar → sızıntısız stacking). BCE/CE loss, class weight ~4.3.
- **Çıktı:** her pair için `ce_relevance_prob` → (1) GBDT'ye feature, (2) OOF ile
  4. ensemble üyesi, (3) tek başına submission olarak da denenebilir (senin
  "threshold yerine classification" isteğin — model doğrudan argmax verir).
- **Neden bu en yüksek getiri:** bi-encoder cosine'in göremediği token-etkileşimini
  (ör. "kırmızı elbise" vs "kırmızı ayakkabı") yakalar; competition'ın kendi
  etiketleriyle eğitildiği için Qwen'in sıfır-atış zaafını taşımaz.
- **Maliyet:** fine-tune ~2–4 saat/T4; 3.36M inference ~1–2 saat/T4 (128 token, fp16).

### B. Embedding feature'ları (TY-ecomm-embed)

1. **Query→Query kNN etiket transferi** (senin "benzer query'lere benzer skor atfet"
   fikrin): 50k query'yi encode et → her query için en yakın K komşu →
   komşuların train'deki pozitif item'larıyla örtüşme / komşu-relevance ortalaması
   yeni feature'lar. Küçük query uzayı (50k) bunu çok ucuz kılar.
2. **Çok-alanlı max/mean-sim:** query vs {title, category, brand, attributes} ayrı ayrı
   → mevcut `sim_title/sim_category`'nin ötesinde `sim_attributes`, `sim_max_field`.
3. **Matryoshka çok-ölçek:** 128 ve 768 dim cosine farkı bir "kaba-vs-ince eşleşme"
   sinyali (gürültü göstergesi).
4. **Paraphrase mining / expansion:** query'nin embedding-komşusu diğer query'ler →
   genişletilmiş query metni cross-encoder ve leksik recall'ı besler.

### C. LLM zenginleştirme (8B-T1) — sadece query tarafında ucuz

- **Query normalize + expand + summarize:** 50k query → `/no_think` ile kısa yapılandırılmış
  çıktı: `{normalized, brand?, category_guess, expanded_terms[]}`. ~50k çağrı = birkaç saat.
- **Attribute/key-value extraction:** item'lar zaten `attributes` içeriyor → LLM'i
  öncelikle **query** attribute niyetini çıkarmak için kullan (ör. "40 beden siyah spor
  ayakkabı" → {beden:40, renk:siyah, tip:spor ayakkabı}), sonra item attribute'larıyla
  yapılandırılmış eşleştirme feature'ı üret. Item extraction opsiyonel (966k, pahalı,
  düşük marjinal çünkü alan zaten var).
- Çıktılar **feature ve cross-encoder girdisi** olur; asla doğrudan override değil.

### D. Position-aware nöral eşleştirme (DeepRank / PACRR) — opsiyonel ileri aşama

- Araştırılan makale (DeepRank, Pang et al. 2017): relevance **yerel ve pozisyon-duyarlı**;
  query terimlerinin doküman içinde nerede/nasıl eşleştiğini modelliyor.
- E-ticaret başlıkları kısa olduğu için saf pozisyon daha az; pratik uyarlama:
  query×title terim **etkileşim matrisi** (embedding cosine) üstünde küçük bir
  CNN/k-NRM çekirdeği → tek bir "soft match" skoru. Cross-encoder'a ek/alternatif reranker.
- **Öncelik: düşük.** İyi bir cross-encoder çoğu kazanımı zaten alır; bunu ancak
  cross-encoder platoya ulaşırsa deneriz. Yüksek kod+GPU maliyeti.

---

## 4. Nasıl birleşir

```
             ┌─ leksik feature'lar (mevcut) ─┐
query,item ──┼─ embed feature'ları (B) ──────┼─→ GBDT (lgb/xgb/cat)
             ├─ LLM query feature'ları (C) ──┤        +
             └─ cross-encoder prob (A) ──────┴─→ 4-üyeli stacking ensemble → threshold
                                                        │
                        (opsiyonel) DeepRank reranker (D) sadece belirsiz banda
```

- Hiçbiri override değil; hepsi stacking meta-öğreniciye girdi. Meta-öğrenici
  her sinyalin ağırlığını OOF'ta öğrenir — zayıf sinyal otomatik bastırılır
  (Qwen felaketinin tekrarını yapısal olarak engeller).
- "Classification" isteği iki biçimde: (i) cross-encoder tek başına argmax submission,
  (ii) ensemble olasılığını F1-optimal threshold yerine öğrenilmiş meta-classifier'ın
  kararıyla ver.

---

## 5. Compute bütçesi & öncelik

| Aşama | Model | Eğitim | Inference (3.36M) | Beklenen getiri | Öncelik |
|---|---|---|---|---|---|
| A. Cross-encoder | tyroberta 110M | 2–4 sa/T4 | 1–2 sa/T4 | **Yüksek** | 1 |
| B. Embed feature'ları | embed 0.3B | — | <1 sa (50k query + item cache) | Orta-Yüksek | 2 |
| C. LLM query enrich | 8B (2×T4/AWQ) | — | 2–4 sa (50k query) | Orta | 3 |
| D. DeepRank reranker | özel | 3–6 sa | belirsiz banda | Belirsiz | 4 (opsiyonel) |

**Tavsiye edilen sıra:** A → B → C → (gerekirse) D. Her aşama bittiğinde
LB submission ile ölç; getiri yoksa bir sonrakine geçmeden dur. A tek başına
en olası kazanç.

---

## 6. Klasör/modül planı (NewPipeline/)

```
config.py                 # yollar, model id'leri, ortak sabitler (Claude-src fold'larıyla uyumlu)
data.py                   # items/terms/pairs yükleme, metin birleştirme (title|cat|brand|attrs)
20_build_ce_dataset.py    # cross-encoder için (query,item,label) + fold atama
21_train_crossencoder.py  # tyroberta fine-tune, 5-fold OOF, checkpoint
22_score_crossencoder.py  # train OOF + submission prob → parquet (shard+resume)
30_embed_features.py      # query/item encode, kNN transfer, çok-alanlı sim → parquet
31_llm_query_enrich.py    # 8B ile query normalize/expand/attr → parquet (50k, cache)
40_merge_features.py      # A+B+C'yi Claude-src feature setine join
41_train_ensemble.py      # 4-üyeli stacking (mevcut 3 + cross-encoder), threshold/meta
07_make_submission.py     # nihai submission + varyantlar
tests/                    # her modül için küçük dry-run + sızıntı kontrolleri
```

Her script Claude-src konvansiyonlarını izler: `TY_*` env override, shard+checkpoint
resume, `--dry-run` küçük dilim, Kaggle/Colab yol oto-tespiti (ama düzeltilmiş:
veri gerçekten oradaysa Kaggle say).

---

## 7. Riskler & korkuluklar (Qwen dersinden)

- **Sızıntı:** cross-encoder ve embed-kNN feature'ları MUTLAKA OOF üretilmeli;
  aksi halde train'de şişer, LB'de çöker. Fold'lar Claude-src ile aynı.
- **Override yasak:** hiçbir metin-sinyali son kararı tek başına vermez (opsiyon i hariç,
  o da yalnız A/B testi için).
- **8B bütçesi:** T4'te fp16 sığmaz; AWQ/GPTQ veya 2×T4. Sadece 50k query'de kullan.
- **Her aşama LB ile kapıdan geçer:** getiri kanıtlanmadan bir sonrakine geçilmez.
- **Item attribute extraction'ı erken yapma:** alan zaten var, marjinal getiri düşük,
  maliyet yüksek — ancak A/B/C platoya ulaşırsa.
```
