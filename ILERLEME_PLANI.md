# Top 20 Hedefi — İlerleme Planı

**Şu an:** 35. sıra, 0.891 (macro-F1)
**Hedef:** Top 20 — 20. sıra 0.910, 19. sıra 0.913
**Gereken fark:** ~+0.019 ile +0.022 arası

Bu küçük bir fark gibi görünüyor ama macro-F1'de son %2'lik kısım en pahalı kısımdır —
kolay kazanımlar (brand/gender override, eşik kalibrasyonu, embedding fine-tuning) zaten
alındı (0.83→0.88). Kalan fark muhtemelen birkaç orta-etkili iyileştirmenin TOPLAMından
gelecek, tek bir "sihirli değnek" değişiklikten değil.

**Altın kural (bu sezon tekrar tekrar doğrulandı):** OOF macro-F1, gerçek LB'yi güvenilir
şekilde tahmin etmiyor. Her adım gerçek bir submission ile doğrulanmadan "işe yaradı"
denemez. Round-3 hard-negative mining OOF'ta iyi görünüp gerçek LB'de 0.88→0.51'e çakıldı —
bu riski her deneyde göz önünde bulundur.

---

## Faz 0 — Ortamı sağlamlaştır (engel, puan getirmez ama şart)

Şu anki Kaggle ortamında `sentence-transformers` paketi bozuk kuruluyor (dosya
karışıklığı — bkz. son debug turu). Bu, sıfırdan retrain gerektiren HERHANGİ bir adımı
(embedding fine-tuning, encode) engelliyor.

- Bu ortamın gerçekten standart Kaggle GPU T4x2 imajı mı yoksa başka (Colab benzeri,
  bigframes/google-colab/cuml gibi paketler barındıran) bir imaj mı olduğunu kontrol et.
  Notebook'u silip yeni bir Kaggle notebook'unda GPU T4 x2 seçerek baştan denemek,
  onlarca paket çakışmasıyla uğraşmaktan daha hızlı olabilir.
- Zaten eğitilmiş round-2 modelin (0.88'i üreten) dosyaları (`models_round2/`,
  cache'lenmiş embeddingler) duruyorsa, aşağıdaki Faz 1-3'ün çoğu SIFIRDAN embedding
  fine-tuning gerektirmez — sadece `04_build_features.py` + `05_train.py` +
  `07_predict.py` çalışır durumda olması yeterli. Önce bunu doğrula, gerekirse sadece
  o üçünü çalıştıracak kadar temiz bir ortam kur.

## Faz 1 — Düşük efor, hızlı test (GPU gerektirmez, saatler değil dakikalar)

1. **DONE (kod zaten var).** Beden/numara eşleşmesi (`size_match`/`size_conflict`,
   `SIZE_VOCAB`) `features.py`'da zaten eklenmiş durumda — bu planın yazıldığı
   andan sonra bir noktada uygulanmış, aşağıdaki adımların hiçbiri bunu
   beklemiyor. Doğrulama: 2026-07-07 morfoloji koşusunun `meta.json`'ında
   `feature_cols` listesinde `size_match`/`size_conflict` mevcut.
2. **YAPILDI, KISMİ.** 0.17/0.19/0.20 taraması yapıldı (lesson #8, DESIGN.md) —
   monoton sonuç: 0.20 şimdiye kadarki en iyisi. **Henüz denenmedi: 0.20'nin
   ÜZERİ** (0.21, 0.22, 0.23, 0.25...) — lesson #8'in kendi önerisi buydu,
   hâlâ açık.
3. **YAPILDI (bilgi amaçlı).** `meta.json`: blend kazandı (`selection_holdout_macro_f1`
   0.7080 vs stacking 0.7066) — marj gerçekten küçük (0.0014). Stacking'in `C`
   parametresini oynatmak `05_train.py`'da kod değişikliği gerektiriyor — HPO/eşik
   sonuçları çıkana kadar ERTELE, bu ayrı ve daha maliyetli bir deney.
4. **YENİ, HENÜZ DENENMEDİ: HPO/Optuna.** `meta.json`'da
   `hyperparameter_overrides_used: {}` — round-2 (ve morfoloji koşusu) hiç HPO
   kullanmamış, hâlâ hardcoded hiperparametrelerle eğitilmiş. Bu, planın kendi
   "HPO atlanmışsa ucuz bir sonraki adım" notunun tam karşılığı.

**Not:** morfoloji koşusunun `meta.json`'ı kirli bir referans (root_overlap_n/
root_recall içeriyor, gerçek LB'de -0.012 verdi) — aşağıdaki adımlar TEMİZ
baseline'dan (TY_USE_TURKISH_MORPHOLOGY kapalı, varsayılan) başlamalı.

## Faz 2 — Orta efor, orta risk: LLM belirsiz-bant yeniden puanlama (06)

Hız düzeltmesi (max_new_tokens 200→40) uygulandı ama gerçek yeni hızı henüz
ölçmedik. Yapılacaklar:

1. Düzeltilmiş script ile küçük bir ölçüm yap (birkaç yüz satır, birkaç dakika) —
   gerçek satır/sn oranını öğren.
2. Tüm 504,980 satır hâlâ günler sürüyorsa, **kapsamı daralt**: belirsiz bandın
   TAMAMI yerine, modelin en belirsiz olduğu alt-dilimi işle (örn. 0.15-0.25 arası,
   ya da rastgele 50-100K satırlık bir örneklem) — kalan satırlar zaten temel
   modelin tahminiyle kalır, bu bir bug değil, bilinçli bir kapsam kısıtlaması.
3. Ayrı `submission_rescored.csv` olarak dene, gerçek LB'de 0.88'e karşı test et.
   İyileşme yoksa (ya da kötüleşme varsa) sonucu DESIGN.md'ye not düş ve bu yolu
   terk et — round-3 hard-negative mining'de yaptığımız gibi.

Beklenen katkı: +0.005 ile +0.02 arası (belirsiz, ama hedef farkın (~0.02) önemli bir
kısmını kapatabilir tek başına).

## Faz 3 — Sistematik hata analizi (SHAP zaten var, kullanılmadı)

`08_explainability.py`'ın ürettiği SHAP feature importance ve `example_explanations.json`
zaten elimizde. Bunları gerçekten OKUYUP hangi feature'ların gerçek hatalara (false
positive/false negative) yol açtığını incelemek, rastgele yeni feature denemekten
daha isabetli olur. Örnek: yanlış tahmin edilen satırların bir örneklemini elle
gözden geçir (query/title/kategori), tekrar eden bir hata deseni var mı bak
(belirli bir kategori mi, belirli bir marka mı, uzun kuyruk sorgular mı).

Bu adım doğrudan puan getirmez ama Faz 1-2'nin nereye odaklanması gerektiğini
gösterir ve aynı zamanda Explainability UI için de kullanılabilir malzeme üretir
(iki kuş bir taş).

## Faz 4 — Son çare, yüksek risk (dikkatli kullan)

- Round-3 hard-negative mining'i TEKRAR deneme — bir kere gerçek regresyona
  (0.88→0.51) yol açtı, aynı implicit-feedback yanlış etiketleme riski hâlâ geçerli.
  Eğer denenecekse, round-2'yi mutlaka yedekle ve SADECE ayrı bir submission ile test
  et, round-2'nin üzerine yazma.
- HPO'yu daha uzun timeout ile tekrar çalıştırmak (`HPO_TIMEOUT_SECONDS`) — round-2
  zaten HPO ile mi eğitildi, yoksa hardcoded hyperparametrelerle mi? `meta.json`daki
  `hyperparameter_overrides_used` alanına bak, HPO atlanmışsa bu ucuz bir sonraki adım
  olabilir.

---

## Önerilen sıra (zaman/GPU kısıtı varsa)

1. Faz 0 (ortam) — şart, atlanamaz
2. Faz 1 (beden özelliği + eşik taraması + ensemble kontrolü) — bugün/yarın, düşük risk
3. Faz 3 (SHAP hata analizi) — Faz 2'ye başlamadan önce, nereye odaklanacağını netleştirir
4. Faz 2 (LLM rescore, kapsamı daraltılmış) — asıl büyük bahis
5. Faz 4 — sadece 1-4 yeterli gelmezse, dikkatli ve yedekli şekilde

Her fazdan sonra gerçek bir Kaggle submission at ve skoru buraya getir — hangi
adımın gerçekten işe yaradığını sadece böyle öğreniriz.
