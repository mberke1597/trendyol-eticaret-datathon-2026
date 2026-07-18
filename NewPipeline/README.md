# NewPipeline — çalıştırma kılavuzu

Metin-tabanlı relevance pipeline. Üç Trendyol modelinin **metin** yeteneklerini
mevcut 0.891 ensemble'a **feature ve öğrenilmiş sınıflandırıcı** olarak ekler —
asla sert override değil (Qwen dersinden: override LB'de kaybetti).

Tam mimari ve gerekçe: `DESIGN.md`. Bu dosya sadece nasıl çalıştırılacağını anlatır.

## Modeller ve rolleri (HF model kartlarından doğrulandı)

| Model | Boyut | Ne için | Doğrulanan yetenek |
|---|---|---|---|
| `Trendyol/tyroberta` | 0.1B | **Cross-encoder relevance CLASSIFIER** (merkez) | RoBERTa encoder → `AutoModelForSequenceClassification` ile 2-sınıf baş takılır |
| `Trendyol/TY-ecomm-embed-...v1.2.0` | 0.3B | Bi-encoder embed feature'ları | 768d, Matryoshka 768/512/128, cosine, `trust_remote_code`, 384 token |
| `Trendyol/Trendyol-LLM-8B-T1` | 8B | Query normalize/expand + attribute extraction | Qwen3-8B tabanlı, `/no_think`, özetleme + key-value extraction |

## Kritik: "classification, threshold değil"

Cross-encoder'ın 2-sınıf başı doğrudan `P(relevant)` verir; karar `argmax = (prob≥0.5)`.
Bu tam olarak istediğin şey. İki biçimde kullanılıyor:
1. `50_make_submission.py` → saf sınıflandırıcı submission (`ce_argmax.csv`).
2. `41` → CE olasılığı ensemble'a hem **feature** hem **4. üye** olarak girer; nihai
   kararı öğrenilmiş meta-model verir (tek bir elle-ayarlı threshold yerine).

## Ön koşul

Claude-src'nin şu çıktıları hazır olmalı (bunları zaten üretiyorsun):
- `cache/train_pairs_labeled.parquet` (stage 03 — 1.34M etiketli çift)
- `cache/train_features.parquet`, `cache/submission_features.parquet` (stage 04 — 51 feature)

`TY_CLAUDE_CACHE_DIR` bunların olduğu klasörü göstermeli (varsayılan `WORK_DIR/cache`).

## Kurulum (Colab/Kaggle)

```bash
pip install -q "transformers<5.0" sentence-transformers scikit-learn pandas pyarrow
# LLM enrich (stage 31) için ayrıca: pip install -q "vllm==0.10.1.1"

export TY_DATA_DIR=/root/.cache/kagglehub/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle
export TY_WORK_DIR=/content/work
export TY_CLAUDE_CACHE_DIR=/content/work/cache   # Claude-src feature parquet'lerinin yeri
```

## Çalıştırma sırası

Öncelik: **A (cross-encoder) → B (embed) → C (LLM)**. Her aşamadan sonra LB'de ölç;
getiri yoksa dur. A tek başına en olası kazanç.

```bash
# --- A. Cross-encoder (merkez parça) ---
python 20_build_ce_dataset.py            # fold ata (GroupKFold term_id, Claude-src ile aynı)
python 21_train_crossencoder.py          # tyroberta fine-tune, 5-fold OOF (~2-4 sa/T4)
#   tek oturuma sığmazsa fold fold: python 21_train_crossencoder.py --fold 0 ... --fold 4
python 22_score_crossencoder.py          # 3.36M submission puanla (shard+resume, ~1-2 sa/T4)

# --- B. Embedding feature'ları ---
python 30_embed_features.py              # query-kNN transfer + çok-alan sim + Matryoshka

# --- C. LLM query zenginleştirme (opsiyonel, en pahalı) ---
python 31_llm_query_enrich.py            # 8B, sadece 50k query (2xT4 için TY_LLM_ENRICH_TP=2)

# --- Birleştir + eğit + submission ---
python 40_merge_features.py              # A+B+C'yi Claude-src feature'larına join
python 41_train_ensemble.py              # enriched feature'ları Claude-src cache'ine stage'le
cd <Claude-src> && TY_LLM_REL_ENSEMBLE=1 python 05_train.py && TY_LLM_REL_ENSEMBLE=1 python 07_predict.py
python 50_make_submission.py             # CE-only sınıflandırıcı submission'lar (A/B için)
```

`41` orijinal feature parquet'lerini `.bak` olarak yedekler; geri almak için
`python 41_train_ensemble.py --restore`.

## Leakage korkulukları (doğrulandı)

- CE OOF **fold-güvenli**: her satır, kendi `term_id`'sini görmemiş fold modeliyle
  puanlanır. Fold'lar Claude-src `05_train.py` ile **birebir aynı** (GroupKFold,
  term_id, N_FOLDS=5, deterministik — test edildi: hiçbir term birden çok fold'a yayılmıyor).
- embed-kNN transfer'i `training_pairs` pozitiflerinden gelir; submission satırları
  train'e sızmaz çünkü komşuluk query-uzayında, etiket değil.
- Hiçbir metin sinyali son kararı tek başına vermez (50'deki A/B testi hariç).

## Compute özeti (T4)

| Aşama | Süre | Not |
|---|---|---|
| 21 fine-tune | 2–4 sa | 1 epoch × 5 fold, fp16 |
| 22 score 3.36M | 1–2 sa | shard+resume |
| 30 embed | <1 sa | 50k query ucuz; `TY_EMBED_ITEMS=0` ile item encode atlanır |
| 31 LLM 8B | 2–4 sa | 8B fp16 T4'e sığmaz → `TY_LLM_ENRICH_TP=2` (2×T4) veya AWQ |

## Durum

Kod yazıldı, CPU-tarafı mantık (yol tespiti, metin birleştirme, fold determinizmi
ve leakage, merge anahtarları) gerçek veriyle **test edildi ve geçti**. GPU
aşamaları (21/22/30/31) Colab/Kaggle'da çalıştırılacak. Her iddia gerçek bir
Kaggle submission'ıyla doğrulanana kadar hipotezdir.
