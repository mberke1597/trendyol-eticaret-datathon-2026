#!/usr/bin/env bash
# =============================================================================
# STEP 2 — Veri hazırlama & üretme + eğitim + model kaydetme
# Kullanım:
#   bash step2.sh <competition_data_path> <extra_data_path> <model_dump_path>
# Örn: bash step2.sh competition_data/ extra_generated_data/ models/
#
# Üretilen ek veriler (üretici script -> çıktı eşlemesi):
#   Claude-src/01_encode_embeddings.py -> <extra>/cache/*.npy         (embedding cache)
#   Claude-src/03_negative_sampling.py -> <extra>/train_pairs_labeled.parquet (negatif örnekler)
#   NewPipeline/62_clean_negatives.py  -> <extra>/train_pairs_labeled_clean.parquet (veto'lu temiz set)
#   NewPipeline/90_build_channels.py   -> <extra>/channels_train_clean.npz, channels_submission.npz
# Eğitilen modeller:
#   Claude-src/05_train.py             -> <models>/lgb_fold*.txt, xgb_fold*.json, cat_fold*.cbm, meta.json (TABAN)
#   NewPipeline/91_train_corrector.py  -> <models>/corrector_lgb.txt (DÜZELTİCİ KATMAN)
# =============================================================================
set -euo pipefail
COMP="${1:?competition_data_path}"; EXTRA="${2:?extra_data_path}"; MODELS="${3:?model_dump_path}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$EXTRA" "$MODELS"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ty2026

export TY_DATA_DIR="$(cd "$COMP" && pwd)"
export TY_WORK_DIR="$(cd "$EXTRA" && pwd)"          # Claude-src cache/output buraya
export TY_EXTRA_DATA_PATH="$(cd "$EXTRA" && pwd)"
export TY_MODEL_DUMP_PATH="$(cd "$MODELS" && pwd)"
export TY_TRAIN_PAIRS_FILE=train_pairs_labeled_clean.parquet   # temiz etiketler (KANITLI)
export TY_NEG_VETO=1                                           # madencilik-anı veto (KANITLI)

# --- TABAN PIPELINE (embeddings -> negatifler -> features -> 3'lü GBDT) ---
python "$ROOT/Claude-src/01_encode_embeddings.py"
python "$ROOT/Claude-src/03_negative_sampling.py"
python "$ROOT/NewPipeline/62_clean_negatives.py"               # SKOR KATMANI: temiz etiketler
python "$ROOT/Claude-src/04_build_features.py"
python "$ROOT/Claude-src/12_add_group_features.py"             # SKOR KATMANI: liste-yapısı feature'ları
python "$ROOT/Claude-src/13_query_neighbor_features.py"        # SKOR KATMANI: komşu-transfer feature'ları
python "$ROOT/Claude-src/05_train.py"
cp -f "$TY_WORK_DIR"/models/* "$TY_MODEL_DUMP_PATH"/ 2>/dev/null || true

# --- DÜZELTİCİ KATMAN (0.894 -> 0.904; ayrıntı: SOLUTION_README.md §NewPipeline) ---
python "$ROOT/NewPipeline/94_mine_synonyms.py"                 # graf'tan eşanlam/typo sözlüğü
python "$ROOT/NewPipeline/90_build_channels.py" --pairs train_clean
python "$ROOT/NewPipeline/90_build_channels.py" --pairs submission
python "$ROOT/NewPipeline/93_graph_features.py" --pairs train_clean   # TAM graf madenciliği (24 kanal)
python "$ROOT/NewPipeline/93_graph_features.py" --pairs submission

# (Opsiyonel, GPU) CE entegrasyonu: TY_USE_CE=1 ise 0.88'lik bge-reranker
# checkpoint'i temiz etiketlerle fine-tune edilir ve skoru 25. kanal olur.
if [ "${TY_USE_CE:-0}" = "1" ]; then
  python "$ROOT/NewPipeline/95_ce_finetune_clean.py"
  python "$ROOT/NewPipeline/96_add_ce_channel.py"
fi

python "$ROOT/NewPipeline/91_train_corrector.py"

echo "[step2] tamamlandı: ek veriler -> $EXTRA , modeller -> $MODELS"
