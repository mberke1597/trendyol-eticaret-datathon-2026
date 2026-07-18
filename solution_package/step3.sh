#!/usr/bin/env bash
# =============================================================================
# STEP 3 — Inference (İNTERNETSİZ). Model yükle + tahmin + submission yaz.
# Kullanım:
#   bash step3.sh <model_dump_path> <competition_data_path> <out_path>
# Örn: bash step3.sh models/ competition_data/ output/
# submission_pairs.csv otomatik bulunur; sonuç: <out_path>/submission.csv
# =============================================================================
set -euo pipefail
MODELS="${1:?model_dump_path}"; COMP="${2:?competition_data_path}"; OUT="${3:?out_path}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate ty2026

export TY_DATA_DIR="$(cd "$COMP" && pwd)"
export TY_WORK_DIR="${TY_WORK_DIR:-$(pwd)/extra_generated_data}"   # step2 cache'i
export TY_EXTRA_DATA_PATH="$TY_WORK_DIR"
export TY_MODEL_DUMP_PATH="$(cd "$MODELS" && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1                      # internet kapalı garanti

# 1) Taban GBDT tahmini (Claude-src 07; threshold meta.json'dan, rate-matched)
python "$ROOT/Claude-src/07_predict.py" --save-proba
BASE="$TY_WORK_DIR/output/submission.csv"

# 2) Düzeltici katman (0.894 -> 0.904): kanallar cache'te, corrector modeli yüklü,
#    distilasyon test-içi ve etiketsiz — tamamı offline & deterministik.
python "$ROOT/NewPipeline/92_apply_corrections.py" --base "$BASE" --out "$OUT/submission.csv"

echo "[step3] tamamlandı: $OUT/submission.csv"
