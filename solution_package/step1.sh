#!/usr/bin/env bash
# =============================================================================
# STEP 1 — Ortam kurulumu (İNTERNET GEREKTİREN TEK AŞAMA)
# Conda env + pip gereksinimleri + backbone modellerin indirilmesi.
# Kullanım:  bash step1.sh
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

conda create -y -n ty2026 python=3.10
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ty2026

pip install -r "$ROOT/requirements_solution.txt"

# Backbone modeller (embedding cache üretimi step2'de offline yapılabilsin diye
# burada indirilir; LORA/finetune YOK, backbone'lara dokunulmuyor):
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0")
snapshot_download("atasoglu/turkish-tiny-bert-uncased-mean-nli-stsb-tr")
print("backbone modeller indirildi (HF cache)")
PY

# (Opsiyonel, TY_USE_CE=1 için) takımın CE checkpoint'i — 95/96 aşamalarının girdisi.
# efeyol11 ana repo; erişilemezse bvrtuu aynası denenir.
if [ "${TY_USE_CE:-0}" = "1" ]; then
python - <<'PY'
from huggingface_hub import snapshot_download
import os
dst = os.environ.get("TY_MODEL_DUMP_PATH", "models")
for repo in ["efeyol11/trendyol-eticaret-2026-models", "bvrtuu/trendyol-eticaret-2026-models"]:
    try:
        snapshot_download(repo, allow_patterns=["ce_bgeattrmm43/*"], local_dir=dst)
        print(f"CE checkpoint indirildi: {repo} -> {dst}/ce_bgeattrmm43")
        break
    except Exception as e:
        print(f"{repo} olmadı: {e}")
PY
fi
echo "[step1] ortam hazır: conda activate ty2026"
