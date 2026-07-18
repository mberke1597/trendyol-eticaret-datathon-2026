# Çözümün Geliştirildiği Donanım

| Aşama | Ortam | GPU | CPU | RAM |
|---|---|---|---|---|
| Embedding cache (01) | Kaggle Notebook | 2x NVIDIA T4 (16GB) | Intel Xeon 2.0GHz, 4 vCPU | 29 GB |
| Taban GBDT eğitimi (05) | Google Colab | 1x NVIDIA L4 (24GB) | Intel Xeon 2.2GHz, 8 vCPU | 53 GB |
| Düzeltici katman (62/90/91/92) | CPU-only (lokal WSL2 + sandbox) | — | AMD/Intel x86_64, 4 çekirdek | 16 GB |

Notlar:
- Düzeltici katmanın tamamı (kanallar, corrector, düzeltme zinciri) GPU
  GEREKTİRMEZ; 3.36M satır için uçtan uca ~25-35 dk CPU süresi yeterlidir.
- GPU yalnızca embedding cache üretimi ve taban GBDT (XGBoost/CatBoost GPU
  modu) için hız amaçlı kullanılmıştır; CPU'da da çalışır (daha yavaş).
- Doğrulama için minimum gereksinim: 4 çekirdek CPU, 16GB RAM, ~40GB disk,
  (opsiyonel) tek T4 sınıfı GPU.
