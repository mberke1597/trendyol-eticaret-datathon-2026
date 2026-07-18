"""
============================================================================
 TRENDYOL RELEVANCE — 0.894'ten yukarı YOL HARİTASI  (macro-F1 yarışması)
============================================================================
 Metrik: macro-F1 = (F1_relevant + F1_irrelevant) / 2   (global)
 Durum : base model AUC 0.928 -> macro-F1 ~0.89 tavan.  Bizim en iyi: 0.894.
 Hedef : 0.93+ (leaderboard #25 = 0.930, benchmark = 0.960).
 İlke  : HER adımı 80_macrof1_scorer.py ile OOF'ta ÖLÇ. LB'ye kör atma.
         Kazanç yoksa devam etme. Base modeli büyüt; filtre değil.
============================================================================

Bu dosya çalıştırılmaz — sıralı bir REÇETE. Her blok Colab/L4'te koşulacak
komutları ve beklenen çıktıyı içerir. Ortam (her oturum başında):

    import os
    os.environ["TY_DATA_DIR"]="/root/.cache/kagglehub/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle"
    os.environ["TY_WORK_DIR"]="/content/drive/MyDrive/trendyol_work"   # Drive = kalıcı
    os.environ["TY_CLAUDE_CACHE_DIR"]="/content/work/cache"
    NP="/root/.cache/kagglehub/datasets/muhammetberkeaaya/tyv41597/versions/2"
"""

# ===========================================================================
# ADIM 0 — ÖLÇÜM TEMELİ  (ucuz, ~10 dk)  |  amaç: kör uçmayı bitir
# ===========================================================================
def adim_0_olcum():
    """
    05_train.py OOF proba+label kaydediyor mu? Kaydetmiyorsa küçük ekleme yap
    (Claude fold döngüsünde oof['blend'] + y'yi np.save et).
    Sonra:
        python NewPipeline/80_macrof1_scorer.py \
            --oof-proba oof_proba.npy --oof-labels oof_labels.npy --target-rate 0.29
    ÇIKTI: OOF-optimal macroF1, F1_rel, F1_irr, ve %29 eşiğindeki macroF1.
    KARAR: F1_rel (darboğaz) kaç? 0.85 altındaysa modelin RELEVANT ayrımı zayıf
           -> Adım 1 (cross-encoder) şart.
    NOT: OOF %18.7 pozitif, test %28-31. Eşiği HEP --target-rate 0.29 ile ver.
    """

# ===========================================================================
# ADIM 1 — GÜÇLÜ CROSS-ENCODER  (ASIL kazanç, ~gün)  |  0.86 -> hedef 0.90+
# ===========================================================================
def adim_1_cross_encoder():
    """
    Mevcut CE zayıftı: tyroberta 110M, 1 epoch, max_len 192. Yükselt:
      config.py:  TY_CE_EPOCHS=3  TY_CE_MAX_LEN=256  TY_CE_LR=2e-5
      girdi: query [SEP] title | kategori | marka | cinsiyet | yaş | ATTRIBUTES
             (data.build_item_text -> attr_chars artır: 240)
      hard-negatifler: 09_hard_negative_mining sonrası train_pairs_labeled_round2 kullan.
    KOŞ (fold fold, Drive'a):
        python NewPipeline/20_build_ce_dataset.py
        python NewPipeline/21_train_crossencoder.py --fold 0   # ... 4
        python NewPipeline/22_score_crossencoder.py            # submission proba
    ÖLÇ:
        # ce OOF'u macro-F1 skorerine ver -> CE tek başına macroF1?
        python NewPipeline/80_macrof1_scorer.py --oof-proba ce_oof.npy --oof-labels y.npy
    KARAR: CE OOF macroF1 > base (0.89) mı? Evetse feature olarak kesin ekle.
           Değilse model/epoch/metin daha da güçlendir (veya XLM-R large dene).
    """

# ===========================================================================
# ADIM 2 — BASE MODELİ BÜYÜT  (orta, ~yarım gün)
# ===========================================================================
def adim_2_base_buyut():
    """
    2a) HARD-NEGATIF ROUND-2 (karar sınırını keskinleştirir, genelde +):
        python Claude-src/05_train.py            # round-1 modeller
        python Claude-src/09_hard_negative_mining.py
        TY_TRAIN_PAIRS_FILE=train_pairs_labeled_round2.parquet python Claude-src/04_build_features.py
        python Claude-src/05_train.py            # round-2 model
        -> 80 ile OOF ölç, arttı mı?
    2b) HPO (lgb/xgb/cat parametreleri):
        python Claude-src/hpo_search.py          # best_hyperparams.json yazar
        python Claude-src/05_train.py            # otomatik yükler
    2c) EMBEDDING FINE-TUNE (opsiyonel, pahalı):
        python Claude-src/00_finetune_embeddings.py
        TY_FINETUNED_MODEL_DIR=... python Claude-src/01_encode_embeddings.py
    HER 2x SONRASI: 80 ile OOF macroF1 ölç, kazanç yoksa o aşamayı at.
    """

# ===========================================================================
# ADIM 3 — BİRLEŞTİR: FEATURE + ENSEMBLE  (orta)
# ===========================================================================
def adim_3_birlestir():
    """
    Yeni sinyalleri Claude-src feature setine feature olarak ekle (flip DEĞİL):
      - cross-encoder OOF/submission prob   (Adım 1)
      - spec_conflict                       (61_spec_conflict_feature.py, %0.5 pozitif -> temiz)
      - embed kNN + çok-alan sim            (30_embed_features.py)
    KOŞ:
        python NewPipeline/61_spec_conflict_feature.py
        python NewPipeline/30_embed_features.py
        python NewPipeline/40_merge_features.py      # hepsini join
        python NewPipeline/41_train_ensemble.py      # CE'yi 4. üye + feature olarak stage'le
        TY_LLM_REL_ENSEMBLE=1 python Claude-src/05_train.py
        TY_LLM_REL_ENSEMBLE=1 python Claude-src/07_predict.py --save-proba
    ÖLÇ: 80 ile OOF macroF1 -> base'i geçti mi? geçtiyse ilerle.
    """

# ===========================================================================
# ADIM 4 — EŞİK KALİBRASYONU + SON RÖTUŞ  (ucuz)
# ===========================================================================
def adim_4_esik_ve_filtre():
    """
    a) macro-F1-optimal eşik (test yoğunluğuna):
        python NewPipeline/80_macrof1_scorer.py --apply submission_proba.npy \
            --ids submission_pairs.csv --target-rate 0.29 --out submission.csv
       (0.28 / 0.29 / 0.30 üçünü LB'ye at, en iyisini seç.)
    b) DOĞRULANMIŞ öz-nitelik filtresi (precision rötuşu, train-güvenli):
        python NewPipeline/60_attribute_contradiction_filter.py \
            --in submission.csv --out submission_final.csv
       (renk/materyal/inç YOK; sadece <%5 uyuşmayan güvenli kurallar.)
    """

# ===========================================================================
# ÖNCELİK & BEKLENEN GETİRİ
# ===========================================================================
PRIORITY = [
    # (adım,                 çaba,      beklenen getiri,      şart)
    ("0 ölçüm",              "10 dk",   "ölçebilmek",         "hemen"),
    ("1 cross-encoder",      "1-2 gün", "YÜKSEK (asıl)",      "L4/GPU"),
    ("2a hard-neg round2",   "yarım gün","orta-yüksek",       "-"),
    ("3 birleştir+ensemble", "yarım gün","orta",              "1 biter"),
    ("2b HPO",               "1-3 saat","düşük-orta",         "-"),
    ("4 eşik+filtre",        "ucuz",    "düşük (+0.003)",     "en son"),
]

# ===========================================================================
# ALTIN KURAL
# ===========================================================================
# 1) Her değişikliği ÖNCE 80 ile OOF'ta ölç; kazanç yoksa LB'ye atma.
# 2) Eşiği HER ZAMAN ~%29 pozitif hedefiyle ver (test yoğunluğu).
# 3) Base modeli büyüt (Adım 1) -> tavan bu. Filtre/eşik son rötuş.
# 4) 891_validated_v8.csv (0.894) base büyüyene kadar güvenli yedek.
