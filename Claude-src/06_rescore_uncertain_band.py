"""
OPTIONAL uncertain-band re-scoring -- OFF by default (config.USE_UNCERTAIN_BAND_RESCORE).
07_predict.py produces a complete, correct submission.csv without ever calling
this script.

Why this exists (see Claude-src/DESIGN.md, Taobao/Alibaba "coarse-to-fine"
pattern): once the GBDT ensemble's threshold is calibrated, most classification
errors concentrate in a narrow probability band right around the cut point
(config.UNCERTAIN_BAND_LOW..HIGH, default 0.35-0.65) -- rows the model is
genuinely unsure about. Rather than re-scoring all 3.36M submission rows with
something heavier (explicitly disallowed by prompt.md section 3), this script
re-scores ONLY that narrow band with an LLM, bounding the extra compute to
however many rows actually fall in the uncertain zone (typically a small
fraction of the full submission set).

Re-scoring reuses the same LLMClient/MockLLMClient abstraction as
02_llm_enrichment.py (see that file for the real-vs-mock split rationale).
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    CACHE_DIR,
    DATA_DIR,
    LLM_BATCH_SIZE,
    MODEL_DIR,
    OUTPUT_DIR,
    RESCORE_BLEND_LLM_WEIGHT,
    RESCORE_MODE,
    UNCERTAIN_BAND_HIGH,
    UNCERTAIN_BAND_LOW,
    USE_UNCERTAIN_BAND_RESCORE,
)

RESCORE_PROMPT_TMPL = """Sen bir e-ticaret arama alaka (relevance) uzmanısın.
Aşağıdaki arama sorgusu ile ürün gerçekten alakalı mı, dikkatlice değerlendir.

Sorgu: "{query}"
Ürün başlığı: "{title}"
Kategori: "{category}"
Marka: "{brand}"
Öznitelikler: "{attributes}"

Modelin bu çift için verdiği olasılık {model_proba:.3f} idi (belirsiz bölgede,
{low:.2f}-{high:.2f} arası) -- yeniden değerlendir.
YALNIZCA geçerli JSON döndür: {{"relevant": true/false, "confidence": 0.0-1.0}}"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_llm_json(text):
    m = _JSON_BLOCK_RE.search(text or "")
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def rescore_band(df_band, client, batch_size=LLM_BATCH_SIZE):
    """df_band must have: query, title, category, brand, attributes, model_proba.
    Returns an array of 0/1 predictions, one per row, falling back to the
    original threshold-based prediction (proba >= mid-band) whenever the LLM
    response fails to parse -- re-scoring must never leave a row unpredicted.

    HARDENED 2026-07-04: batches LLM_BATCH_SIZE rows per client.generate_batch
    call instead of one row at a time (see 02_llm_enrichment.LLMClient's
    docstring for why this matters at scale) -- the uncertain band can easily
    be tens-to-hundreds of thousands of rows out of 3.36M, so this is not an
    optional micro-optimization here either."""
    out = np.empty(len(df_band), dtype=np.int8)
    mid = (UNCERTAIN_BAND_LOW + UNCERTAIN_BAND_HIGH) / 2
    n = len(df_band)
    t0 = time.time()
    rows = list(df_band.itertuples())
    for start in range(0, n, batch_size):
        chunk = rows[start:start + batch_size]
        prompts = [
            RESCORE_PROMPT_TMPL.format(
                query=row.query, title=row.title, category=row.category, brand=row.brand,
                attributes=row.attributes, model_proba=row.model_proba,
                low=UNCERTAIN_BAND_LOW, high=UNCERTAIN_BAND_HIGH,
            )
            for row in chunk
        ]
        # PERF FIX 2026-07-06: default max_new_tokens=200 was ~10x more than this
        # prompt ever needs (the required output is a tiny JSON object like
        # {"relevant": true, "confidence": 0.85}, well under 40 tokens). With no
        # early EOS, every batch paid for the full 200-token generation regardless
        # -- confirmed on real Kaggle run at 0.3 rows/s (504,980 rows would take
        # ~20 days). Capping at 40 tokens should cut generation time ~5x for free.
        responses = client.generate_batch(prompts, max_new_tokens=40)
        for j, (row, resp) in enumerate(zip(chunk, responses)):
            parsed = parse_llm_json(resp)
            idx = start + j
            if "relevant" in parsed:
                out[idx] = 1 if parsed["relevant"] else 0
            else:
                out[idx] = 1 if row.model_proba >= mid else 0
        done = min(start + batch_size, n)
        if done % 20 < batch_size or done == n:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else float("nan")
            print(f"  ...{done:,}/{n:,} rescored ({elapsed:.1f}s, {rate:.1f}/s, ETA {eta/60:.1f}min)")
    return out


def rescore_band_blend(df_band, client, batch_size=LLM_BATCH_SIZE):
    """NEW 2026-07-08, "blend" mode (config.RESCORE_MODE): same batched LLM
    calls as rescore_band(), but instead of the LLM's binary call REPLACING
    the GBDT's band predictions outright, this returns a per-row LLM_PROBA
    (a real-valued relevance probability -- `confidence` if relevant=true,
    `1-confidence` if relevant=false) that the caller weighted-averages with
    the GBDT's own proba (config.RESCORE_BLEND_LLM_WEIGHT). This is the
    "combine two ways of computing relevancy" mode the GBDT-ensemble's own
    lgb/xgb/cat blend already does one level down -- letting both signals
    vote proportionally to confidence rather than one hard-overriding the
    other. Falls back to the row's own model_proba (i.e. a no-op contribution
    to the blend) whenever the LLM response fails to parse, same "never leave
    a row unpredicted" discipline as rescore_band()."""
    n = len(df_band)
    llm_proba = np.asarray(df_band["model_proba"].values, dtype=np.float64).copy()
    t0 = time.time()
    rows = list(df_band.itertuples())
    for start in range(0, n, batch_size):
        chunk = rows[start:start + batch_size]
        prompts = [
            RESCORE_PROMPT_TMPL.format(
                query=row.query, title=row.title, category=row.category, brand=row.brand,
                attributes=row.attributes, model_proba=row.model_proba,
                low=UNCERTAIN_BAND_LOW, high=UNCERTAIN_BAND_HIGH,
            )
            for row in chunk
        ]
        responses = client.generate_batch(prompts, max_new_tokens=40)
        for j, (row, resp) in enumerate(zip(chunk, responses)):
            parsed = parse_llm_json(resp)
            idx = start + j
            if "relevant" in parsed:
                conf = float(parsed.get("confidence", 0.5))
                conf = min(max(conf, 0.0), 1.0)  # LLM confidence isn't guaranteed in-range
                llm_proba[idx] = conf if parsed["relevant"] else (1.0 - conf)
            # else: leave llm_proba[idx] at its model_proba initialization --
            # an unparseable response contributes nothing to the blend rather
            # than corrupting it with a made-up value.
        done = min(start + batch_size, n)
        if done % 20 < batch_size or done == n:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else float("nan")
            print(f"  ...{done:,}/{n:,} rescored (blend mode, {elapsed:.1f}s, {rate:.1f}/s, ETA {eta/60:.1f}min)")
    return llm_proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Use MockLLMClient on a tiny sample")
    parser.add_argument("--force", action="store_true", help="Run even if USE_UNCERTAIN_BAND_RESCORE=0")
    args = parser.parse_args()

    if not (USE_UNCERTAIN_BAND_RESCORE or args.force or args.dry_run):
        print("[main] USE_UNCERTAIN_BAND_RESCORE is off (default) -- skipping. "
              "07_predict.py's output is already complete without this stage.")
        return

    proba_path = Path(f"{OUTPUT_DIR}/submission_proba.npy")
    if not proba_path.exists():
        print(f"[main] {proba_path} not found -- run 07_predict.py with --save-proba first.")
        return
    proba = np.load(proba_path)

    sub_pairs = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv")
    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    df = sub_pairs.merge(terms, on="term_id", how="left").merge(items, on="item_id", how="left")
    df["model_proba"] = proba

    band_mask = (proba >= UNCERTAIN_BAND_LOW) & (proba <= UNCERTAIN_BAND_HIGH)
    df_band = df[band_mask].copy()
    print(f"[main] {band_mask.sum():,}/{len(df):,} rows ({band_mask.mean()*100:.2f}%) "
          f"fall in the uncertain band [{UNCERTAIN_BAND_LOW}, {UNCERTAIN_BAND_HIGH}]")

    if args.dry_run:
        from importlib import import_module
        llm_mod = import_module("02_llm_enrichment")
        client = llm_mod.MockLLMClient()
        df_band = df_band.head(50)
        print("[main] --dry-run: rescoring only 50 rows with MockLLMClient")
    else:
        from importlib import import_module
        llm_mod = import_module("02_llm_enrichment")
        client = llm_mod.LLMClient()

    with open(MODEL_DIR / "meta.json") as fh:
        meta = json.load(fh)
    # FIXED 2026-07-05: this used to hardcode meta["threshold"] (the OOF-
    # calibrated default) as the baseline for every row outside the
    # re-scored band. Real runs this competition have repeatedly needed a
    # much lower TY_THRESHOLD override to hit the actual ~28-31% target
    # density (see 07_predict.py / DESIGN.md) -- silently ignoring that
    # override here would have reverted ~99% of the submission (everything
    # outside the narrow re-scored band) back to the wrong ~9% positive
    # rate instead of the calibrated ~28-31%. Must match 07_predict.py.
    threshold = float(os.environ.get("TY_THRESHOLD", meta["threshold"]))
    print(f"[main] baseline threshold={threshold:.4f} (meta default={meta['threshold']:.4f})")
    final_pred = (proba >= threshold).astype(np.int8)

    print(f"[main] rescore mode={RESCORE_MODE!r}"
          + (f" (llm_weight={RESCORE_BLEND_LLM_WEIGHT})" if RESCORE_MODE == "blend" else ""))
    if RESCORE_MODE == "blend":
        # NEW 2026-07-08: soft-combine instead of hard-override -- see
        # rescore_band_blend()'s docstring. Re-threshold the BLENDED proba at
        # the same `threshold` used for every other row, so band rows are
        # decided on a like-for-like probability scale, not a separate rule.
        llm_proba = rescore_band_blend(df_band, client)
        blended = ((1 - RESCORE_BLEND_LLM_WEIGHT) * df_band["model_proba"].values
                   + RESCORE_BLEND_LLM_WEIGHT * llm_proba)
        final_pred[df_band.index.values] = (blended >= threshold).astype(np.int8)
    else:
        band_pred = rescore_band(df_band, client)
        final_pred[df_band.index.values] = band_pred

    # FIXED 2026-07-05: this script built `df` from raw sub_pairs/terms/items
    # only -- it never applied the hard gender/age override 07_predict.py's
    # module docstring calls "NOT optional and must never be removed" (see
    # that file). Re-apply it here too, from the same submission_features.parquet
    # columns 07_predict.py uses, so submission_rescored.csv is never missing
    # this rule just because it went through the LLM re-scoring path instead.
    feat_small = pd.read_parquet(
        f"{CACHE_DIR}/submission_features.parquet",
        columns=["id", "gender_contradiction", "age_contradiction"],
    ).set_index("id")
    feat_aligned = feat_small.reindex(df["id"].values)
    gender_override = feat_aligned["gender_contradiction"].values > 0
    age_override = feat_aligned["age_contradiction"].values > 0
    any_override = gender_override | age_override
    n_overridden = int((any_override & (final_pred == 1)).sum())
    final_pred = np.where(any_override, 0, final_pred)
    print(f"[main] hard gender/age override flipped {n_overridden:,} predictions to 0 "
          f"(gender={int(gender_override.sum()):,}, age={int(age_override.sum()):,} rows flagged)")

    out_path = OUTPUT_DIR / "submission_rescored.csv"
    pd.DataFrame({"id": df["id"].values, "prediction": final_pred}).to_csv(out_path, index=False)
    n_changed = int((final_pred[df_band.index.values] != (proba[df_band.index.values] >= threshold)).sum())
    print(f"[main] rescoring changed {n_changed:,}/{band_mask.sum():,} band predictions")
    print(f"[main] predicted positive rate = {final_pred.mean()*100:.2f}%")
    print(f"[main] wrote {out_path}")


if __name__ == "__main__":
    main()
