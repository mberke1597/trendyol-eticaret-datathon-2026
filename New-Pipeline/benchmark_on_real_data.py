"""
Run this ON KAGGLE (needs the real items.csv/terms.csv, and the un-suppressable
zeyrek print noise makes this too slow/loud to run in a small dev sandbox at
scale) to answer DESIGN.md's "recommended next steps" #1: does typo-correction
+ morphological rooting actually DO anything on real query/title text, and how
fast is it?

This does NOT touch Claude-src or write any features -- it only measures, so
you know whether it's worth the retrain before spending Kaggle GPU/CPU time on
a full integration (same "check the fire-rate first" discipline as
Claude-src/ILERLEME_PLANI.md's guidance on the size feature).

Usage (Kaggle):
    !pip install -q -r New-Pipeline/requirements.txt
    !python New-Pipeline/benchmark_on_real_data.py 2>&1 | grep -v "^APPENDING RESULT"

Env vars:
    TY_DATA_DIR       -- defaults to the same Kaggle competition path Claude-src uses
    TY_SAMPLE_QUERIES -- how many unique queries to sample for the timing/fire-rate
                         check (default 5000 -- fast enough to run in a couple
                         minutes even at zeyrek's real, un-benchmarked-until-now speed)
"""
import os
import random
import sys
import time
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import turkish_morphology as tm
from typo_tolerance import TurkishTypoCorrector, tokenize_simple

DATA_DIR = os.environ.get(
    "TY_DATA_DIR",
    "/kaggle/input/competitions/trendyol-e-ticaret-yarismasi-2026-kaggle",
)
N_SAMPLE = int(os.environ.get("TY_SAMPLE_QUERIES", "5000"))


def main():
    random.seed(0)
    print(f"[bench] reading terms.csv + items.csv from {DATA_DIR}")
    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    queries = terms["query"].fillna("").tolist()
    titles = items["title"].fillna("").tolist()
    print(f"[bench] {len(queries):,} queries, {len(titles):,} item titles")

    print("[bench] building domain vocabulary from full corpus (queries + titles)...")
    t0 = time.time()
    corrector = TurkishTypoCorrector()
    corrector.build_vocab_from_corpus(queries + titles)
    print(f"[bench] vocab built: {len(corrector.vocab_counts):,} unique tokens, "
          f"{time.time()-t0:.1f}s")

    sample_queries = random.sample(queries, min(N_SAMPLE, len(queries)))

    # ---- Part 1: typo/diacritic correction fire-rate ----
    print(f"\n[bench] --- typo_tolerance fire-rate on {len(sample_queries):,} sampled queries ---")
    n_tokens = 0
    n_changed = 0
    examples = []
    t0 = time.time()
    for q in sample_queries:
        for tok in tokenize_simple(q):
            n_tokens += 1
            corrected = corrector.correct(tok)
            if corrected != tok:
                n_changed += 1
                if len(examples) < 20:
                    examples.append((tok, corrected))
    elapsed = time.time() - t0
    fire_rate = n_changed / max(n_tokens, 1)
    print(f"[bench] {n_tokens:,} query tokens, {n_changed:,} changed by correction "
          f"({fire_rate*100:.2f}%), {elapsed:.1f}s ({n_tokens/max(elapsed,1e-6):.0f} tok/s)")
    print("[bench] sample corrections (first 20):")
    for tok, corrected in examples:
        print(f"    {tok!r:20s} -> {corrected!r}")
    if fire_rate < 0.005:
        print("[bench] WARNING: fire-rate under 0.5% -- per Claude-src's own "
              "discipline (see ILERLEME_PLANI.md's size-feature exit criterion), "
              "this is likely too rare to move macro-F1 meaningfully. Don't "
              "commit to a full retrain on this alone.")

    # ---- Part 2: morphological rooting throughput + change-rate vs naive lowercase ----
    print(f"\n[bench] --- turkish_morphology throughput on {len(sample_queries):,} sampled queries ---")
    tm.init()
    n_tokens2 = 0
    n_parsed = 0
    n_root_differs_from_lower = 0
    t0 = time.time()
    for q in sample_queries:
        for tok in tokenize_simple(q):
            n_tokens2 += 1
            result = tm.analyze_word(tok)
            if result.parsed:
                n_parsed += 1
            if result.lemma != tok.lower():
                n_root_differs_from_lower += 1
    elapsed2 = time.time() - t0
    print(f"[bench] {n_tokens2:,} tokens, {n_parsed:,} successfully parsed "
          f"({n_parsed/max(n_tokens2,1)*100:.1f}%), "
          f"{n_root_differs_from_lower:,} roots differ from raw lowercased token "
          f"({n_root_differs_from_lower/max(n_tokens2,1)*100:.1f}%), "
          f"{elapsed2:.1f}s ({n_tokens2/max(elapsed2,1e-6):.0f} tok/s)")

    full_dataset_estimate_hours = (elapsed2 / max(n_tokens2, 1)) * (len(queries) + len(titles)) * 5 / 3600
    print(f"\n[bench] rough extrapolation if you ran this over the FULL corpus "
          f"once each (no caching): ~{full_dataset_estimate_hours:.1f}h. "
          f"NOTE: real usage should cache root_of()/correct() per UNIQUE token "
          f"(Counter above shows how many unique tokens actually exist) rather "
          f"than reprocessing repeated words -- this raw estimate is a "
          f"deliberately pessimistic upper bound, not a forecast.")

    print("\n[bench] --- unique token reduction from caching (why memoization matters) ---")
    all_tokens = [tok for q in queries for tok in tokenize_simple(q)]
    print(f"[bench] {len(all_tokens):,} total query tokens, "
          f"{len(set(all_tokens)):,} unique -- caching root_of()/correct() by "
          f"unique token cuts the real workload by "
          f"{(1 - len(set(all_tokens))/max(len(all_tokens),1))*100:.1f}%")


if __name__ == "__main__":
    main()
