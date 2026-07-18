"""
Stage 31 (v6-safe) — LLM QUERY UNDERSTANDING with Trendyol-LLM-8B-T1.

Runs the 8B model over the ~50k UNIQUE queries ONLY (never the 3.36M pairs).
For each query it produces a STRUCTURED understanding that is robust to Turkish
morphology / synonyms / spelling — the thing regex cannot do:

    "cat bayan bot"           -> {normalized:"cat kadın bot", tip:"bot",
                                   marka:"cat", cinsiyet:"kadın"}
    "tekerli sandalye"        -> {normalized:"tekerlekli sandalye", tip:"sandalye"}
    "colins erkek jean"       -> {normalized:"colins erkek kot pantolon",
                                   tip:"pantolon", cinsiyet:"erkek",
                                   expanded:["kot","jean","denim"]}

DESIGN LESSON BAKED IN (from LB experiments):
  * We do NOT ask the LLM to judge relevance and we do NOT flip rows by it —
    zero-shot LLM prediction lost on the LB (0.86 vs 0.891).
  * We do NOT build contradictions against the item's VARIANT attributes
    (renk/materyal) — that lost too (v6 = 0.892), because an item page covers
    many colour/material variants.
  * We DO extract only the SAFE, intrinsic signals: the normalized query string,
    the product TYPE (intrinsic, like "kablo" vs "başlık"), gender, and synonym
    expansions. These feed (a) better lexical features and (b) a high-precision
    product-TYPE contradiction (stage 70) — the same reliable lane that took
    0.891 -> 0.894.

Output: LLM_QUERY_PARQUET keyed by term_id, columns:
    term_id, q_normalized, q_type, q_gender, q_expanded
Sharded + resumable.

Compute: 8B fits a single L4 (24GB, bf16). ~50k short generations = a few 10s of min.
Run:  python 31_llm_query_understanding.py     (L4: TY_LLM_ENRICH_DTYPE=bfloat16)
"""
import json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from config import (LLM_MODEL, LLM_TP, LLM_MAX_LEN, LLM_DTYPE, LLM_QUANT,
                    LLM_QUERY_PARQUET, CACHE_DIR)
from data import load_catalog

SHARD = 5000

SYS = ("Sen bir Türk e-ticaret arama motorusun. Kullanıcının kısa arama sorgusunu "
       "anlamlandır. Yalnızca geçerli JSON döndür, açıklama yazma.")

TMPL = (
    "Arama sorgusu: \"{q}\"\n"
    "Bu şemayı doldur (yalnız JSON):\n"
    '{{"normalized":"", "tip":"", "cinsiyet":"", "expanded":[]}}\n'
    "- normalized: sorgunun düzeltilmiş hali; yazım hatalarını düzelt "
    "(tekerli->tekerlekli, gardrop->gardırop), kısaltmaları aç, ekleri sadeleştir, "
    "eşanlamı standardize et (bayan->kadın, jean->kot).\n"
    "- tip: aranan ürünün ANA TÜRÜ, tekil ve sade (ör. 'bot', 'kablo', 'pantolon', "
    "'elbise'). Aksesuar mı ana ürün mü ayır (ör. 'telefon' vs 'telefon kılıfı').\n"
    "- cinsiyet: erkek/kadın/çocuk/bebek/unisex, sorguda varsa; yoksa boş.\n"
    "- expanded: 2-4 eşanlam veya alternatif arama terimi.\n"
    "Renk/materyal ÇIKARMA — onları istemiyoruz. /no_think"
)


def _parse(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def main():
    _, terms = load_catalog()
    queries = terms[["term_id", "query"]].drop_duplicates("term_id").reset_index(drop=True)
    n = len(queries)
    print(f"[31] understanding {n:,} unique queries with {LLM_MODEL}")

    from vllm import LLM, SamplingParams
    kw = dict(model=LLM_MODEL, dtype=LLM_DTYPE, tensor_parallel_size=LLM_TP,
              max_model_len=LLM_MAX_LEN, trust_remote_code=True)
    if LLM_QUANT:
        kw["quantization"] = LLM_QUANT
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=200)

    n_shards = (n + SHARD - 1) // SHARD
    for s in range(n_shards):
        sp_path = CACHE_DIR / f"_llm_qu_shard_{s}.parquet"
        if sp_path.exists():
            print(f"[31] shard {s+1}/{n_shards} exists -- skip"); continue
        lo, hi = s * SHARD, min((s + 1) * SHARD, n)
        chunk = queries.iloc[lo:hi]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYS},
             {"role": "user", "content": TMPL.format(q=q)}],
            tokenize=False, add_generation_prompt=True) for q in chunk["query"].values]
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        rows = []
        for term_id, q, o in zip(chunk["term_id"].values, chunk["query"].values, outs):
            d = _parse(o.outputs[0].text)
            exp = d.get("expanded", [])
            rows.append({
                "term_id": term_id,
                "q_normalized": str(d.get("normalized", "") or q).lower().strip(),
                "q_type": str(d.get("tip", "") or "").lower().strip(),
                "q_gender": str(d.get("cinsiyet", "") or "").lower().strip(),
                "q_expanded": " ".join(exp if isinstance(exp, list) else []).lower(),
            })
        pd.DataFrame(rows).to_parquet(sp_path, index=False)
        print(f"[31] shard {s+1}/{n_shards} [{lo:,}-{hi:,}] {time.time()-t0:.0f}s")

    parts = [pd.read_parquet(CACHE_DIR / f"_llm_qu_shard_{s}.parquet") for s in range(n_shards)]
    full = pd.concat(parts, ignore_index=True)
    full.to_parquet(LLM_QUERY_PARQUET, index=False)
    print(f"[31] wrote {full.shape} -> {LLM_QUERY_PARQUET}")
    print("[31] next: 70_type_contradiction_filter.py (product-type mismatch, high precision)")
    print("      and/or feed q_normalized/q_expanded into Claude-src lexical features.")


if __name__ == "__main__":
    main()
