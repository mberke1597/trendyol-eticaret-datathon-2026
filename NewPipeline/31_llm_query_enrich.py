"""
Stage 31 — generative QUERY enrichment with Trendyol-LLM-8B-T1 (Qwen3-8B based).

Runs the 8B model over the ~50k UNIQUE queries ONLY (never the 3.36M pairs — that
was the Qwen mistake). For each query, one `/no_think` structured extraction:

    {
      "normalized":  "<temizlenmiş sorgu>",
      "category":    "<tahmini kategori>",
      "brand":       "<varsa marka, yoksa boş>",
      "attributes":  {"renk": "...", "beden": "...", "tip": "..."},   # query niyeti
      "expanded":    ["eşanlamlı1", "eşanlamlı2", ...]
    }

These become (a) extra text appended to the cross-encoder / lexical query, and
(b) structured features (does the query specify a brand? colour? size? does the
item's attributes match the query's extracted attributes?). This uses the model's
documented strengths: summarisation/paraphrasing, category classification, and
attribute/key-value extraction for catalogue enrichment.

Output: LLM_QUERY_PARQUET keyed by term_id. Sharded + resumable.

Compute: 8B fp16 needs ~16GB -> single T4 is tight (use TY_LLM_ENRICH_TP=2 for
2xT4, or an AWQ build via TY_LLM_ENRICH_QUANT=awq + a quantized model id). 50k
short generations ~ a few GPU-hours.

Run:  python 31_llm_query_enrich.py
"""
import json, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from config import (
    LLM_MODEL, LLM_TP, LLM_MAX_LEN, LLM_DTYPE, LLM_QUANT, LLM_QUERY_PARQUET, CACHE_DIR,
)
from data import load_catalog

SHARD = 5000

SYS = ("Sen bir e-ticaret arama motoru asistanısın. Kullanıcının arama sorgusunu "
       "analiz et. SADECE geçerli JSON döndür, başka metin yazma.")

TMPL = ("Arama sorgusu: \"{q}\"\n"
        "Şu JSON şemasını doldur:\n"
        '{{"normalized": "", "category": "", "brand": "", '
        '"attributes": {{}}, "expanded": []}}\n'
        "normalized: sorgunun temiz hali. category: tahmini ürün kategorisi. "
        "brand: sorguda marka geçiyorsa yaz, yoksa boş. attributes: sorgudaki "
        "renk/beden/materyal/tip gibi anahtar:değer niyetleri. expanded: 3 eşanlamlı "
        "veya ilgili arama terimi. /no_think")


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
    print(f"[31] enriching {n:,} unique queries with {LLM_MODEL}")

    from vllm import LLM, SamplingParams
    kw = dict(model=LLM_MODEL, dtype=LLM_DTYPE, tensor_parallel_size=LLM_TP,
              max_model_len=LLM_MAX_LEN, trust_remote_code=True)
    if LLM_QUANT:
        kw["quantization"] = LLM_QUANT
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=256)

    n_shards = (n + SHARD - 1) // SHARD
    for s in range(n_shards):
        sp_path = CACHE_DIR / f"_llm_query_shard_{s}.parquet"
        if sp_path.exists():
            print(f"[31] shard {s+1}/{n_shards} exists -- skip"); continue
        lo, hi = s * SHARD, min((s + 1) * SHARD, n)
        chunk = queries.iloc[lo:hi]
        prompts = []
        for q in chunk["query"].values:
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": TMPL.format(q=q)}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        t0 = time.time()
        outs = llm.generate(prompts, sp)
        rows = []
        for term_id, q, o in zip(chunk["term_id"].values, chunk["query"].values, outs):
            d = _parse(o.outputs[0].text)
            attrs = d.get("attributes", {}) or {}
            rows.append({
                "term_id": term_id,
                "q_normalized": str(d.get("normalized", "") or q),
                "q_category": str(d.get("category", "") or ""),
                "q_brand": str(d.get("brand", "") or ""),
                "q_has_brand": int(bool(d.get("brand"))),
                "q_n_attrs": int(len(attrs) if isinstance(attrs, dict) else 0),
                "q_attrs_json": json.dumps(attrs, ensure_ascii=False),
                "q_expanded": " ".join(d.get("expanded", []) if isinstance(d.get("expanded"), list) else []),
            })
        pd.DataFrame(rows).to_parquet(sp_path, index=False)
        print(f"[31] shard {s+1}/{n_shards} [{lo:,}-{hi:,}] {time.time()-t0:.0f}s")

    parts = [pd.read_parquet(CACHE_DIR / f"_llm_query_shard_{s}.parquet") for s in range(n_shards)]
    full = pd.concat(parts, ignore_index=True)
    full.to_parquet(LLM_QUERY_PARQUET, index=False)
    print(f"[31] wrote {full.shape} -> {LLM_QUERY_PARQUET}")


if __name__ == "__main__":
    main()
