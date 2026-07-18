"""
OPTIONAL, offline/nearline LLM enrichment pass -- OFF by default
(config.USE_LLM_ENRICHMENT). The rest of the pipeline runs correctly without
ever calling this script; it is a quality upgrade, not a dependency.

Why this exists (see Claude-src/DESIGN.md "buyuk sirket arastirmasi" section):
Amazon's "Query Structure LLM" pattern and Walmart's LLM-distilled cross-encoder
(ACM Web Conf 2025) both use an LLM *offline*, over a bounded set of unique
queries and/or a catalog subset, to produce structured signals that get baked
into cheap features for a GBDT at serving time -- never an LLM call in the
3.36M-row hot inference path. This script does the same two things:

  1. Query intent extraction: for each UNIQUE query (small, bounded set --
     terms.csv, not per-pair), ask the LLM for structured fields (gender/age/
     color/material/pattern/brand hints, category guess). This is strictly an
     upgrade path over item_meta.py's regex/vocab extraction for queries the
     regex misses (typos, slang, abbreviations).
  2. Item attribute cleanup: for a BOUNDED subset of items (only those that
     appear in train_pairs/candidates -- config.LLM_ENRICHMENT_MAX_ITEMS caps
     this, since Kaggle's ~30h/week GPU quota can't cover a 12B-param pass over
     the full 962k-item catalog), ask the LLM to re-extract color/material/
     pattern/gender/age from the noisy `attributes` + `title` text, as a
     cleaner alternative to item_meta.py's regex when seller metadata is messy.

Both outputs are cached to parquet and consumed as EXTRA features in
features.py (LLM columns are optional -- if the cache file doesn't exist,
04_build_features.py simply skips those columns).

Model: config.LLM_MODEL_NAME (default Trendyol/Trendyol-LLM-Asure-12B, a
Gemma3-12B-based model whose HF card explicitly lists e-commerce relevance
tasks). Loading a 12B model requires a Kaggle GPU session with internet on;
this script is NOT runnable in a CPU-only / no-GPU sandbox, and there is no
attempt here to fake that -- see `--dry-run` below for logic verification
without a real model.
"""
import argparse
import json
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
    LLM_ENRICHMENT_MAX_ITEMS,
    LLM_MODEL_NAME,
    USE_LLM_ENRICHMENT,
)

QUERY_PROMPT_TMPL = """Sen bir e-ticaret arama motoru için sorgu analiz asistanısın.
Aşağıdaki Türkçe arama sorgusunu analiz et ve YALNIZCA geçerli JSON döndür.

Sorgu: "{query}"

JSON şeması:
{{
  "gender": "kadın" | "erkek" | "unisex" | null,
  "age_group": "yetiskin" | "cocuk" | "bebek" | null,
  "color": string veya null,
  "material": string veya null,
  "pattern": string veya null,
  "brand_hint": string veya null,
  "category_guess": string veya null
}}"""

ITEM_PROMPT_TMPL = """Sen bir e-ticaret ürün kataloğu analiz asistanısın. Satıcı
tarafından girilen ürün metadatası gürültülü/eksik olabilir. Başlık ve
özniteliklerden gerçek değerleri çıkar, YALNIZCA geçerli JSON döndür.

Başlık: "{title}"
Öznitelikler: "{attributes}"
Kategori: "{category}"

JSON şeması:
{{
  "gender": "kadın" | "erkek" | "unisex" | "unknown",
  "age_group": "yetiskin" | "cocuk" | "bebek" | "unknown",
  "color": string veya null,
  "material": string veya null,
  "pattern": string veya null
}}"""

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_llm_json(text):
    """Best-effort JSON extraction from an LLM completion (handles markdown
    fences / leading chatter). Returns {} on failure -- callers must treat LLM
    fields as optional and fall back to rule-based extraction."""
    m = _JSON_BLOCK_RE.search(text or "")
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


class LLMClient:
    """Thin wrapper so 04_build_features.py / tests never import transformers
    directly. Real backend loads Trendyol-LLM-Asure-12B via `transformers`;
    swap `generate_batch()` for an API-backed client (vLLM server, etc.) if
    preferred -- nothing else in the pipeline needs to change.

    HARDENED 2026-07-04 for real use: the original version only exposed a
    one-prompt-at-a-time `generate()`, which `enrich_queries`/`enrich_items`
    called in a plain Python loop -- for terms.csv's ~50k unique queries that's
    ~50k SEQUENTIAL forward passes through a 12B model. At a conservative
    1-2s/call that's 14-28 GPU-hours for query enrichment ALONE, before even
    getting to items -- almost certainly infeasible inside one Kaggle session
    (let alone the weekly GPU quota). `generate_batch()` below pads and runs
    LLM_BATCH_SIZE prompts through the model in one forward pass, which is the
    difference between "impossible on Kaggle" and "actually runnable".

    FIXED 2026-07-08 (real bug, never previously exercised on a real GPU --
    02_llm_enrichment.py had only ever been --dry-run tested until now):
    this loaded the model via `AutoModelForCausalLM` + `AutoTokenizer`, but
    Trendyol-LLM-Asure-12B is a Gemma3-based MULTIMODAL checkpoint (its own
    HF model card's "Basic Usage" section loads it via `AutoModelForImageTextToText`
    + `AutoProcessor`, and applies the chat template rather than raw string
    tokenization). Loading a multimodal Gemma3 checkpoint through the plain
    causal-LM auto-class is not the officially supported path and risks
    silently wrong behavior (or an outright load error) -- switched to the
    model card's own documented classes. We never send images (this
    competition's items.csv/terms.csv are text-only, no image field --
    confirmed by grep, so the model's vision half is simply unused here),
    but Gemma3's chat template still expects the `content: [{"type": "text",
    ...}]` message structure even for text-only turns, not a raw prompt
    string. `transformers==4.51.3` (already pinned in requirements.txt for
    the embedding model) is new enough to include Gemma3 + AutoModelForImageTextToText
    support (added ~4.46-4.50), so no new pin was needed.

    ARCHITECTURE-AUTO-DETECTED 2026-07-08 (real throughput problem, not a
    bug): a real Kaggle run of Asure-12B, even after the max_new_tokens cap
    and single-GPU 8-bit quantization fixes, still measured ~0.3 rows/s
    (50,153 queries -> tens of hours) -- HF's plain eager-attention
    `generate()` on a T4 doing autoregressive decoding through 12B params is
    just slow, no further easy win available without a serving framework
    (vLLM/TGI) this project doesn't have set up. The practical mitigation is
    swapping to a SMALLER Trendyol model (e.g. `Trendyol/Trendyol-LLM-7B-chat-v4.1.0`,
    a Qwen2.5-7B-based text-only chat model -- NOT multimodal, NOT the same
    "new-impl" Gemma3 remote-code path, so it doesn't need eager-attention
    forcing either, a plausible second speedup on top of the raw size cut)
    via `TY_LLM_MODEL=Trendyol/Trendyol-LLM-7B-chat-v4.1.0`. Rather than hardcode
    one loading path and break the moment someone swaps `LLM_MODEL_NAME`,
    `__init__` now checks `AutoConfig.from_pretrained(model_name)` for a
    `vision_config` attribute (a reliable signal for multimodal checkpoints
    -- Gemma3's multimodal config has one, Qwen2.5/Llama/Mistral-style
    text-only configs don't) and picks `AutoModelForImageTextToText`+
    `AutoProcessor` or `AutoModelForCausalLM`+`AutoTokenizer` accordingly,
    so both model families work through the same class without a manual
    flag to remember. Quality tradeoff, not free: the 7B model is a general
    Turkish/English chat model, NOT specifically tuned for e-commerce
    structured-extraction/relevance the way Asure-12B's card claims to be --
    untested whether its JSON-extraction quality holds up for this use case,
    verify a handful of outputs by eye before trusting it at scale."""

    def __init__(self, model_name=LLM_MODEL_NAME):
        import torch
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name)
        self.is_multimodal = hasattr(cfg, "vision_config")
        print(f"[LLMClient] {model_name}: detected "
              f"{'multimodal (image-text-to-text)' if self.is_multimodal else 'text-only causal-LM'} "
              "architecture (via AutoConfig.vision_config presence)")

        if self.is_multimodal:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            self.tok = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = self.tok.tokenizer
            model_cls = AutoModelForImageTextToText
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tok = self.tokenizer  # same object -- __call__/apply_chat_template both live on it
            model_cls = AutoModelForCausalLM

        # Left-padding is required for batched causal-LM generation: with right-
        # padding, each sequence's "next token to generate" position differs per
        # row and the model would be generating from mid-sequence for shorter
        # prompts.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Kaggle's free-tier GPU is T4 (Turing, compute capability 7.5), which has
        # NO hardware bfloat16 support -- bf16 ops either silently upcast (slow) or
        # error depending on the op/driver. Only use bf16 when the visible CUDA
        # device actually supports it (Ampere+ / A100, T4 x2 upgrades, etc.);
        # fall back to float16 (T4's native low-precision format) otherwise.
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        print(f"[LLMClient] loading {model_name} with dtype={dtype} (bf16_supported={use_bf16})")

        # A 12B-param model needs ~24GB just for fp16/bf16 weights -- fits Kaggle's
        # T4 x2 (32GB total, device_map="auto" shards across both) but NOT a
        # single-GPU box like Colab's one 15GB T4.
        #
        # FIXED 2026-07-08 (real failure on a single-GPU Colab session): the old
        # `load_in_8bit=True` raw-boolean kwarg is deprecated AND, more
        # importantly, bitsandbytes' 8-bit quantizer only gets ~12GB down from
        # 24GB -- device_map="auto" on a single 15GB GPU still had to dispatch
        # some layers to CPU/disk, and the 8-bit quantizer refuses that by
        # default: `ValueError: Some modules are dispatched on the CPU or the
        # disk...`. Switched to a proper `BitsAndBytesConfig`, and default to
        # **4-bit (NF4)** instead of 8-bit when quantizing -- 4-bit needs only
        # ~6-7GB for the weights, comfortably fits a single 15GB GPU with room
        # for activations/KV-cache, no CPU offload needed. 8-bit is still
        # available (TY_LLM_QUANT=8bit) for Kaggle's roomier T4 x2, where the
        # quality-vs-memory tradeoff is less forced.
        #
        # STRONG RECOMMENDATION even on Kaggle's 2xT4 (2026-07-08): quantize
        # anyway for THIS script specifically. Unquantized bf16 needs both
        # GPUs, so `device_map="auto"` naively pipeline-splits layers across
        # them -- fine for a single embedding forward pass (00/01 already
        # avoid this by forcing one device), but ruinous for autoregressive
        # generation, where EVERY generated token re-crosses the GPU0<->GPU1
        # boundary at PCIe latency, once per layer split, for as many tokens
        # as max_new_tokens. A real run confirmed 0.17 rows/s (50,153 queries
        # projected to ~81 hours) with this 2-GPU bf16 path. Quantizing to
        # 8-bit (TY_LLM_QUANT=8bit, ~12GB) lets the WHOLE model fit on ONE of
        # the two T4s -- accelerate's device_map="auto" then keeps it on a
        # single device with no cross-GPU hop at all, which should remove
        # this specific bottleneck (not yet re-measured after this fix --
        # confirm the new rows/s before trusting it at full scale).
        import os

        quant = os.environ.get("TY_LLM_QUANT", "")
        if not quant and os.environ.get("TY_LLM_LOAD_IN_8BIT", "0") == "1":
            quant = "8bit"  # back-compat with the old env var name

        # FIXED 2026-07-11 (real bug, seen live via Kaggle's GPU-monitor sidebar
        # during an actual 7B-model run): device_map="auto" does NOT mean "pack
        # onto one GPU if it fits". With T4 x2 visible, accelerate's balancer
        # split this 7B model's layers across BOTH GPUs anyway (observed: GPU0
        # 13% util / GPU1 93% util -- wildly uneven, the signature of
        # pipeline-parallel generation where every token's forward pass
        # re-crosses the GPU0<->GPU1 PCIe boundary at the layer-split point).
        # This is the exact cross-GPU cost already flagged below for the
        # unquantized 12B/2-GPU case -- it turns out "auto" triggers it even
        # for a 7B model in bf16 (~14GB) that comfortably fits ONE 15GB T4
        # alone. Default now FORCES single-GPU placement (device_map={"": 0})
        # regardless of model size; only override to real multi-GPU sharding
        # via TY_LLM_DEVICE_MAP=auto for a model too large to fit one GPU even
        # quantized (shouldn't be needed up to ~12B at 4-bit, ~6-7GB -- the
        # unquantized 12B bf16 case below, ~24GB, is the one real exception).
        device_map_env = os.environ.get("TY_LLM_DEVICE_MAP", "0")
        device_map = "auto" if device_map_env == "auto" else {"": int(device_map_env)}

        load_kwargs = dict(torch_dtype=dtype, device_map=device_map)
        if quant == "4bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            )
            print("[LLMClient] TY_LLM_QUANT=4bit -- loading in 4-bit NF4 (requires bitsandbytes, "
                  "~6-7GB weights -- use this on a single-GPU box like Colab's 1xT4)")
        elif quant == "8bit":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            print("[LLMClient] TY_LLM_QUANT=8bit -- loading in 8-bit (requires bitsandbytes, "
                  "~12GB weights -- needs Kaggle's T4 x2 or another 16GB+ single GPU; "
                  "if you hit 'dispatched on the CPU or the disk', use TY_LLM_QUANT=4bit instead)")
        self.model = model_cls.from_pretrained(model_name, **load_kwargs)

    def generate_batch(self, prompts, max_new_tokens=200):
        # apply_chat_template PER-PROMPT (tokenize=False -- just formats the
        # string with the model's chat markup), then batch-tokenize all the
        # formatted strings together via self.tok for padding. Batching
        # apply_chat_template itself across multiple conversations has
        # inconsistent support across transformers versions; this two-step
        # form (format each, then batch-tokenize) is the robust pattern and
        # matches how 00/01's SentenceTransformer batching is structured
        # elsewhere in this codebase (build strings first, tokenize as a batch).
        #
        # Multimodal (Gemma3-style) chat templates expect `content` as a list
        # of typed parts (`[{"type": "text", "text": ...}]`) even for
        # text-only turns; text-only causal-LM templates (Qwen2.5/Llama/
        # Mistral-style) expect `content` as a plain string. Branch on
        # self.is_multimodal (set in __init__ via AutoConfig) so both
        # families work through the same generate_batch().
        content = (lambda p: [{"type": "text", "text": p}]) if self.is_multimodal else (lambda p: p)
        texts = [
            self.tok.apply_chat_template(
                [{"role": "user", "content": content(p)}],
                add_generation_prompt=True, tokenize=False,
            )
            for p in prompts
        ]
        inputs = self.tok(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        out = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen_only = out[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

    def generate(self, prompt, max_new_tokens=200):
        """Single-prompt convenience wrapper -- prefer generate_batch for real
        runs, this exists for interface parity with MockLLMClient / tests."""
        return self.generate_batch([prompt], max_new_tokens=max_new_tokens)[0]


class MockLLMClient:
    """Deterministic stand-in used by --dry-run and by tests -- exercises the
    prompt-build / JSON-parse / caching logic end-to-end without a real model
    or GPU. Returns item_meta.py-style rule-based guesses, clearly NOT a real
    LLM signal (field values are simplistic on purpose)."""

    def generate(self, prompt, max_new_tokens=200):
        q_match = re.search(r'Sorgu: "([^"]*)"', prompt)
        if q_match:
            from item_meta import age_intent_from_tokens, gender_intent_from_tokens
            from text_utils import tokenize

            toks = tokenize(q_match.group(1))
            gender = gender_intent_from_tokens(toks)
            age = age_intent_from_tokens(toks)
            return json.dumps({
                "gender": gender, "age_group": age, "color": None, "material": None,
                "pattern": None, "brand_hint": None, "category_guess": None,
            }, ensure_ascii=False)
        return json.dumps({
            "gender": "unknown", "age_group": "unknown", "color": None,
            "material": None, "pattern": None,
        }, ensure_ascii=False)

    def generate_batch(self, prompts, max_new_tokens=200):
        return [self.generate(p, max_new_tokens=max_new_tokens) for p in prompts]


def _run_batched(items_iterable, n, id_field, id_values, client, desc, batch_size=LLM_BATCH_SIZE,
                  max_new_tokens=100):
    """Shared batched-generation driver for both queries and items: builds
    prompts batch_size at a time, calls client.generate_batch ONCE per batch
    (not once per row -- see LLMClient's module-docstring note on why this
    matters), parses each response, and reports progress in wall-clock terms
    so a real Kaggle run's ETA is visible early rather than only at the end.

    PERF FIX 2026-07-08 (same class of bug as 06_rescore_uncertain_band.py's
    2026-07-06 fix, just never applied here): this used to call
    client.generate_batch(batch_prompts) with NO max_new_tokens override,
    silently defaulting to 200 -- ~2-3x more than QUERY_PROMPT_TMPL/
    ITEM_PROMPT_TMPL's small JSON schemas ever need, with no early-stopping
    savings observed in practice. Confirmed on a real Kaggle run at
    0.17 rows/s (50,153 queries alone projected to ~81 HOURS) -- capping at
    100 tokens is the first of two fixes for that (see LLMClient's
    module docstring for the other: naive 2-GPU device_map="auto" pipeline
    parallelism is also a likely major contributor for generation-heavy
    workloads specifically, unlike the embedding encode step)."""
    records = []
    t0 = time.time()
    batch_prompts, batch_ids = [], []

    def flush():
        if not batch_prompts:
            return
        responses = client.generate_batch(batch_prompts, max_new_tokens=max_new_tokens)
        for pid, resp in zip(batch_ids, responses):
            parsed = parse_llm_json(resp)
            parsed[id_field] = pid
            records.append(parsed)

    for i, prompt in enumerate(items_iterable):
        batch_prompts.append(prompt)
        batch_ids.append(id_values[i])
        if len(batch_prompts) >= batch_size:
            flush()
            batch_prompts, batch_ids = [], []
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            print(f"  ...{i+1:,}/{n:,} {desc} enriched ({elapsed:.1f}s, "
                  f"{rate:.1f}/s, ETA {eta/60:.1f}min)")
    flush()
    return pd.DataFrame(records)


def enrich_queries(terms, client):
    prompts = (QUERY_PROMPT_TMPL.format(query=q) for q in terms["query"].values)
    return _run_batched(prompts, len(terms), "term_id", terms["term_id"].values, client, "queries")


def enrich_items(items_subset, client):
    prompts = (
        ITEM_PROMPT_TMPL.format(title=t or "", attributes=a or "", category=c or "")
        for t, a, c in zip(items_subset["title"].values, items_subset["attributes"].values, items_subset["category"].values)
    )
    return _run_batched(prompts, len(items_subset), "item_id", items_subset["item_id"].values, client, "items")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Use MockLLMClient + a tiny subset to verify plumbing without a real model")
    parser.add_argument("--force", action="store_true", help="Run even if USE_LLM_ENRICHMENT=0")
    args = parser.parse_args()

    if not (USE_LLM_ENRICHMENT or args.force or args.dry_run):
        print("[main] USE_LLM_ENRICHMENT is off (default) -- skipping. "
              "Set TY_USE_LLM_ENRICHMENT=1 or pass --force/--dry-run to run this stage.")
        return

    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    train_pairs = pd.read_csv(f"{DATA_DIR}/training_pairs.csv")

    candidates_path = Path(f"{CACHE_DIR}/train_pairs_labeled.parquet")
    if candidates_path.exists():
        candidates = pd.read_parquet(candidates_path)
        relevant_item_ids = pd.unique(pd.concat([train_pairs["item_id"], candidates["item_id"]]))
    else:
        relevant_item_ids = train_pairs["item_id"].unique()

    items_subset = items[items["item_id"].isin(relevant_item_ids)]
    if len(items_subset) > LLM_ENRICHMENT_MAX_ITEMS:
        items_subset = items_subset.sample(LLM_ENRICHMENT_MAX_ITEMS, random_state=42)

    if args.dry_run:
        print("[main] --dry-run: using MockLLMClient on a tiny subset (50 queries / 50 items)")
        terms = terms.head(50)
        items_subset = items_subset.head(50)
        client = MockLLMClient()
    else:
        print(f"[main] loading real LLM client: {LLM_MODEL_NAME} (requires GPU + internet)")
        client = LLMClient()

    print(f"[main] enriching {len(terms):,} unique queries...")
    q_enriched = enrich_queries(terms, client)
    q_out = f"{CACHE_DIR}/query_llm_enrichment.parquet"
    q_enriched.to_parquet(q_out, index=False)
    print(f"[main] wrote {q_enriched.shape} -> {q_out}")

    print(f"[main] enriching {len(items_subset):,} items (bounded subset, not full catalog)...")
    i_enriched = enrich_items(items_subset, client)
    i_out = f"{CACHE_DIR}/item_llm_enrichment.parquet"
    i_enriched.to_parquet(i_out, index=False)
    print(f"[main] wrote {i_enriched.shape} -> {i_out}")


if __name__ == "__main__":
    main()
