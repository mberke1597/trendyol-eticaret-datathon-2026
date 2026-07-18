"""
Encode queries and items with:
  - Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 (768d) -> title, category, query
  - atasoglu/turkish-tiny-bert-uncased-mean-nli-stsb-tr (128d) -> attributes, query

Caches to Claude-src/cache/ as float16 .npy, aligned to item_id / term_id row order.

Single-device encoding ONLY (2026-07-03 fix): the multi-GPU `encode_multi_process`
path (spawning one worker process per GPU) is INCOMPATIBLE with
TY-ecomm-embed's custom `trust_remote_code=True` implementation (Alibaba-NLP
"new-impl" architecture with a cached RoPE table) -- on a real Kaggle T4 x2
run this produced a real, reproducible crash:
    torch.AcceleratorError: CUDA error: device-side assert triggered
    .../modeling.py", line 400, in forward -> token_type_ids = position_ids.mul(0)
The cached RoPE buffer gets created against one CUDA device but the worker
process for the *other* GPU ends up indexing into it, producing out-of-bounds
device-side asserts. `sentence-transformers` itself also flags
`encode_multi_process` as deprecated in recent versions. Given the crash is
inside third-party remote code (not something we can patch), the fix is to
never use the multi-process path for this model -- always encode on a single
CUDA device (cuda:0). This does not use the second T4 for embedding, but this
stage is not the pipeline's bottleneck; 05_train.py's GBDT training is where
multi-GPU actually matters (XGBoost/CatBoost device="cuda", handled there).

REVISED 2026-07-04: the SAME "index out of bounds" device-side assert
reappeared on a fresh Kaggle session even with single-device encoding (no
multi-process involved at all), so the 2026-07-03 diagnosis above was
incomplete -- multi-GPU RoPE contamination was A cause, not THE cause. The
run's log showed the "new-impl" remote code (configuration.py/modeling.py)
being freshly re-downloaded from https://huggingface.co/Alibaba-NLP/new-impl
with no pinned revision, i.e. every fresh session silently picks up whatever
is on that repo's default branch at run time. GTE-style "new-impl" models are
known to take a different, less-tested internal code path (unpadded inputs +
memory-efficient/flash attention) when flash_attn isn't installed (it isn't,
by default, on Kaggle), which is a documented source of exactly this
"index out of bounds in the embeddings layer" failure for this model family.
Fix: force the safe/standard eager attention path and disable the
unpad/memory-efficient-attention code path explicitly via model_kwargs at
load time, so behavior doesn't depend on whatever the upstream repo's HEAD
happens to do this week.

On Kaggle, if internet access is disabled for the kernel you must attach the
model weights as a dataset first, e.g.:
    hf download Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 --local-dir ./ty-embed
and point MAIN_MODEL at that local directory instead of the hub id.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CACHE_DIR, DATA_DIR, FINETUNED_MODEL_DIR, MAIN_MODEL, TINY_MODEL  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPUS = torch.cuda.device_count() if DEVICE == "cuda" else 0

# See 00_finetune_embeddings.py: if a fine-tuned checkpoint exists (either
# TY_FINETUNED_MODEL_DIR points at one, or config.FINETUNE_OUTPUT_DIR has one
# from a previous run), use it INSTEAD of the base MAIN_MODEL hub id -- this is
# the only place that decision needs to be made; everything downstream just
# consumes whatever embeddings land in CACHE_DIR either way.
_finetuned_candidate = Path(FINETUNED_MODEL_DIR) if FINETUNED_MODEL_DIR else None
if _finetuned_candidate is None:
    from config import FINETUNE_OUTPUT_DIR

    if (FINETUNE_OUTPUT_DIR / "config.json").exists():
        _finetuned_candidate = FINETUNE_OUTPUT_DIR
ACTIVE_MAIN_MODEL = str(_finetuned_candidate) if _finetuned_candidate and _finetuned_candidate.exists() else MAIN_MODEL


def encode(model, texts, batch_size, desc):
    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device=DEVICE,
    ).astype(np.float16)
    print(f"[{desc}] {len(texts):,} rows encoded in {time.time()-t0:.1f}s -> shape {emb.shape}")
    return emb


def main():
    print(f"Device={DEVICE} n_gpus={N_GPUS} DATA_DIR={DATA_DIR} CACHE_DIR={CACHE_DIR}")
    print(f"Loading models... main={ACTIVE_MAIN_MODEL} "
          f"({'FINE-TUNED local checkpoint' if ACTIVE_MAIN_MODEL != MAIN_MODEL else 'base hub model'})")
    main_model = SentenceTransformer(
        ACTIVE_MAIN_MODEL,
        trust_remote_code=True,
        device=DEVICE,
        model_kwargs={"attn_implementation": "eager"},
        config_kwargs={
            "unpad_inputs": False,
            "use_memory_efficient_attention": False,
        },
    )
    main_model.max_seq_length = 384
    tiny_model = SentenceTransformer(TINY_MODEL, device=DEVICE)
    tiny_model.max_seq_length = 128

    if DEVICE == "cuda":
        main_model.half()
        tiny_model.half()

    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    queries = terms["query"].fillna("").tolist()

    q_main = encode(main_model, queries, batch_size=256, desc="query-main-768d")
    q_tiny = encode(tiny_model, queries, batch_size=512, desc="query-tiny-128d")

    np.save(f"{CACHE_DIR}/term_id.npy", terms["term_id"].values.astype(str))
    np.save(f"{CACHE_DIR}/query_emb_main.npy", q_main)
    np.save(f"{CACHE_DIR}/query_emb_tiny.npy", q_tiny)
    del q_main, q_tiny

    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    items["brand"] = items["brand"].fillna("")

    titles = items["title"].fillna("").tolist()
    t_main = encode(main_model, titles, batch_size=256, desc="item-title-768d")
    np.save(f"{CACHE_DIR}/item_id.npy", items["item_id"].values.astype(str))
    np.save(f"{CACHE_DIR}/item_title_emb.npy", t_main)
    del t_main, titles

    attrs = items["attributes"].fillna("").tolist()
    a_tiny = encode(tiny_model, attrs, batch_size=512, desc="item-attributes-128d")
    np.save(f"{CACHE_DIR}/item_attr_emb.npy", a_tiny)
    del a_tiny, attrs

    cat_series = items["category"].fillna("")
    unique_cats, cat_idx = np.unique(cat_series.values, return_inverse=True)
    c_main = encode(main_model, unique_cats.tolist(), batch_size=256, desc="unique-category-768d")
    np.save(f"{CACHE_DIR}/category_unique_emb.npy", c_main)
    np.save(f"{CACHE_DIR}/item_category_idx.npy", cat_idx.astype(np.int32))
    np.save(f"{CACHE_DIR}/category_unique_strings.npy", unique_cats)

    print("Done. All embeddings cached to", CACHE_DIR)


if __name__ == "__main__":
    main()
