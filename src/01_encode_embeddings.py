"""
Encode queries and items with:
  - Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 (768d) -> title, category, query
  - atasoglu/turkish-tiny-bert-uncased-mean-nli-stsb-tr (128d) -> attributes, query

Caches everything to cache/ as float16 .npy, aligned to item_id / term_id row order
(id arrays are saved too so downstream scripts can just np.load + zip/lookup).

Works unmodified on Kaggle (P100 / 2xT4) or locally -- see config.py for path
auto-detection. On Kaggle, if internet access is disabled for the kernel you
must attach the model weights as a dataset first, e.g.:
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
from config import CACHE_DIR, DATA_DIR, MAIN_MODEL, TINY_MODEL  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPUS = torch.cuda.device_count() if DEVICE == "cuda" else 0


def encode(model, texts, batch_size, desc):
    t0 = time.time()
    if N_GPUS > 1:
        # multi-GPU pool (Kaggle 2xT4): splits the batch across all visible GPUs
        pool = model.start_multi_process_pool()
        emb = model.encode_multi_process(texts, pool, batch_size=batch_size)
        model.stop_multi_process_pool(pool)
        emb = emb.astype(np.float16)
    else:
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
    print("Loading models...")
    main_model = SentenceTransformer(MAIN_MODEL, trust_remote_code=True, device=DEVICE)
    main_model.max_seq_length = 384
    tiny_model = SentenceTransformer(TINY_MODEL, device=DEVICE)
    tiny_model.max_seq_length = 128  # attributes/queries are short; cap for speed

    if DEVICE == "cuda" and N_GPUS <= 1:
        main_model.half()
        tiny_model.half()

    # ---------------- terms.csv ----------------
    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    queries = terms["query"].fillna("").tolist()

    q_main = encode(main_model, queries, batch_size=256, desc="query-main-768d")
    q_tiny = encode(tiny_model, queries, batch_size=512, desc="query-tiny-128d")

    np.save(f"{CACHE_DIR}/term_id.npy", terms["term_id"].values.astype(str))
    np.save(f"{CACHE_DIR}/query_emb_main.npy", q_main)
    np.save(f"{CACHE_DIR}/query_emb_tiny.npy", q_tiny)
    del q_main, q_tiny

    # ---------------- items.csv ----------------
    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    items["brand"] = items["brand"].fillna("")

    # title -> main model (encode all rows; dedup gain is small ~5%, not worth complexity)
    titles = items["title"].fillna("").tolist()
    t_main = encode(main_model, titles, batch_size=256, desc="item-title-768d")
    np.save(f"{CACHE_DIR}/item_id.npy", items["item_id"].values.astype(str))
    np.save(f"{CACHE_DIR}/item_title_emb.npy", t_main)
    del t_main, titles

    # attributes -> tiny model (encode all rows)
    attrs = items["attributes"].fillna("").tolist()
    a_tiny = encode(tiny_model, attrs, batch_size=512, desc="item-attributes-128d")
    np.save(f"{CACHE_DIR}/item_attr_emb.npy", a_tiny)
    del a_tiny, attrs

    # category -> dedup (only ~2,932 unique chains) then map back via index
    cat_series = items["category"].fillna("")
    unique_cats, cat_idx = np.unique(cat_series.values, return_inverse=True)
    c_main = encode(main_model, unique_cats.tolist(), batch_size=256, desc="unique-category-768d")
    np.save(f"{CACHE_DIR}/category_unique_emb.npy", c_main)
    np.save(f"{CACHE_DIR}/item_category_idx.npy", cat_idx.astype(np.int32))
    np.save(f"{CACHE_DIR}/category_unique_strings.npy", unique_cats)

    print("Done. All embeddings cached to", CACHE_DIR)


if __name__ == "__main__":
    main()
