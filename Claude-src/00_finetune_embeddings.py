"""
OPTIONAL contrastive fine-tuning of MAIN_MODEL on this competition's own
query -> clicked-item pairs -- OFF by default (01_encode_embeddings.py only
uses it if config.FINETUNED_MODEL_DIR / a prior run's output exists, see that
file's ACTIVE_MAIN_MODEL logic).

Why this exists: Trendyol/TY-ecomm-embed-multilingual-base-v1.2.0 is a strong
general-purpose Turkish e-commerce embedding model, but it was never trained on
THIS catalog's specific vocabulary, brand names, or (most importantly) this
competition's actual query -> click behavior. feature_importance.csv's own
output consistently ranks sim_title/sim_mean_title_cat among the most
important features for the trained GBDT -- improving the embedding model
underneath those two features has more leverage than almost any other single
change available, precisely because so much else in the pipeline already
depends on it.

Method: MultipleNegativesRankingLoss (the standard sentence-transformers
approach for retrieval fine-tuning) on (query, positive item title) pairs from
training_pairs.csv. This loss uses every OTHER pair's positive in the same
batch as an implicit negative for a given query -- no explicit negative
mining needed for the loss itself, unlike the GBDT's negative sampling.

Cold-start discipline, same reason as GroupKFold(term_id) everywhere else in
this pipeline: a held-out FINETUNE_VAL_TERM_FRAC of TERM_IDs (not rows) is
excluded from training and used to build an InformationRetrievalEvaluator,
so overfitting to the training terms' specific vocabulary is visible BEFORE
this checkpoint ever reaches 01_encode_embeddings.py and gets baked into
every downstream feature.

Usage:
    python 00_finetune_embeddings.py
    # writes Claude-src/finetuned_embed/ (a full SentenceTransformer checkpoint)
    python 01_encode_embeddings.py
    # automatically picks up finetuned_embed/ instead of the hub MAIN_MODEL
    # (see that file's ACTIVE_MAIN_MODEL logic) -- everything downstream is
    # unchanged, it just consumes better embeddings.

Cost: one epoch over ~250k positive pairs at batch_size=64 is a few thousand
optimizer steps -- expect tens of minutes on a single T4, not hours. Compare
this to the ~20-30 GPU-minutes 01_encode_embeddings.py itself takes; this is
a meaningful but not dominant addition to the total pipeline wall-clock.
"""
import os
import sys
import time
from pathlib import Path

# Must be set BEFORE sentence-transformers/transformers touch any Trainer
# object: model.fit() below uses transformers' Trainer internally, which
# auto-detects wandb if it's installed and calls its interactive "create an
# account? (1)/(2)/(3)" prompt on first use -- fatal in a non-interactive
# `!python script.py` Kaggle cell (input() has no stdin to read, so the run
# just hangs until the cell is manually stopped). Disable it unconditionally.
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("WANDB_MODE", "disabled")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    DATA_DIR,
    FINETUNE_BATCH_SIZE,
    FINETUNE_EPOCHS,
    FINETUNE_LR,
    FINETUNE_OUTPUT_DIR,
    FINETUNE_VAL_TERM_FRAC,
    MAIN_MODEL,
    RANDOM_SEED,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Distractor pool size for the held-out IR evaluator -- large enough that
# "the model just memorized there are only 3 items" isn't possible, small
# enough that evaluation doesn't itself become the bottleneck (see module
# docstring: the point is a fast overfitting sanity check, not a full-catalog
# benchmark -- 07_predict.py's real submission-set performance is the actual
# test, this is a canary that fires DURING training instead of after).
EVAL_DISTRACTOR_POOL_SIZE = 20_000


def build_train_examples(train_pairs, terms, items, train_term_ids):
    query_of_term = terms.set_index("term_id")["query"]
    title_of_item = items.set_index("item_id")["title"].fillna("")
    sub = train_pairs[train_pairs["term_id"].isin(train_term_ids)]
    examples = []
    for term_id, item_id in zip(sub["term_id"].values, sub["item_id"].values):
        q = query_of_term.get(term_id)
        t = title_of_item.get(item_id)
        if not isinstance(q, str) or not q or not isinstance(t, str) or not t:
            continue
        examples.append(InputExample(texts=[q, t]))
    return examples


def build_ir_evaluator(train_pairs, terms, items, val_term_ids, rng):
    """queries: held-out term_id -> query text. corpus: relevant items for those
    terms + a random distractor pool (so 'rank the positive above everything
    else' is a real retrieval task, not a 3-way multiple choice)."""
    query_of_term = terms.set_index("term_id")["query"]
    title_of_item = items.set_index("item_id")["title"].fillna("")

    val_sub = train_pairs[train_pairs["term_id"].isin(val_term_ids)]
    queries = {}
    relevant_docs = {}
    corpus = {}
    for term_id, item_id in zip(val_sub["term_id"].values, val_sub["item_id"].values):
        q = query_of_term.get(term_id)
        t = title_of_item.get(item_id)
        if not isinstance(q, str) or not q or not isinstance(t, str) or not t:
            continue
        term_key = str(term_id)
        item_key = str(item_id)
        queries[term_key] = q
        relevant_docs.setdefault(term_key, set()).add(item_key)
        corpus[item_key] = t

    all_item_ids = items["item_id"].values
    distractor_ids = rng.choice(all_item_ids, size=min(EVAL_DISTRACTOR_POOL_SIZE, len(all_item_ids)), replace=False)
    for item_id in distractor_ids:
        item_key = str(item_id)
        if item_key in corpus:
            continue
        t = title_of_item.get(item_id)
        if isinstance(t, str) and t:
            corpus[item_key] = t

    return InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs,
        name="held_out_terms", show_progress_bar=False,
        precision_recall_at_k=[10], mrr_at_k=[10], accuracy_at_k=[1, 10],
    )


def _scalar_score(evaluator, result):
    """SentenceEvaluator.__call__ return type has varied across
    sentence-transformers versions (a bare float vs. a dict of named metrics).
    Handle both so the before/after comparison below doesn't break on a minor
    version difference: prefer evaluator.primary_metric if the result is a
    dict and that key is present, else the max of the dict's values, else the
    result itself (already a float)."""
    if isinstance(result, dict):
        key = getattr(evaluator, "primary_metric", None)
        if key and key in result:
            return float(result[key])
        return float(max(result.values()))
    return float(result)


def main():
    t_start = time.time()
    rng = np.random.default_rng(RANDOM_SEED)
    print(f"Device={DEVICE} base_model={MAIN_MODEL} -> output={FINETUNE_OUTPUT_DIR}")

    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    train_pairs = pd.read_csv(f"{DATA_DIR}/training_pairs.csv")

    unique_terms = train_pairs["term_id"].unique()
    rng.shuffle(unique_terms)
    n_val = max(1, int(len(unique_terms) * FINETUNE_VAL_TERM_FRAC))
    val_term_ids = set(unique_terms[:n_val])
    train_term_ids = set(unique_terms[n_val:])
    print(f"[main] {len(train_term_ids):,} train terms / {len(val_term_ids):,} held-out terms "
          f"(cold-start split, same rationale as GroupKFold(term_id) elsewhere)")

    train_examples = build_train_examples(train_pairs, terms, items, train_term_ids)
    print(f"[main] {len(train_examples):,} (query, positive title) training pairs")

    print(f"[main] loading base model {MAIN_MODEL} (trust_remote_code=True)...")
    # See 01_encode_embeddings.py's 2026-07-04 docstring note: without pinning
    # attention behavior, this "new-impl" (Alibaba-NLP) architecture can pick
    # an unpad/memory-efficient-attention code path that crashes with a CUDA
    # "index out of bounds" device-side assert when flash_attn isn't
    # installed (it isn't on Kaggle by default). Force the safe eager path.
    model = SentenceTransformer(
        MAIN_MODEL,
        trust_remote_code=True,
        device=DEVICE,
        model_kwargs={"attn_implementation": "eager"},
        config_kwargs={
            "unpad_inputs": False,
            "use_memory_efficient_attention": False,
        },
    )
    model.max_seq_length = 384

    evaluator = build_ir_evaluator(train_pairs, terms, items, val_term_ids, rng)
    print("[main] evaluating BASE (pre-fine-tune) model on held-out terms...")
    base_result = evaluator(model)
    base_score = _scalar_score(evaluator, base_result)
    print(f"[main] base scores: {base_result} (scalar={base_score:.4f})")

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=FINETUNE_BATCH_SIZE)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    warmup_steps = int(len(train_dataloader) * FINETUNE_EPOCHS * 0.1)

    print(f"[main] fine-tuning {FINETUNE_EPOCHS} epoch(s), batch_size={FINETUNE_BATCH_SIZE}, "
          f"lr={FINETUNE_LR}, warmup_steps={warmup_steps}...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=FINETUNE_EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": FINETUNE_LR},
        evaluator=evaluator,
        evaluation_steps=max(1, len(train_dataloader) // 4),
        output_path=str(FINETUNE_OUTPUT_DIR),
        save_best_model=True,
        show_progress_bar=True,
    )

    print("[main] evaluating FINE-TUNED model on the same held-out terms...")
    final_result = evaluator(model)
    final_score = _scalar_score(evaluator, final_result)
    print(f"[main] base scores:       {base_result} (scalar={base_score:.4f})")
    print(f"[main] fine-tuned scores: {final_result} (scalar={final_score:.4f})")
    if final_score <= base_score:
        print("[main] WARNING: fine-tuned model did NOT beat the base model on the held-out "
              "IR evaluator -- inspect before pointing 01_encode_embeddings.py at this "
              "checkpoint (it will otherwise silently prefer it over the base model). "
              "Consider fewer epochs, a lower learning rate, or more training pairs.")
    print(f"[main] saved best checkpoint -> {FINETUNE_OUTPUT_DIR}")
    print(f"[main] total time {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
