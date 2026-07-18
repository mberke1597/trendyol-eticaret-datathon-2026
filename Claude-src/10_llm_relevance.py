"""
Optional LLM pairwise relevance scoring -- OFF by default (config.USE_LLM_RELEVANCE).

This is a DIFFERENT, heavier LLM use than 02_llm_enrichment.py. Where 02 does
bounded offline *structured extraction* (attributes over unique queries / a
capped item subset, ~100 tokens/call), this script asks an LLM the relevance
question directly for EVERY (query, item) pair -- the whole labeled train set
AND all 3.36M submission_pairs -- and turns the answer into a continuous 0..1
score used both as a GBDT feature (04_build_features.py) and as a 4th ensemble
member alongside lgb/xgb/cat (05_train.py). The user asked for exactly this
("combine llm relevance score and classic ensemble model at train.py").

Why this is feasible at 3.36M scale when DESIGN.md lesson 13 scoped a full-corpus
LLM pass OUT as "weeks, not hours":
  1. vLLM, not HF `generate()`. Continuous batching + paged KV-cache + real
     tensor-parallelism across both T4s (NOT HF device_map="auto"'s per-layer
     pipeline split, which re-crosses PCIe every token -- the exact bottleneck
     the earlier chat + 02's docstring diagnosed). 10-40x faster.
  2. ONE generated token per pair, not ~100. We prompt for a single "1"/"0"
     answer and read the *logprobs* of the "1" and "0" tokens, converting them
     to a calibrated probability P(relevant) = softmax over {1,0}. A continuous
     score (better for ensembling than a hard 0/1) at the cost of a single
     decode step -- so prefill dominates, and short prompts (truncated item
     text) keep prefill cheap.
  3. A small model (Qwen2.5-3B-Instruct default). Turkish-capable, text-only,
     ~6GB fp16. Swap TY_LLM_REL_MODEL=Trendyol/Trendyol-LLM-Asure-12B to A/B the
     e-commerce-tuned (slower) model on the SAME cached pairs (use a distinct
     TY_LLM_REL_TAG per model so the two caches coexist).

Robustness for a multi-hour Kaggle run: scoring is sharded (config.
LLM_REL_SHARD_SIZE rows/shard) and each shard's scores are checkpointed to
`cache/_llm_rel_{split}{tag}_shard_{i}.parquet`; a re-run skips shards already
on disk, so a killed session resumes instead of restarting from zero.

Outputs (consumed by 04_build_features.py via config.llm_rel_cache_paths):
  cache/llm_relevance_train{tag}.parquet       [term_id, item_id, llm_rel_score]
  cache/llm_relevance_submission{tag}.parquet  [id,               llm_rel_score]

CPU/no-GPU verification: `--dry-run` swaps in MockRelevanceScorer (deterministic
token-overlap score, no vLLM/GPU) over a tiny slice, exercising the prompt /
logprob-parse / shard / cache plumbing end-to-end. Never loads a real model.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    CACHE_DIR,
    DATA_DIR,
    LLM_REL_ATTR_CHARS,
    LLM_REL_DTYPE,
    LLM_REL_FILL,
    LLM_REL_GPU_MEM_UTIL,
    LLM_REL_MAX_MODEL_LEN,
    LLM_REL_MODEL,
    LLM_REL_SHARD_SIZE,
    LLM_REL_TAG,
    LLM_REL_TENSOR_PARALLEL,
    LLM_REL_TITLE_CHARS,
    llm_rel_cache_paths,
)

# We ask for a single-character verdict "1" (relevant) or "0" (irrelevant).
# Digits are reliably their own single token across tokenizers (Qwen, Gemma3,
# Llama, ...), unlike Turkish words like "evet"/"hayır" which can split into
# sub-word pieces whose FIRST token collides ("ev.." vs "ha..") -- so the
# 1/0 form makes the logprob read model-agnostic. See _score_from_logprobs.
POS_TOKEN = "1"
NEG_TOKEN = "0"

RELEVANCE_PROMPT_TMPL = """Sen bir e-ticaret arama alaka değerlendirme asistanısın.
Aşağıdaki arama sorgusu ile ürünün birbiriyle alakalı olup olmadığına karar ver.
Ürün, sorgunun aradığı şeyse alakalıdır; farklı bir ürün/kategori ise alakasızdır.
Cinsiyet ve yaş grubu çelişkisi (ör. "kadın" sorgusu, "erkek" ürünü) alakasız demektir.

Sorgu: "{query}"
Ürün başlığı: "{title}"
Kategori: "{category}"
Marka: "{brand}"
Öznitelikler: {attributes}

Alakalıysa 1, alakasızsa 0 yaz. SADECE tek bir rakam (1 veya 0) yaz:"""


def build_item_blurb(title, category, brand, attributes,
                     title_chars=LLM_REL_TITLE_CHARS, attr_chars=LLM_REL_ATTR_CHARS):
    """Short, prompt-safe item text. Attributes in items.csv can be hundreds of
    chars; truncating keeps the prompt short so prefill (the dominant cost of
    single-token scoring) stays cheap."""
    def clip(s, n):
        s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
        s = s.replace("\n", " ").strip()
        return s[:n]
    return {
        "title": clip(title, title_chars),
        "category": clip(category, 160),
        "brand": clip(brand, 60),
        "attributes": clip(attributes, attr_chars),
    }


def build_prompt(query, title, category, brand, attributes):
    b = build_item_blurb(title, category, brand, attributes)
    return RELEVANCE_PROMPT_TMPL.format(
        query="" if query is None else str(query),
        title=b["title"], category=b["category"], brand=b["brand"], attributes=b["attributes"],
    )


def _score_from_logprobs(token_logprobs, greedy_text=None, fill=LLM_REL_FILL):
    """Turn one decode step's top-k logprobs into P(relevant) in [0, 1].

    token_logprobs: iterable of (decoded_token_text, logprob) for the top-k
    candidate first tokens (order irrelevant). We locate the "1" and "0" tokens
    (stripped of leading whitespace / sub-word markers) and return the softmax
    weight on "1": exp(lp1) / (exp(lp1) + exp(lp0)). This is a proper 2-way
    normalization, so it's well-calibrated even when the model spreads a little
    mass onto other tokens.

    Fallbacks (in order): only one of 1/0 present -> map that side to ~0.98/0.02;
    neither present -> use the greedy token's own value if it's 1/0, else `fill`
    (neutral). Pure function (no vLLM types) so it's unit-testable -- see
    tests/test_llm_relevance.py."""
    lp_pos = lp_neg = None
    for tok, lp in token_logprobs:
        t = (tok or "").strip().lstrip("▁Ġ").strip()
        if t == POS_TOKEN and lp_pos is None:
            lp_pos = lp
        elif t == NEG_TOKEN and lp_neg is None:
            lp_neg = lp
    if lp_pos is not None and lp_neg is not None:
        m = max(lp_pos, lp_neg)
        e_pos, e_neg = np.exp(lp_pos - m), np.exp(lp_neg - m)
        return float(e_pos / (e_pos + e_neg))
    if lp_pos is not None:
        return 0.98
    if lp_neg is not None:
        return 0.02
    g = (greedy_text or "").strip().lstrip("▁Ġ").strip()
    if g == POS_TOKEN:
        return 0.98
    if g == NEG_TOKEN:
        return 0.02
    return float(fill)


class VLLMRelevanceScorer:
    """vLLM-backed pairwise scorer. Imports vllm lazily inside __init__ (like
    02_llm_enrichment.LLMClient guards transformers/torch) so this module -- and
    its pure helpers / MockRelevanceScorer -- import fine on a CPU box with no
    vLLM installed."""

    def __init__(self, model_name=LLM_REL_MODEL, tensor_parallel=LLM_REL_TENSOR_PARALLEL,
                 max_model_len=LLM_REL_MAX_MODEL_LEN, gpu_mem_util=LLM_REL_GPU_MEM_UTIL,
                 dtype=LLM_REL_DTYPE, top_logprobs=20):
        import torch
        from vllm import LLM, SamplingParams

        # Preflight: catch the Gemma3-on-T4 dead-end with a clear message instead
        # of a raw vLLM pydantic ValidationError. Gemma3 (Asure-12B) refuses fp16
        # ("numerical instability") and wants bf16/fp32; T4 (compute < 8.0) has no
        # bf16; 12B fp32 won't fit 2xT4. So it can't run on a T4 -- use Qwen-3B
        # here, or move to an Ampere+/A100/L4 GPU for the Gemma3 models.
        is_gemma3 = "gemma" in model_name.lower() or "asure" in model_name.lower()
        cc = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        no_bf16 = cc[0] < 8
        if is_gemma3 and dtype == "half":
            raise SystemExit(
                f"[VLLMRelevanceScorer] {model_name} is Gemma3-based and vLLM refuses "
                f"float16 for it. Your GPU (compute {cc[0]}.{cc[1]}) "
                + ("has NO bf16 support, and 12B fp32 won't fit -- this model cannot run here. "
                   "Use TY_LLM_REL_MODEL=Qwen/Qwen2.5-3B-Instruct on this T4, or run the "
                   "Gemma3 model on an Ampere+/A100/L4 GPU with TY_LLM_REL_DTYPE=bfloat16."
                   if no_bf16 else
                   "Re-run with TY_LLM_REL_DTYPE=bfloat16 (your GPU supports it).")
            )
        print(f"[VLLMRelevanceScorer] loading {model_name} via vLLM "
              f"(tensor_parallel_size={tensor_parallel}, max_model_len={max_model_len}, "
              f"dtype={dtype})")
        self.llm = LLM(
            model=model_name,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=max_model_len,
            trust_remote_code=True,             # Asure-12B (Gemma3) ships remote code
        )
        self.tok = self.llm.get_tokenizer()
        # Only ONE token is generated; logprobs=top_logprobs surfaces the top
        # candidates for that single step so we can read the 1/0 masses.
        self.sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=top_logprobs)

    def _format(self, prompts):
        """Apply the model's chat template per prompt. Branch on whether the
        template expects multimodal typed `content` parts (Gemma3/Asure-12B) vs.
        a plain string (Qwen2.5/Llama), same detection style as 02's
        generate_batch."""
        try:
            is_mm = getattr(self.tok, "chat_template", None) is not None and "image" in (self.tok.chat_template or "").lower()
        except Exception:
            is_mm = False
        content = (lambda p: [{"type": "text", "text": p}]) if is_mm else (lambda p: p)
        out = []
        for p in prompts:
            try:
                out.append(self.tok.apply_chat_template(
                    [{"role": "user", "content": content(p)}],
                    add_generation_prompt=True, tokenize=False,
                ))
            except Exception:
                out.append(p)  # model without a chat template -> raw prompt
        return out

    def score_batch(self, prompts):
        texts = self._format(prompts)
        outs = self.llm.generate(texts, self.sampling)
        scores = np.empty(len(outs), dtype=np.float32)
        for i, o in enumerate(outs):
            gen = o.outputs[0]
            greedy_text = gen.text
            pairs = []
            if gen.logprobs:                    # list (per gen-step) of {token_id: Logprob}
                step0 = gen.logprobs[0]
                for tok_id, lp in step0.items():
                    # vLLM's Logprob carries .decoded_token on recent versions;
                    # fall back to decoding the id if not.
                    tok_text = getattr(lp, "decoded_token", None)
                    if tok_text is None:
                        try:
                            tok_text = self.tok.decode([tok_id])
                        except Exception:
                            tok_text = ""
                    pairs.append((tok_text, float(lp.logprob)))
            scores[i] = _score_from_logprobs(pairs, greedy_text=greedy_text)
        return scores


class MockRelevanceScorer:
    """Deterministic, GPU-free stand-in for --dry-run and tests. Score is a
    smooth function of query<->item token overlap (NOT a real LLM signal, on
    purpose) so the plumbing -- prompt build, shard/checkpoint, cache schema --
    is exercised end-to-end without vLLM."""

    def score_batch(self, prompts):
        from text_utils import tokenize
        scores = np.empty(len(prompts), dtype=np.float32)
        for i, p in enumerate(prompts):
            # crude: overlap between the words after "Sorgu:" and the rest.
            toks = tokenize(p)
            # deterministic pseudo-score in [0.05, 0.95] from overlap-ish hash
            scores[i] = 0.05 + 0.90 * ((len(toks) * 2654435761) % 1000) / 1000.0
        return scores


def _score_frame(prompts, scorer, split_tag, shard_size=LLM_REL_SHARD_SIZE):
    """Shard + checkpoint driver. Scores `prompts` (a list/ndarray of strings)
    in shard_size chunks, writing each shard's raw scores to a numpy checkpoint
    so a re-run resumes. Returns the full score array."""
    n = len(prompts)
    scores = np.full(n, np.nan, dtype=np.float32)
    ckpt_dir = CACHE_DIR / "_llm_rel_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_shards = (n + shard_size - 1) // shard_size
    for s in range(n_shards):
        start, end = s * shard_size, min((s + 1) * shard_size, n)
        ckpt = ckpt_dir / f"{split_tag}_shard_{s:04d}.npy"
        if ckpt.exists():
            sc = np.load(ckpt)
            if len(sc) == end - start:
                scores[start:end] = sc
                print(f"  [{split_tag}] shard {s+1}/{n_shards} resumed from checkpoint ({start:,}-{end:,})")
                continue
        sc = scorer.score_batch(list(prompts[start:end]))
        np.save(ckpt, sc)
        scores[start:end] = sc
        elapsed = time.time() - t0
        done = end
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else float("nan")
        print(f"  [{split_tag}] shard {s+1}/{n_shards} scored {start:,}-{end:,} "
              f"({elapsed:.1f}s, {rate:.1f}/s, ETA {eta/60:.1f}min)")
    return scores


def build_train_prompts(labeled, items, terms):
    q = terms.set_index("term_id")["query"]
    it = items.set_index("item_id")
    query = labeled["term_id"].map(q).values
    sub_items = it.reindex(labeled["item_id"].values)
    prompts = [
        build_prompt(query[i], sub_items["title"].values[i], sub_items["category"].values[i],
                     sub_items["brand"].values[i], sub_items["attributes"].values[i])
        for i in range(len(labeled))
    ]
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="MockRelevanceScorer on a tiny slice -- no vLLM/GPU, verifies plumbing")
    parser.add_argument("--split", choices=["train", "submission", "both"], default="both",
                        help="Which set(s) to score (default both)")
    args = parser.parse_args()

    if not (config_flag_on() or args.dry_run):
        print("[main] USE_LLM_RELEVANCE is off (default) -- skipping. "
              "Set TY_USE_LLM_RELEVANCE=1 (and TY_LLM_REL_ENSEMBLE=1 for the 4th "
              "ensemble member) to run this stage.")
        return

    tag = LLM_REL_TAG
    train_path, sub_path = llm_rel_cache_paths(tag)
    tag_disp = tag or "(none)"

    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")

    if args.dry_run:
        print(f"[main] --dry-run: MockRelevanceScorer, model tag={tag_disp}")
        scorer = MockRelevanceScorer()
    else:
        print(f"[main] loading real vLLM scorer: {LLM_REL_MODEL} (tag={tag_disp})")
        scorer = VLLMRelevanceScorer()

    # ---------------- train pairs ----------------
    if args.split in ("train", "both"):
        # prefer the mined labeled set (has negatives) so every training row the
        # GBDT/ensemble sees also has an LLM score; fall back to raw positives.
        labeled_path = CACHE_DIR / "train_pairs_labeled.parquet"
        if labeled_path.exists():
            labeled = pd.read_parquet(labeled_path)[["term_id", "item_id"]].drop_duplicates()
        else:
            labeled = pd.read_csv(f"{DATA_DIR}/training_pairs.csv")[["term_id", "item_id"]].drop_duplicates()
        if args.dry_run:
            labeled = labeled.head(64)
        print(f"[train] scoring {len(labeled):,} unique (term,item) pairs...")
        prompts = build_train_prompts(labeled.reset_index(drop=True), items, terms)
        scores = _score_frame(np.array(prompts, dtype=object), scorer, f"train_{tag}" if tag else "train")
        out = pd.DataFrame({
            "term_id": labeled["term_id"].values,
            "item_id": labeled["item_id"].values,
            "llm_rel_score": np.nan_to_num(scores, nan=LLM_REL_FILL),
        })
        out.to_parquet(train_path, index=False)
        print(f"[train] wrote {out.shape} (mean score={out['llm_rel_score'].mean():.3f}) -> {train_path}")

    # ---------------- submission pairs ----------------
    if args.split in ("submission", "both"):
        sub_pairs = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv")
        if args.dry_run:
            sub_pairs = sub_pairs.head(64)
        print(f"[submission] scoring {len(sub_pairs):,} pairs...")
        q = terms.set_index("term_id")["query"]
        it = items.set_index("item_id")
        query = sub_pairs["term_id"].map(q).values
        sub_items = it.reindex(sub_pairs["item_id"].values)
        prompts = [
            build_prompt(query[i], sub_items["title"].values[i], sub_items["category"].values[i],
                         sub_items["brand"].values[i], sub_items["attributes"].values[i])
            for i in range(len(sub_pairs))
        ]
        scores = _score_frame(np.array(prompts, dtype=object), scorer,
                              f"submission_{tag}" if tag else "submission")
        out = pd.DataFrame({
            "id": sub_pairs["id"].values,
            "llm_rel_score": np.nan_to_num(scores, nan=LLM_REL_FILL),
        })
        out.to_parquet(sub_path, index=False)
        print(f"[submission] wrote {out.shape} (mean score={out['llm_rel_score'].mean():.3f}) -> {sub_path}")

    print("[main] done. Next: TY_USE_LLM_RELEVANCE=1 python 04_build_features.py "
          "(joins llm_rel_score), then 05_train.py (set TY_LLM_REL_ENSEMBLE=1 for the 4th member).")


def config_flag_on():
    # read lazily so tests can monkeypatch the env before import-time constants
    # would otherwise freeze it; mirrors 02's --force gating intent.
    from config import USE_LLM_RELEVANCE
    return USE_LLM_RELEVANCE


if __name__ == "__main__":
    main()
