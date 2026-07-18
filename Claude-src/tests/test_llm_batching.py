"""
Tests for the 2026-07-04 LLM-batching hardening (see 02_llm_enrichment.py's
LLMClient docstring): the original one-prompt-at-a-time design would need tens
of thousands of sequential model.generate() calls for a real Kaggle run --
these tests lock in that enrich_queries/enrich_items/rescore_band now batch
through client.generate_batch() and produce correct, complete output
regardless of how the row count divides against the batch size (exact
multiple, with remainder, smaller than one batch).

Uses only MockLLMClient -- no transformers/torch/GPU needed.
"""
import importlib

import numpy as np
import pandas as pd
import pytest

llm_mod = pytest.importorskip("importlib").import_module("02_llm_enrichment")
rescore_mod = importlib.import_module("06_rescore_uncertain_band")


@pytest.mark.parametrize("n_terms,batch_size", [(37, 16), (16, 16), (5, 16), (1, 16)])
def test_enrich_queries_covers_every_row_regardless_of_batch_boundary(n_terms, batch_size):
    """37/16=2 remainder 5, 16/16=exact, 5/16=partial single batch, 1/16=trivial --
    covers every remainder case relative to LLM_BATCH_SIZE."""
    client = llm_mod.MockLLMClient()
    terms = pd.DataFrame({
        "term_id": [f"T{i}" for i in range(n_terms)],
        "query": ["kadın ayakkabı"] * n_terms,
    })
    out = llm_mod._run_batched(
        (llm_mod.QUERY_PROMPT_TMPL.format(query=q) for q in terms["query"].values),
        len(terms), "term_id", terms["term_id"].values, client, "queries", batch_size=batch_size,
    )
    assert len(out) == n_terms
    assert set(out["term_id"]) == set(terms["term_id"])
    # MockLLMClient deterministically extracts gender from the query text
    assert (out["gender"] == "kadın").all()


def test_enrich_items_produces_one_row_per_item():
    client = llm_mod.MockLLMClient()
    items = pd.DataFrame({
        "item_id": [f"I{i}" for i in range(9)],
        "title": ["Erkek Ayakkabi"] * 9,
        "attributes": ["renk: mavi"] * 9,
        "category": ["ayakkabi/spor"] * 9,
    })
    out = llm_mod.enrich_items(items, client)
    assert len(out) == 9
    assert set(out["item_id"]) == set(items["item_id"])


@pytest.mark.parametrize("n_rows,batch_size", [(20, 7), (14, 7), (3, 7)])
def test_rescore_band_covers_every_row_and_returns_binary(n_rows, batch_size):
    client = llm_mod.MockLLMClient()
    df_band = pd.DataFrame({
        "query": ["kadın ayakkabı"] * n_rows,
        "title": ["Erkek Spor Ayakkabi"] * n_rows,
        "category": ["ayakkabi/spor"] * n_rows,
        "brand": ["Nike"] * n_rows,
        "attributes": ["renk: siyah"] * n_rows,
        "model_proba": np.linspace(0.35, 0.65, n_rows),
    })
    out = rescore_mod.rescore_band(df_band, client, batch_size=batch_size)
    assert len(out) == n_rows
    assert set(np.unique(out)) <= {0, 1}


def test_rescore_band_falls_back_to_threshold_when_llm_parse_fails():
    """MockLLMClient's generic (no-query-match) branch never returns a
    "relevant" key -- rescore_band must fall back to the model's own
    threshold-based prediction rather than leaving the row unpredicted."""

    class NoRelevantKeyClient:
        def generate_batch(self, prompts, max_new_tokens=200):
            return ["{}" for _ in prompts]  # valid JSON, but no "relevant" key

    df_band = pd.DataFrame({
        "query": ["x"], "title": ["y"], "category": ["z"], "brand": ["b"], "attributes": ["a"],
        "model_proba": [0.9],  # well above mid-band -> fallback should predict 1
    })
    out = rescore_mod.rescore_band(df_band, NoRelevantKeyClient(), batch_size=8)
    assert out[0] == 1


def test_rescore_band_blend_falls_back_to_model_proba_when_llm_parse_fails():
    """Regression test for the 2026-07-08 'blend' mode (config.RESCORE_MODE):
    when the LLM response has no "relevant" key, rescore_band_blend() must
    leave that row's llm_proba at its ORIGINAL model_proba -- a no-op
    contribution to the blend -- not corrupt it with e.g. 0.5."""

    class NoRelevantKeyClient:
        def generate_batch(self, prompts, max_new_tokens=200):
            return ["{}" for _ in prompts]

    df_band = pd.DataFrame({
        "query": ["x", "y"], "title": ["a", "b"], "category": ["c", "d"],
        "brand": ["e", "f"], "attributes": ["g", "h"],
        "model_proba": [0.42, 0.58],
    })
    out = rescore_mod.rescore_band_blend(df_band, NoRelevantKeyClient(), batch_size=8)
    np.testing.assert_allclose(out, [0.42, 0.58])


def test_rescore_band_blend_combines_llm_confidence_correctly():
    """A parseable relevant=true/false + confidence response must convert to
    llm_proba = confidence (if relevant) or 1-confidence (if not) -- this is
    the value 06_rescore_uncertain_band.py's main() then weighted-averages
    with the GBDT's own model_proba (config.RESCORE_BLEND_LLM_WEIGHT)."""

    class FixedConfidenceClient:
        def __init__(self, relevant, confidence):
            self.relevant, self.confidence = relevant, confidence

        def generate_batch(self, prompts, max_new_tokens=200):
            return [f'{{"relevant": {str(self.relevant).lower()}, "confidence": {self.confidence}}}'
                    for _ in prompts]

    df_band = pd.DataFrame({
        "query": ["x"], "title": ["a"], "category": ["c"], "brand": ["e"], "attributes": ["g"],
        "model_proba": [0.5],  # irrelevant here -- overwritten if parse succeeds
    })
    out_true = rescore_mod.rescore_band_blend(df_band, FixedConfidenceClient(True, 0.9), batch_size=8)
    np.testing.assert_allclose(out_true, [0.9])
    out_false = rescore_mod.rescore_band_blend(df_band, FixedConfidenceClient(False, 0.9), batch_size=8)
    np.testing.assert_allclose(out_false, [0.1])
