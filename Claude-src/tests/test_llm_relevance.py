"""
Tests for the 2026-07-11 LLM pairwise relevance stage (10_llm_relevance.py) and
its wiring as a 4th ensemble member (05_train.py / 07_predict.py combine logic).

All GPU-free: the logprob->score conversion is a pure function, the sharded
scoring driver is exercised with MockRelevanceScorer, and the member-generic
combiner is checked directly. No vLLM/torch/transformers import required (the
module guards `import vllm` inside VLLMRelevanceScorer.__init__).
"""
import importlib

import numpy as np
import pandas as pd
import pytest

rel_mod = importlib.import_module("10_llm_relevance")


# ---------------- logprob -> P(relevant) ----------------

def test_score_from_logprobs_normalizes_1_vs_0():
    """Equal logprobs on '1' and '0' -> 0.5; heavier '1' -> >0.5."""
    assert rel_mod._score_from_logprobs([("1", -0.5), ("0", -0.5)]) == pytest.approx(0.5)
    s = rel_mod._score_from_logprobs([("1", -0.1), ("0", -2.0)])
    assert s > 0.8
    s = rel_mod._score_from_logprobs([("0", -0.1), ("1", -2.0)])
    assert s < 0.2


def test_score_from_logprobs_handles_subword_markers():
    """SentencePiece/BPE leading markers (▁, Ġ) must be stripped before match."""
    s = rel_mod._score_from_logprobs([("▁" + "1", -0.05), ("Ā1" if False else "Ġ" + "0", -3.0)])
    assert s > 0.9


def test_score_from_logprobs_one_sided_and_fallback():
    # only "1" present -> high; only "0" -> low
    assert rel_mod._score_from_logprobs([("1", -0.2), ("evet", -1.0)]) == pytest.approx(0.98)
    assert rel_mod._score_from_logprobs([("0", -0.2), ("x", -1.0)]) == pytest.approx(0.02)
    # neither present -> greedy text decides
    assert rel_mod._score_from_logprobs([("foo", -0.2)], greedy_text="1") == pytest.approx(0.98)
    # neither present, no usable greedy -> neutral fill
    assert rel_mod._score_from_logprobs([("foo", -0.2)], greedy_text="bar", fill=0.5) == pytest.approx(0.5)


def test_score_always_in_unit_interval():
    rng = np.random.default_rng(0)
    for _ in range(200):
        lp1, lp0 = rng.normal(size=2) * 3
        s = rel_mod._score_from_logprobs([("1", float(lp1)), ("0", float(lp0))])
        assert 0.0 <= s <= 1.0


# ---------------- prompt building ----------------

def test_build_prompt_truncates_and_contains_query():
    p = rel_mod.build_prompt("kadın ayakkabı", "x" * 500, "ayakkabı/sneaker", "nike", "renk: siyah, " * 100)
    assert "kadın ayakkabı" in p
    # title/attributes truncated -> prompt shouldn't contain the full 500-char title
    assert "x" * 500 not in p
    assert p.strip().endswith(":")  # ends asking for the single digit


def test_build_prompt_handles_none_and_nan():
    p = rel_mod.build_prompt(None, None, None, None, float("nan"))
    assert isinstance(p, str) and len(p) > 0


# ---------------- sharded / resumable scoring driver ----------------

@pytest.mark.parametrize("n,shard", [(64, 16), (16, 16), (5, 16), (200, 64)])
def test_score_frame_covers_every_row(tmp_path, monkeypatch, n, shard):
    # redirect checkpoint dir into tmp so shards don't collide across tests
    monkeypatch.setattr(rel_mod, "CACHE_DIR", tmp_path)
    prompts = np.array([f"Sorgu: q{i} ürün {i}" for i in range(n)], dtype=object)
    scorer = rel_mod.MockRelevanceScorer()
    scores = rel_mod._score_frame(prompts, scorer, "unit", shard_size=shard)
    assert len(scores) == n
    assert not np.isnan(scores).any()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_score_frame_resumes_from_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(rel_mod, "CACHE_DIR", tmp_path)
    prompts = np.array([f"p{i}" for i in range(50)], dtype=object)

    calls = {"n": 0}

    class CountingScorer:
        def score_batch(self, prompts):
            calls["n"] += len(prompts)
            return np.full(len(prompts), 0.3, dtype=np.float32)

    rel_mod._score_frame(prompts, CountingScorer(), "resume", shard_size=20)
    first = calls["n"]
    assert first == 50
    # second run: all shards checkpointed -> scorer never called again
    rel_mod._score_frame(prompts, CountingScorer(), "resume", shard_size=20)
    assert calls["n"] == first  # unchanged: fully resumed from disk


# ---------------- member-generic combiner (training_utils, pure) ----------------

def test_combiner_handles_3_and_4_members():
    from training_utils import combine_ensemble_predictions
    p = {"lgb": np.array([0.2, 0.8]), "xgb": np.array([0.3, 0.7]),
         "cat": np.array([0.25, 0.75]), "llm": np.array([0.9, 0.1])}

    # blend over 4 members
    meta4 = {"method": "blend",
             "blend_weights": {"lgb": 0.25, "xgb": 0.25, "cat": 0.25, "llm": 0.25}}
    out4 = combine_ensemble_predictions(p, meta4)
    assert out4[0] == pytest.approx((0.2 + 0.3 + 0.25 + 0.9) / 4)

    # blend over the original 3 members still works (llm simply absent from weights)
    meta3 = {"method": "blend", "blend_weights": {"lgb": 0.5, "xgb": 0.3, "cat": 0.2}}
    out3 = combine_ensemble_predictions(p, meta3)
    assert out3[1] == pytest.approx(0.5 * 0.8 + 0.3 * 0.7 + 0.2 * 0.75)


def test_combiner_stacking_with_llm_member():
    from training_utils import combine_ensemble_predictions
    p = {"lgb": np.array([0.4]), "xgb": np.array([0.6]),
         "cat": np.array([0.5]), "llm": np.array([0.95])}
    meta = {"method": "stacking",
            "meta_coef": {"lgb": 1.0, "xgb": 1.0, "cat": 1.0, "llm": 2.0},
            "meta_intercept": -1.0}
    z = -1.0 + 0.4 + 0.6 + 0.5 + 2.0 * 0.95
    assert combine_ensemble_predictions(p, meta)[0] == pytest.approx(1.0 / (1.0 + np.exp(-z)))


def test_predict_blend_injects_llm_member_only_when_expected():
    # predict_blend lives in 07_predict, which imports lightgbm/xgboost/catboost
    # /pyarrow at module load -- skip if those heavy deps aren't installed.
    pytest.importorskip("lightgbm")
    pytest.importorskip("xgboost")
    pytest.importorskip("catboost")
    pytest.importorskip("pyarrow")
    predict_mod = importlib.import_module("07_predict")

    class FakeBooster:
        def __init__(self, v):
            self.v = v

        def predict(self, X):
            return np.full(len(X), self.v)

    class FakeProba:
        def __init__(self, v):
            self.v = v

        def predict_proba(self, X):
            n = len(X)
            return np.column_stack([1 - np.full(n, self.v), np.full(n, self.v)])

    models = {"lgb": [FakeBooster(0.2)], "xgb": [FakeProba(0.4)], "cat": [FakeProba(0.6)]}
    X = np.zeros((3, 2), dtype=np.float32)

    # combiner WITHOUT llm -> p_llm ignored
    meta3 = {"method": "blend", "blend_weights": {"lgb": 0.5, "xgb": 0.25, "cat": 0.25}}
    out = predict_mod.predict_blend(X, models, meta3)
    assert out == pytest.approx(0.5 * 0.2 + 0.25 * 0.4 + 0.25 * 0.6)

    # combiner WITH llm -> supplied p_llm is used
    meta4 = {"method": "blend", "blend_weights": {"lgb": 0.25, "xgb": 0.25, "cat": 0.25, "llm": 0.25}}
    p_llm = np.array([0.8, 0.8, 0.8], dtype=np.float32)
    out = predict_mod.predict_blend(X, models, meta4, p_llm=p_llm)
    assert out[0] == pytest.approx((0.2 + 0.4 + 0.6 + 0.8) / 4)
