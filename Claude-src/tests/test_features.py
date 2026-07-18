import numpy as np
import pandas as pd
import pytest

feat = pytest.importorskip("features")


@pytest.fixture
def toy_catalog():
    items = pd.DataFrame({
        "item_id": ["I1", "I2", "I3", "I4"],
        "title": [
            "Kadın Spor Ayakkabı Beyaz", "Erkek Kot Pantolon Mavi",
            "Multivitamin 60 Tablet", "Nike Air Force Sneaker Beyaz",
        ],
        "category": [
            "ayakkabı/spor ayakkabı/sneaker", "giyim/pantolon/kot pantolon",
            "süpermarket/sağlık/vitamin", "ayakkabı/spor ayakkabı/sneaker",
        ],
        "brand": ["Trendyol Milla", "LC Waikiki", "Solgar", "Nike"],
        "gender": ["kadın", "erkek", "unknown", "unisex"],
        "age_group": ["yetiskin", "yetiskin", "unknown", "yetiskin"],
        "attributes": [
            "renk: beyaz, materyal: pamuklu", "renk: mavi, materyal: kot", "", "renk: beyaz",
        ],
    })
    terms = pd.DataFrame({
        "term_id": ["T1", "T2", "T3", "T4"],
        "query": ["kadın ayakkabı", "multi vitamin", "nike ayakkabi", "erkek kadın ayakkabı"],
    })
    return items, terms


@pytest.fixture
def toy_emb_pop():
    emb = {
        "query_main": np.random.RandomState(0).randn(4, 8).astype(np.float16),
        "query_tiny": np.random.RandomState(1).randn(4, 4).astype(np.float16),
        "item_title": np.random.RandomState(2).randn(4, 8).astype(np.float16),
        "item_attr": np.random.RandomState(3).randn(4, 4).astype(np.float16),
        "category_emb": np.random.RandomState(4).randn(3, 8).astype(np.float16),
        "item_category_idx": np.array([0, 1, 2, 0], dtype=np.int32),
    }
    pop = {
        "item_click_count": np.array([5, 2, 0, 10], dtype=np.int64),
        "item_cat_mean_log": np.array([1.0, 0.5, 0.0, 1.0], dtype=np.float32),
        "brand_click_count": np.array([5, 2, 0, 10], dtype=np.int64),
    }
    return emb, pop


def test_lexical_index_builds_without_llm_enrichment(toy_catalog):
    items, terms = toy_catalog
    lex = feat.LexicalIndex(items, terms)
    assert lex.has_llm_query is False
    assert lex.has_llm_item is False
    assert lex.item_word_X.shape[0] == 4


def test_compute_batch_features_no_crash_and_expected_shape(toy_catalog, toy_emb_pop):
    items, terms = toy_catalog
    emb, pop = toy_emb_pop
    lex = feat.LexicalIndex(items, terms)
    term_idx = np.array([0, 1, 2, 3])
    item_idx = np.array([0, 2, 3, 0])
    labels = np.array([1, 1, 1, 0], dtype=np.int8)
    out = feat.compute_batch_features(term_idx, item_idx, lex, emb, pop, label=labels)
    assert len(out) == 4
    assert out.isnull().sum().sum() == 0


def test_bigram_merge_expanded_overlap_beats_plain(toy_catalog, toy_emb_pop):
    """'multi vitamin' query vs 'Multivitamin' item title: plain word overlap
    should be lower than the synonym/bigram-expanded overlap."""
    items, terms = toy_catalog
    emb, pop = toy_emb_pop
    lex = feat.LexicalIndex(items, terms)
    term_idx = np.array([1])  # "multi vitamin"
    item_idx = np.array([2])  # "Multivitamin 60 Tablet"
    out = feat.compute_batch_features(term_idx, item_idx, lex, emb, pop)
    assert out.loc[0, "expanded_overlap_n"] >= out.loc[0, "word_overlap_n"]
    assert out.loc[0, "expanded_overlap_n"] > 0


def test_category_level_overlap_present_for_matching_category(toy_catalog, toy_emb_pop):
    items, terms = toy_catalog
    emb, pop = toy_emb_pop
    lex = feat.LexicalIndex(items, terms)
    term_idx = np.array([0])  # "kadın ayakkabı"
    item_idx = np.array([0])  # ayakkabı/spor ayakkabı/sneaker
    out = feat.compute_batch_features(term_idx, item_idx, lex, emb, pop)
    assert out.loc[0, "cat_word_overlap_n"] > 0  # "ayakkabı" appears in the category path


def test_gender_contradiction_feature_matches_item_meta(toy_catalog, toy_emb_pop):
    items, terms = toy_catalog
    emb, pop = toy_emb_pop
    lex = feat.LexicalIndex(items, terms)
    # T1 "kadın ayakkabı" vs I2 (erkek item) -> must contradict
    term_idx = np.array([0])
    item_idx = np.array([1])
    out = feat.compute_batch_features(term_idx, item_idx, lex, emb, pop)
    assert out.loc[0, "gender_contradiction"] == 1.0


def test_brand_contradiction_feature(toy_catalog, toy_emb_pop):
    items, terms = toy_catalog
    emb, pop = toy_emb_pop
    # brand_min_item_count=1: this 4-item toy catalog has every brand appearing
    # exactly once, always below BrandMatcher's real default (MIN_ITEM_COUNT=20,
    # see item_meta.BrandMatcher docstring) -- override it here so this test
    # exercises the matching logic itself, not the frequency filter (that has
    # its own dedicated test, test_brand_matcher_min_item_count_filters_rare_brands).
    lex = feat.LexicalIndex(items, terms, brand_min_item_count=1)
    # T3 "nike ayakkabi" vs I4 (Nike brand) -> no contradiction
    out_same = feat.compute_batch_features(np.array([2]), np.array([3]), lex, emb, pop)
    assert out_same.loc[0, "brand_contradiction"] == 0.0
    # T3 "nike ayakkabi" vs I1 (Trendyol Milla brand) -> contradiction
    out_diff = feat.compute_batch_features(np.array([2]), np.array([0]), lex, emb, pop)
    assert out_diff.loc[0, "brand_contradiction"] == 1.0


def test_turkish_morphology_off_by_default(toy_catalog):
    """config.USE_TURKISH_MORPHOLOGY defaults to 0 -- no root_* columns should
    exist at all (not just zero-valued) when it's off, so this optional stage
    never changes behavior for anyone who hasn't opted in."""
    import config
    assert config.USE_TURKISH_MORPHOLOGY is False
    items, terms = toy_catalog
    lex = feat.LexicalIndex(items, terms, brand_min_item_count=1)
    assert lex.has_turkish_morphology is False
    assert not hasattr(lex, "query_root_X")


def test_turkish_morphology_root_overlap_recovers_typo_match(toy_catalog, toy_emb_pop, monkeypatch):
    """Regression test for a real bug found integrating this (2026-07-07):
    the typo-correction vocabulary must be built from ITEM text only, never
    query text -- including query text let a query's own typo ("ayakkabi")
    get counted as an "already known real word", so correction never fired.
    T3 = "nike ayakkabi" (typo'd, missing the dotless-ı in "ayakkabı") should
    still overlap I1's root vocabulary via typo-correction + morphological
    rooting, even though the literal word_overlap_n does NOT catch this
    (see word_overlap_n assertion below -- proves root_overlap_n is adding
    real signal, not duplicating word_overlap_n)."""
    pytest.importorskip("zeyrek")
    pytest.importorskip("symspellpy")
    import config
    monkeypatch.setattr(config, "USE_TURKISH_MORPHOLOGY", True)

    items, terms = toy_catalog
    emb, pop = toy_emb_pop
    lex = feat.LexicalIndex(items, terms, brand_min_item_count=1)
    assert lex.has_turkish_morphology is True

    term_idx = np.array([2])  # T3 "nike ayakkabi"
    item_idx = np.array([0])  # I1 "Kadın Spor Ayakkabı Beyaz" (+ category "ayakkabı/...")
    out = feat.compute_batch_features(term_idx, item_idx, lex, emb, pop)
    assert out.loc[0, "word_overlap_n"] == 0, (
        "if this is no longer 0, the toy fixture changed -- this test needs "
        "a query/item pair where literal overlap fails but the typo'd word "
        "is recoverable via correction+rooting"
    )
    assert out.loc[0, "root_overlap_n"] >= 1, (
        "root_overlap_n should recover the 'ayakkabi'(typo) -> 'ayakkabı' "
        "match that word_overlap_n misses -- if this fails, check "
        "LexicalIndex.__init__'s typo-correction vocab is built from "
        "item_text only, not query_text (see the 2026-07-07 bug-fix comment "
        "right above the TurkishTypoCorrector().build_vocab_from_corpus call)"
    )
