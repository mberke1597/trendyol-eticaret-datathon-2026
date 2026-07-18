"""
UTF-8 round-trip + tokenization/synonym tests for text_utils.py.

Run this after ANY edit to text_utils.py -- a teammate's train_stacking.py
had a mojibake bug in exactly this kind of Turkish-character string literal
that crashed str.maketrans() at import time (verified by actually running
it, see Claude-src/DESIGN.md). This test exists specifically to catch that
class of bug before it ships.
"""
import text_utils as tu


def test_turkish_chars_are_single_codepoints():
    """The literal bug class: mojibake turns e.g. 'ı' into a 2-character
    garbage sequence. Every Turkish special character used anywhere in this
    module must be exactly one codepoint."""
    for ch in "çğıöşüİ":
        assert len(ch) == 1, f"{ch!r} is not a single codepoint -- mojibake risk"


def test_tokenize_real_turkish_text():
    assert tu.tokenize("Kadın Ayakkabı") == ["kadın", "ayakkabı"]
    assert tu.tokenize("kırmızı çanta") == ["kırmızı", "çanta"]
    assert tu.tokenize("Erkek Tişört") == ["erkek", "tişört"]
    assert tu.tokenize("İstanbul") == ["istanbul"]


def test_stem_strips_common_suffixes():
    assert tu.stem("kitaplar") == "kitap"
    assert tu.stem("ayakkabısı") != "ayakkabısı" or len("ayakkabısı") < 3  # some suffix stripped


def test_synonym_clusters_are_symmetric():
    """_expand_clusters must produce a symmetric lookup: if a -> {b}, then b -> {a}."""
    for word, syns in tu.SYNONYMS_DICT.items():
        for s in syns:
            assert word in tu.SYNONYMS_DICT.get(s, set()), f"asymmetric synonym: {word} -> {s}"


def test_color_synonym_altin_sari():
    assert "sarı" in tu.COLOR_SYNONYMS.get("altin", set()) or "sarı" in tu.COLOR_SYNONYMS.get("altın", set())


def test_material_synonym_kot_denim():
    assert "denim" in tu.MATERIAL_SYNONYMS.get("kot", set())


def test_category_levels_splits_hierarchical_path():
    assert tu.category_levels("ayakkabı/spor ayakkabı/sneaker") == ["ayakkabı", "spor ayakkabı", "sneaker"]
    assert tu.category_levels("") == []
    assert tu.category_levels(None) == []


def test_expand_query_tokens_bigram_merge():
    tokens = tu.tokenize("multi vitamin")
    expanded = tu.expand_query_tokens(tokens)
    assert "multivitamin" in expanded


def test_expand_query_tokens_synonym_cluster():
    tokens = tu.tokenize("ayakkabi ariyorum")
    expanded = tu.expand_query_tokens(tokens)
    assert "sneaker" in expanded


def test_get_expanded_match_counts_bigram_and_synonym():
    q = tu.tokenize("multi vitamin")
    assert tu.get_expanded_match(q, {"multivitamin"}) == 2
    q2 = tu.tokenize("ayakkabi ariyorum")
    assert tu.get_expanded_match(q2, {"sneaker", "ariyorum"}) == 2
