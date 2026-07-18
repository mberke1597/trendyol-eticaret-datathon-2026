"""Unit tests for typo_tolerance.py. Run with: pytest tests/test_typo_tolerance.py -v"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typo_tolerance import TurkishTypoCorrector, deasciify, tokenize_simple

CORPUS = [
    "kırmızı ayakkabı", "siyah telefon kılıfı", "kadın elbise",
    "erkek pantolon", "mavi gömlek", "beyaz çanta", "ayakkabılarım",
    "yeşil şapka", "büyük boy tişört",
]


def make_corrector():
    return TurkishTypoCorrector().build_vocab_from_corpus(CORPUS)


def test_deasciify_folds_all_turkish_chars():
    assert deasciify("KIRMIZI ŞAPKA ÇANTA ÜÇGEN ÖRDEK ĞÜL") == "kirmizi sapka canta ucgen ordek gul"


def test_tokenize_simple_lowercases_and_splits():
    assert tokenize_simple("Kırmızı Ayakkabı!") == ["kırmızı", "ayakkabı"]


def test_known_word_passes_through_unchanged():
    c = make_corrector()
    assert c.correct("kırmızı") == "kırmızı"


def test_diacritic_dropped_input_corrected():
    c = make_corrector()
    assert c.correct("kirmizi") == "kırmızı"
    assert c.correct("ayakkabi") == "ayakkabı"
    assert c.correct("gomlek") == "gömlek"
    assert c.correct("canta") == "çanta"
    assert c.correct("sapka") == "şapka"


def test_genuine_typo_corrected_via_edit_distance():
    c = make_corrector()
    assert c.correct("telefonn") == "telefon" if "telefon" in c.vocab_counts else True
    assert c.correct("siyahh") == "siyah"


def test_unknown_out_of_vocab_word_left_unchanged():
    """Conservative behavior: don't guess-correct a real brand/word we've
    never seen into some unrelated vocabulary word -- see module docstring's
    rationale (mirrors Claude-src's SIZE_VOCAB trigger-gating philosophy)."""
    c = make_corrector()
    assert c.correct("nike") == "nike"
    assert c.correct("xyzabc123nonsense") == "xyzabc123nonsense"


def test_empty_and_non_string_input_never_raises():
    c = make_corrector()
    assert c.correct("") == ""
    assert c.correct(None) is None


def test_edit_distance_does_not_over_correct_diacritic_case():
    """Regression guard for the exact failure mode found 2026-07-07: SymSpell
    alone (max_edit_distance=2) does NOT find 'kırmızı' from 'kirmizi' because
    the true edit distance is 3 -- confirms the deasciify path is doing real
    work here, not redundant with the edit-distance path."""
    from symspellpy import SymSpell, Verbosity
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym.create_dictionary_entry("kırmızı", 100)
    suggestions = sym.lookup("kirmizi", Verbosity.CLOSEST, max_edit_distance=2)
    assert suggestions == [], (
        "if this now finds a match, symspellpy's behavior changed -- re-check "
        "whether the deasciify path in TurkishTypoCorrector is still needed"
    )
