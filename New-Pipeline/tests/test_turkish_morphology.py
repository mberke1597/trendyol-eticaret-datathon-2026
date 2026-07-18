"""Unit tests for turkish_morphology.py. Run with: pytest tests/test_turkish_morphology.py -v

NOTE: these tests call the real zeyrek engine (no mock) -- they are slower
than typical unit tests (model load once per session) and depend on zeyrek's
actual lexicon, so they double as a "did the dependency/environment break"
smoke test, similar in spirit to how Claude-src's tests validate real behavior
over mocked behavior wherever feasible."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import turkish_morphology as tm


def setup_module():
    tm.init()


def test_engine_initializes():
    assert tm.engine_name() in ("zeyrek", "zemberek_full")


def test_plural_possessive_reduces_to_correct_root():
    # "ayakkabılarım" = ayakkabı(shoe) + lar(plural) + ım(my)
    assert tm.root_of("ayakkabılarım") == "ayakkabı"


def test_ablative_case_reduces_to_correct_root():
    # "evlerinden" = ev(house) + ler(plural) + i(poss) + nden(ablative, "from")
    assert tm.root_of("evlerinden") == "ev"


def test_accusative_case_reduces_to_correct_root():
    assert tm.root_of("telefonu") == "telefon"
    assert tm.root_of("kırmızıyı") == "kırmızı"


def test_bare_root_is_its_own_root():
    assert tm.root_of("siyah") == "siyah"


def test_unparseable_ascii_folded_word_falls_back_gracefully():
    """This is the documented limitation: zeyrek can't parse ASCII-folded
    Turkish. root_of must NEVER raise or return empty -- it should fall back
    to the lowercased input so overlap features still have a stable key,
    even though it won't match the correctly-accented form. This is exactly
    why typo_tolerance.correct() must run BEFORE this, not after."""
    result = tm.analyze_word("kirmizi")
    assert result.parsed is False
    assert result.lemma == "kirmizi"  # unparsed fallback, not None/empty


def test_empty_and_none_input_never_raises():
    r1 = tm.analyze_word("")
    r2 = tm.analyze_word(None)
    assert r1.parsed is False and r2.parsed is False


def test_root_of_never_returns_none_or_empty_for_nonempty_input():
    words = ["ayakkabılarım", "xyznonsense123", "kirmizi", "a"]
    for w in words:
        r = tm.root_of(w)
        assert r, f"root_of({w!r}) returned falsy: {r!r}"
