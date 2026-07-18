import item_meta as im
from text_utils import tokenize


def test_gender_contradiction_basic():
    assert im.gender_contradiction("erkek", "kadın") == 1
    assert im.gender_contradiction("kadın", "erkek") == 1
    assert im.gender_contradiction("kadın", "kadın") == 0
    assert im.gender_contradiction("unisex", "erkek") == 0
    assert im.gender_contradiction(None, "erkek") == 0
    assert im.gender_contradiction("erkek", "unisex") == 0
    assert im.gender_contradiction("erkek", "unknown") == 0


def test_age_contradiction_basic():
    assert im.age_contradiction("cocuk", "yetiskin") == 1
    assert im.age_contradiction("yetiskin", "yetiskin") == 0
    assert im.age_contradiction(None, "cocuk") == 0
    assert im.age_contradiction("cocuk", "bebek_cocuk") == 0
    assert im.age_contradiction("yetiskin", "bebek_cocuk") == 1


def test_brand_matcher_detects_contradiction():
    # min_item_count=1: production default (MIN_ITEM_COUNT=20) exists to filter
    # one-off private-label noise out of a REAL ~80k-brand catalog (see
    # BrandMatcher's docstring for the 58.5%-false-positive-rate measurement
    # that motivated it) -- with a 4-brand test fixture every brand is a
    # "one-off" by construction, so we override the threshold to test the
    # matching logic itself; the filter itself is tested separately below.
    bm = im.BrandMatcher(["Nike", "Nike Air", "LC Waikiki", "Zara"], min_item_count=1)
    q = tokenize("nike spor ayakkabi")
    assert bm.check_brand_contradiction(q, "Adidas") is True
    assert bm.check_brand_contradiction(q, "Nike Air Force") is False


def test_brand_matcher_no_brand_mention_no_contradiction():
    bm = im.BrandMatcher(["Nike", "Adidas"], min_item_count=1)
    q = tokenize("spor ayakkabi")
    assert bm.check_brand_contradiction(q, "Puma") is False


def test_brand_matcher_filters_stop_brands():
    bm = im.BrandMatcher(["Erkek", "Beyaz", "Nike"], min_item_count=1)  # "Erkek"/"Beyaz" should be filtered as common words
    assert "erkek" not in bm.brands
    assert "beyaz" not in bm.brands
    assert "nike" in bm.brands


def test_brand_matcher_min_item_count_filters_rare_brands():
    """Locks in the real 2026-07-04 fix: at the PRODUCTION default threshold
    (min_item_count=20), a brand string appearing fewer than 20 times must be
    dropped even if it isn't a stop-word -- this is what actually closes the
    58.5%-false-positive-rate bug (one-off seller tags like "kremi", "kupa",
    "doğan" measured directly on real items.csv; see class docstring)."""
    brands = ["RealBrand"] * 25 + ["OneOffSellerTag"] * 3
    bm = im.BrandMatcher(brands)  # default min_item_count=20 (production behavior)
    assert "realbrand" in bm.brands
    assert "oneoffsellertag" not in bm.brands


def test_color_synonym_aware_match():
    from text_utils import COLOR_SYNONYMS
    assert im.synonym_aware_match(frozenset({"altın"}), frozenset({"sarı"}), COLOR_SYNONYMS)
    assert not im.synonym_aware_match(frozenset({"mavi"}), frozenset({"kırmızı"}), COLOR_SYNONYMS)


def test_color_collocation_suppresses_match():
    assert im.has_color_collocation("beyaz eşya alacağım")
    assert not im.has_color_collocation("kırmızı çanta arıyorum")


def test_is_generic_query():
    assert im.is_generic_query(tokenize("hediyelik biblo"))
    assert not im.is_generic_query(tokenize("kadın spor ayakkabı"))


def test_expand_with_synonyms():
    from text_utils import COLOR_SYNONYMS
    expanded = im.expand_with_synonyms(frozenset({"altin"}), COLOR_SYNONYMS)
    assert "sarı" in expanded or "sari" in expanded
