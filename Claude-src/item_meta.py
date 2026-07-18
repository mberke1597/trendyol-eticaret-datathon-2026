"""
Attribute parsing + controlled vocabularies for the "hard constraint" features
(prompt.md section 4: gender/age must never contradict the query; color/material/
pattern should be extracted from the free-text `attributes` column; brand should
not contradict either).

Core parsing logic (bounded-lookahead regex for the "key: value, key: value, ..."
`attributes` text, gender/age contradiction with title-derived fallback for noisy
seller metadata) is ported unchanged from the proven ../src/item_meta.py.

Two things are NEW here vs. ../src/item_meta.py:
  1. Color/material matching is synonym-aware (altın<->sarı, gümüş<->gri,
     bordo<->kırmızı, lacivert<->mavi, kot<->denim, ...) with collocation
     exceptions ("beyaz eşya" is not a color reference) -- ported from a
     teammate's train_stacking.py review, re-verified as real UTF-8 here
     (see text_utils.py module docstring for why that matters).
  2. `BrandMatcher` -- a teammate had written good brand-contradiction logic
     (stop-word filtered, handles two-word/concatenated brand names) but never
     actually called it from their feature pipeline. It's wired into
     features.py's compute_batch_features here, not left as dead code.
"""
import re

import numpy as np

from text_utils import (
    COLOR_COLLOCATIONS,
    COLOR_SYNONYMS,
    MATERIAL_SYNONYMS,
    normalize,
    tokenize,
)

_KV_LOOKAHEAD = r"(.*?)(?=,\s*[^,:]{2,30}:|$)"

COLOR_KEYS = ["renk", "color detail", "kasa renk", "kordon renk", "kadran renk"]
MATERIAL_KEYS = ["materyal", "materyal bileşeni", "kumaş tipi", "kasa materyali", "kordon materyali"]
PATTERN_KEYS = ["desen"]

_COLOR_RE = re.compile(
    r"(?:^|,\s*)(?:" + "|".join(re.escape(k) for k in COLOR_KEYS) + r"):\s*" + _KV_LOOKAHEAD
)
_MATERIAL_RE = re.compile(
    r"(?:^|,\s*)(?:" + "|".join(re.escape(k) for k in MATERIAL_KEYS) + r"):\s*" + _KV_LOOKAHEAD
)
_PATTERN_RE = re.compile(
    r"(?:^|,\s*)(?:" + "|".join(re.escape(k) for k in PATTERN_KEYS) + r"):\s*" + _KV_LOOKAHEAD
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")

COLOR_VOCAB = [
    "siyah", "beyaz", "kırmızı", "kirmizi", "mavi", "lacivert", "yeşil", "yesil", "sarı", "sari",
    "turuncu", "mor", "pembe", "gri", "kahverengi", "bej", "bordo", "altın", "altin", "gümüş",
    "gumus", "gold", "füme", "fume", "haki", "turkuaz", "krem", "ekru", "antrasit", "çok renkli",
    "cok renkli", "karışık", "karisik", "taba", "hardal", "petrol", "indigo", "mint", "somon",
    "rose gold", "rose", "kırık beyaz", "buz mavisi", "gece mavisi", "vizon",
]
MATERIAL_VOCAB = [
    "pamuk", "pamuklu", "polyester", "deri", "süet", "suet", "keten", "yün", "yun", "ipek",
    "viskon", "akrilik", "naylon", "metal", "ahşap", "ahsap", "cam", "plastik", "seramik",
    "porselen", "gümüş", "gumus", "altın", "altin", "çelik", "celik", "alüminyum", "aluminyum",
    "kauçuk", "kaucuk", "silikon", "kristal", "taş", "tas", "kumaş", "kumas", "triko", "kot",
    "jean", "likra", "elastan", "modal", "deri", "kadife", "şönil", "sonil", "file", "tül", "tul",
]
PATTERN_VOCAB = [
    "düz", "duz", "çizgili", "cizgili", "ekose", "puantiyeli", "çiçekli", "ciçekli", "geometrik",
    "kareli", "leopar", "yılan", "yilan", "kamuflaj", "batik", "damalı", "damali", "mix", "desenli",
    "karışık", "karisik", "soyut",
]

GENDER_WORDS = {
    "kadın": "kadın", "kadin": "kadın", "bayan": "kadın",
    "erkek": "erkek", "bay": "erkek",
    "unisex": "unisex",
}
AGE_WORDS = {
    "çocuk": "cocuk", "cocuk": "cocuk", "kız": "cocuk", "kiz": "cocuk", "oğlan": "cocuk", "oglan": "cocuk",
    "bebek": "bebek",
    "genç": "genc", "genc": "genc",
    "yetişkin": "yetiskin", "yetiskin": "yetiskin",
}

GENDER_NORM_MAP = {"kadın": "kadın", "erkek": "erkek", "unisex": "unisex", "unknown": "unknown"}
AGE_NORM_MAP = {
    "yetişkin": "yetiskin", "çocuk": "cocuk", "bebek": "bebek", "genç": "genc",
    "bebek & çocuk": "bebek_cocuk", "unknown": "unknown",
}


def _extract_vocab_hits(text, pattern_re, vocab_set):
    if not text:
        return frozenset()
    hits = set()
    for m in pattern_re.finditer(text):
        val = _HTML_TAG_RE.sub(" ", m.group(1))
        for tok in re.split(r"[^0-9a-zçğıöşü]+", val.lower()):
            if tok in vocab_set:
                hits.add(tok)
    return frozenset(hits)


def parse_item_constraints(attributes_series):
    """Vectorized-ish (single pass) extraction of color/material/pattern sets per item."""
    color_vocab = set(COLOR_VOCAB)
    material_vocab = set(MATERIAL_VOCAB)
    pattern_vocab = set(PATTERN_VOCAB)
    colors, materials, patterns = [], [], []
    for attrs in attributes_series.fillna(""):
        colors.append(_extract_vocab_hits(attrs, _COLOR_RE, color_vocab))
        materials.append(_extract_vocab_hits(attrs, _MATERIAL_RE, material_vocab))
        patterns.append(_extract_vocab_hits(attrs, _PATTERN_RE, pattern_vocab))
    return colors, materials, patterns


def extract_query_constraints(tokens):
    """tokens: list[str] (already tokenized query). Returns (colors, materials, patterns) frozensets."""
    color_vocab = set(COLOR_VOCAB)
    material_vocab = set(MATERIAL_VOCAB)
    pattern_vocab = set(PATTERN_VOCAB)
    tset = set(tokens)
    return (
        frozenset(tset & color_vocab),
        frozenset(tset & material_vocab),
        frozenset(tset & pattern_vocab),
    )


def has_color_collocation(query_norm):
    """True if the query contains a phrase where a 'color' word isn't actually a
    color reference (e.g. 'beyaz eşya' = home appliances, not the color white)."""
    return any(coll in query_norm for coll in COLOR_COLLOCATIONS)


def synonym_aware_match(query_set, item_set, synonym_dict):
    """True if query_set and item_set share a token directly, OR a query token's
    synonym cluster intersects item_set (e.g. query 'altın' vs item tagged 'sarı')."""
    if query_set & item_set:
        return True
    for q in query_set:
        syns = synonym_dict.get(q)
        if syns and (syns & item_set):
            return True
    return False


def expand_with_synonyms(token_set, synonym_dict):
    """Query-side-only expansion (same vectorization rationale as
    text_utils.expand_query_tokens): union a token set with every synonym
    cluster reachable from its members, ONCE per unique query, so the
    resulting expanded set can be checked against item constraint sets via a
    plain (vectorizable) intersection instead of a per-pair synonym lookup."""
    expanded = set(token_set)
    for tok in token_set:
        syns = synonym_dict.get(tok)
        if syns:
            expanded |= syns
    return frozenset(expanded)


# Generic/ambiguous query markers (prompt.md section 4 "popularity bias" risk:
# vague queries like "hediye" (gift) or "dekor" match almost anything by
# embedding similarity, so the GBDT benefits from an explicit flag rather than
# relying on similarity scores alone to express "this query is inherently broad").
GENERIC_QUERY_KEYWORDS = {
    "hediye", "hediyesi", "sus", "susu", "aksesuar", "aksesuari", "dekor", "dekoru",
    "hediyelik", "seti", "set",
}


def is_generic_query(tokens):
    return any(t in GENERIC_QUERY_KEYWORDS for t in tokens)


def normalize_gender(raw):
    return GENDER_NORM_MAP.get(raw, "unknown")


def normalize_age(raw):
    return AGE_NORM_MAP.get(raw, "unknown")


def gender_intent_from_tokens(tokens):
    for t in tokens:
        if t in GENDER_WORDS:
            return GENDER_WORDS[t]
    return None


def age_intent_from_tokens(tokens):
    for t in tokens:
        if t in AGE_WORDS:
            return AGE_WORDS[t]
    return None


def gender_contradiction(query_intent, item_gender_norm):
    if query_intent is None or item_gender_norm in ("unknown", "unisex"):
        return 0
    if query_intent == "unisex":
        return 0
    return int(query_intent != item_gender_norm)


def age_contradiction(query_intent, item_age_norm):
    if query_intent is None or item_age_norm == "unknown":
        return 0
    if item_age_norm == "bebek_cocuk":
        return int(query_intent not in ("bebek", "cocuk"))
    return int(query_intent != item_age_norm)


class BrandMatcher:
    """Detects when a query names a specific brand that contradicts the item's
    actual brand. Ported from a teammate's utils.py (good logic, stop-word
    filtered, handles two-word and concatenated brand names) -- ported here with
    verified UTF-8 and *actually wired into* features.py's compute_batch_features,
    unlike the original where it was fully implemented but never called.

    CRITICAL FIX (found on a real Kaggle run, 2026-07-04): the teammate's
    original design took the *unique* brand strings from `items.csv`'s `brand`
    column at face value. Measured directly on this competition's real
    items.csv: that column has 79,790 unique values, but ~30k of them are
    one-off private-label/seller tags rather than recognizable brands --
    including plain Turkish dictionary words a small seller happened to put in
    their brand field ("gold", "cam", "kadın", "mont", "ses", "kahve",
    "servis", "oto", "bayrak", "mutfak", "ayna", "kremi", "tasarım", "kupa",
    "marka", "doğan"). Building the matcher from all unique values caused
    `query_brands()` to fire on 58.5% of the real terms.csv queries (measured:
    29,354/50,153) -- e.g. 'kadın tesettür kışlık gömlek' -> matched {'kadın'},
    'gold boydan ayna' -> matched {'gold', 'ayna'}. On the real Kaggle run this
    produced `brand_contradiction` firing on ~50% of all submission rows and
    collapsed the predicted-positive rate to 10.7% (vs. the leaderboard-proven
    ~28-31% target -- see DESIGN.md "Threshold calibration reality check").
    Fixed two ways: (1) a brand must appear on at least `min_item_count` items
    to be treated as a real brand at all (real brands like adidas/nike/koton
    have thousands of items; noise tags are almost always singletons or near
    it -- >=20 keeps 7,574/79,790 candidates, i.e. drops 90% of the noise while
    keeping every recognizable brand seen in the value_counts sample), and (2)
    STOP_BRANDS is extended with the specific false positives measured above."""

    STOP_BRANDS = {
        "kız", "erkek", "çocuk", "bebek", "siyah", "beyaz", "mavi", "kırmızı",
        "yeşil", "sarı", "gri", "pembe", "mor", "turuncu", "lacivert", "deri",
        "pamuk", "keten", "ipek", "spor", "klasik", "büyük", "küçük", "yeni",
        "eski", "su", "ip", "el", "ev", "al", "ay", "ak", "en", "tek", "bir",
        "store", "home", "life", "pro", "as", "star", "plus", "fit", "wear",
        "art", "design", "style",
        # added 2026-07-04 -- confirmed false positives measured directly
        # against real terms.csv (see class docstring for the measurement)
        "kadın", "gold", "cam", "mont", "ses", "kahve", "servis", "oto",
        "bayrak", "mutfak", "ayna", "kremi", "tasarım", "kupa", "marka",
        "doğan", "peluş",
    }

    # Minimum number of item rows a brand string must appear on to be treated
    # as a real brand for contradiction-checking purposes -- see class
    # docstring. Chosen from the real items.csv brand-count distribution:
    # >=20 keeps 7,574/79,790 (9.5%) unique brand strings, discarding the
    # long tail of one-off seller/private-label tags without needing to name
    # every generic word individually.
    MIN_ITEM_COUNT = 20

    def __init__(self, brand_series, min_item_count=None):
        """brand_series: the FULL (non-deduplicated) item brand column --
        e.g. `items_df["brand"].fillna("")` -- so frequency can be computed.
        Passing a pre-deduplicated list (e.g. via `pd.unique`) silently
        disables the frequency filter and reintroduces the bug above; do not
        do that."""
        if min_item_count is None:
            min_item_count = self.MIN_ITEM_COUNT
        counts = {}
        for brand in brand_series:
            if not isinstance(brand, str) or not brand.strip():
                continue
            norm_brand = normalize(brand)
            if len(norm_brand) < 3 or norm_brand in self.STOP_BRANDS:
                continue
            counts[norm_brand] = counts.get(norm_brand, 0) + 1
        self.brands = {b for b, c in counts.items() if c >= min_item_count}

    def query_brands(self, query_tokens):
        """Extract candidate brand mentions (single/two-word/concatenated) from a
        tokenized query. Returns a (possibly empty) set of matched brand strings."""
        matched = set()
        n = len(query_tokens)
        for i in range(n):
            word = query_tokens[i]
            if word in self.brands:
                matched.add(word)
            if i < n - 1:
                two_word = f"{query_tokens[i]} {query_tokens[i + 1]}"
                if two_word in self.brands:
                    matched.add(two_word)
                concat = f"{query_tokens[i]}{query_tokens[i + 1]}"
                if concat in self.brands:
                    matched.add(concat)
        return matched

    def check_brand_contradiction(self, query_tokens, item_brand):
        """True if the query names a specific brand and the item's brand doesn't
        match any of them (substring match both ways, so 'nike air' item brand
        matches query brand 'nike')."""
        if not item_brand or not isinstance(item_brand, str):
            return False
        norm_item_brand = normalize(item_brand)
        if not norm_item_brand:
            return False
        matched_brands = self.query_brands(query_tokens)
        if not matched_brands:
            return False
        item_brand_clean = norm_item_brand.replace(" ", "")
        for b in matched_brands:
            b_clean = b.replace(" ", "")
            if b_clean in item_brand_clean or item_brand_clean in b_clean:
                return False
        return True
