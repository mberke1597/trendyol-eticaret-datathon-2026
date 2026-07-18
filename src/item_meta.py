"""
Attribute parsing + controlled vocabularies for the "hard constraint" features
(prompt.md section 4: gender/age must never contradict the query; color/material/
pattern should be extracted from the free-text `attributes` column).

`attributes` is "key: value, key: value, ..." but values themselves can contain
commas ("diğer özellikler: çok amaçlı kullanım, akvaryum dışında ..., menşei: tr"),
so a naive split(",") shreds long values into fake keys. Instead we target the
handful of keys we actually need with a bounded-lookahead regex: capture text
after "renk:" (etc.) up to the next ", <2-30 chars>:" or end of string.
"""
import re

import numpy as np

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
    "gold", "rose gold", "rose", "kırık beyaz", "buz mavisi", "gece mavisi", "vizon",
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
