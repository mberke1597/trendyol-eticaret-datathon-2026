"""
Turkish-aware text normalization / tokenization shared by negative sampling
and feature engineering.

Design notes (see prompt.md section 4 "Linguistic Variations"):
  - Turkish is agglutinative -> word-level exact match misses inflected forms
    ("kitap" vs "kitabini"). We add a light suffix stripper (`stem`) as a cheap
    approximation -- not a full morphological analyzer, but it collapses the
    ~30 highest-frequency case/possessive/plural suffixes.
  - Spacing/compounding mismatches ("multi vitamin" vs "multivitamin") are NOT
    fixable by any tokenizer; that's what char n-gram similarity (in
    features.py) is for -- it degrades gracefully across a boundary shift
    instead of an exact-token miss.
"""
import re
from functools import lru_cache

_TOKEN_SPLIT_RE = re.compile(r"[^0-9a-zçğıöşü]+")

# longest-suffix-first so we don't strip a short suffix that is a prefix of a longer one
_SUFFIXES = sorted(
    [
        "lardan", "lerden", "larda", "lerde", "ndan", "nden", "ndaki", "ndeki",
        "lar", "ler", "dan", "den", "tan", "ten", "nin", "nın", "nun", "nün",
        "in", "ın", "un", "ün", "de", "da", "te", "ta", "ye", "ya", "yi", "yı",
        "yu", "yü", "si", "sı", "su", "sü", "i", "ı", "u", "ü", "e", "a", "s",
    ],
    key=len,
    reverse=True,
)
_MIN_STEM_LEN = 3


def normalize(s):
    if not isinstance(s, str) or not s:
        return ""
    s = s.replace("İ", "i").replace("I", "ı")
    s = s.lower().replace("i̇", "i")
    return s


def tokenize(s, min_len=2):
    return [t for t in _TOKEN_SPLIT_RE.split(normalize(s)) if len(t) >= min_len]


@lru_cache(maxsize=200_000)
def stem(token):
    """Strip at most one common Turkish suffix; cached since vocab is small & reused a lot."""
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= _MIN_STEM_LEN:
            return token[: -len(suf)]
    return token


def tokenize_stem(s, min_len=2):
    return [stem(t) for t in tokenize(s, min_len=min_len)]


def char_ngrams(s, n=3):
    s = normalize(s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a) + len(set_b) - inter
    return inter / union if union else 0.0
