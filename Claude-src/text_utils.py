"""
Turkish-aware text normalization / tokenization + domain-synonym expansion,
shared by negative sampling and feature engineering.

Ports the proven parts of ../src/text_utils.py (normalize/tokenize/stem/
char_ngrams -- unchanged, already correct) and adds a domain-synonym /
bigram-merge overlap layer inspired by a teammate's train_stacking.py review
(see ../DESIGN.md and Claude-src/DESIGN.md "Dersler" section for the full
writeup). Two things that review flagged are handled differently here on
purpose:

  1. The teammate's SYNONYMS_DICT / COLORS_EN / TR_TO_EN_MAP had a mojibake
     (double-encoding) bug in the Turkish-character string literals that
     crashed `str.maketrans(...)` at import time (verified by actually
     running it). All Turkish literals below are written directly as UTF-8
     and round-trip tested in tests/test_text_utils.py -- run that test
     after ANY edit to this file before trusting it.
  2. Synonym pairs are declared once as undirected clusters and expanded
     into a symmetric dict programmatically (`_expand_clusters`), instead of
     hand-writing both directions -- the teammate's dict had to spell out
     "a: {b}" and "b: {a}" separately, which is exactly the kind of
     asymmetric-by-typo risk we want to avoid.
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


def category_levels(category_path):
    """'ayakkabı/spor ayakkabı/sneaker' -> ['ayakkabı', 'spor ayakkabı', 'sneaker'].
    Empty/missing path -> []. See Claude-src/DESIGN.md category-overlap finding:
    71.8% of positive pairs have query-token overlap at SOME category-path level,
    64.7% at the most specific (last) level -- this is what makes the per-level
    overlap feature in features.py worth computing explicitly."""
    if not isinstance(category_path, str) or not category_path:
        return []
    return [lvl.strip() for lvl in category_path.split("/") if lvl.strip()]


# ---------------------------------------------------------------------------
# Domain synonym clusters (Turkish e-commerce). Declared as undirected clusters
# and expanded into a symmetric lookup dict -- adding a new synonym means adding
# one word to one cluster, not editing N dict entries by hand.
# ---------------------------------------------------------------------------
_SYNONYM_CLUSTERS = [
    {"ayakkabı", "ayakkabi", "sneaker", "bot", "babet", "stiletto", "çizme", "cizme", "terlik", "panduf"},
    {"mont", "kaban", "ceket", "hırka", "hirka", "parka", "pardösü", "pardesu", "yağmurluk", "yagmurluk"},
    {"tişört", "tisort", "t-shirt", "tshirt", "bluz", "body", "sweatshirt", "sweat"},
    {"pantolon", "jean", "kot", "tayt", "şort", "sort", "denim"},
    {"çanta", "canta", "sırt çantası", "sirt cantasi", "valiz", "bavul", "portföy", "portfoy"},
    {"parfüm", "parfum", "edp", "edt", "deodorant", "koku"},
    {"ruj", "lip gloss", "lipgloss", "dudak parlatıcısı", "dudak parlaticisi"},
    {"saat", "kol saati"},
]

# color/material synonym clusters (item_meta.py COLOR_VOCAB/MATERIAL_VOCAB already
# hold the canonical vocab; these clusters only capture cross-word synonymy that a
# plain vocabulary lookup can't, e.g. "altın" (gold) used as a color word matching
# an item tagged "sarı" (yellow)).
_COLOR_SYNONYM_CLUSTERS = [
    {"altın", "altin", "sarı", "sari"},
    {"gümüş", "gumus", "gri"},
    {"bordo", "kırmızı", "kirmizi"},
    {"lacivert", "mavi"},
]
_MATERIAL_SYNONYM_CLUSTERS = [
    {"kot", "denim"},
    {"triko", "örgü", "orgu", "orgo", "yün", "yun", "akrilik"},
]
# collocations where the "color" word is not actually a color reference and must
# not trigger a color-match/conflict feature at all (teammate's finding, ported).
COLOR_COLLOCATIONS = {"beyaz eşya", "beyaz esya", "yeşil çay", "yesil cay", "altın çilek",
                       "altin cilek", "gümüş şampuan", "gumus sampuan", "kırmızı et", "kirmizi et"}


def _expand_clusters(clusters):
    lookup = {}
    for cluster in clusters:
        for word in cluster:
            lookup[word] = frozenset(cluster - {word})
    return lookup


SYNONYMS_DICT = _expand_clusters(_SYNONYM_CLUSTERS)
COLOR_SYNONYMS = _expand_clusters(_COLOR_SYNONYM_CLUSTERS)
MATERIAL_SYNONYMS = _expand_clusters(_MATERIAL_SYNONYM_CLUSTERS)


def is_stem_match(q_tok, t_tok, min_len=4, drop=2):
    """Cheap fuzzy match for suffix/typo variation beyond what stem() strips:
    two tokens match if their first `check_len` characters agree, where
    check_len backs off by up to `drop` chars from the shorter token's length."""
    if len(q_tok) < min_len or len(t_tok) < min_len:
        return False
    check_len = max(min_len, min(len(q_tok), len(t_tok)) - drop)
    return q_tok[:check_len] == t_tok[:check_len]


def expand_query_tokens(query_tokens):
    """Query-side-only expansion: original tokens + synonym-cluster members +
    adjacent-bigram merges ("multi", "vitamin" -> also add "multivitamin").
    This is the key design difference from a teammate's per-pair
    `get_expanded_match` loop: because the set of UNIQUE queries is small and
    bounded (terms.csv, not the 3.36M submission rows), this expansion is done
    ONCE PER UNIQUE QUERY in features.py's LexicalIndex, and the resulting
    expanded token set is run through the existing CountVectorizer/sparse-matrix
    machinery just like plain word_overlap -- so synonym/compound-aware overlap
    on all 3.36M pairs stays a vectorized sparse matrix operation, not a
    3.36M-iteration Python loop. See features.py LexicalIndex for where this
    is actually used."""
    result = set(query_tokens)
    for tok in query_tokens:
        syns = SYNONYMS_DICT.get(tok)
        if syns:
            result |= syns
    for i in range(len(query_tokens) - 1):
        result.add(query_tokens[i] + query_tokens[i + 1])
    return result


def get_expanded_match(query_tokens, target_tokens):
    """Count query tokens "covered" by target_tokens, expanding coverage through:
    (1) bigram merges ("multi" + "vitamin" -> "multivitamin" as one covered unit,
        catching Turkish spacing/compounding mismatches -- see original src/
        docstring on char n-grams for the same underlying problem);
    (2) the domain SYNONYMS_DICT cluster lookup;
    (3) a bounded stem-match fallback.
    `target_tokens` should be a set/frozenset for O(1) membership checks.
    Returns an integer count in [0, len(query_tokens)]."""
    target_tokens = target_tokens if isinstance(target_tokens, (set, frozenset)) else set(target_tokens)
    n = len(query_tokens)
    count = 0
    matched = set()

    i = 0
    while i < n - 1:
        merged = query_tokens[i] + query_tokens[i + 1]
        if merged in target_tokens:
            count += 2
            matched.add(i)
            matched.add(i + 1)
            i += 2
            continue
        i += 1

    for idx, tok in enumerate(query_tokens):
        if idx in matched:
            continue
        if tok in target_tokens:
            count += 1
            continue
        syns = SYNONYMS_DICT.get(tok)
        if syns and (syns & target_tokens):
            count += 1
            continue
        if any(is_stem_match(tok, t) for t in target_tokens):
            count += 1

    return count
