"""
Turkish query-text normalization: diacritic/typo tolerance BEFORE morphological
analysis (see turkish_morphology.py's docstring for why this is required, not
optional -- zeyrek returns NO PARSE at all for ASCII-folded input like
"kirmizi" when it should match "kırmızı"). Only imported/used when
config.USE_TURKISH_MORPHOLOGY=1 (see features.py). Ported in from
New-Pipeline/ (built and unit-tested there 2026-07-07) -- see
New-Pipeline/DESIGN.md for the fuller narrative.

TWO SEPARATE PROBLEMS, TWO SEPARATE FIXES (conflating them gives worse results
than solving them separately -- verified empirically in New-Pipeline/tests/):

1. DIACRITIC DROPPING ("kirmizi" for "kırmızı", "ayakkabi" for "ayakkabı").
   Fixed via `deasciify()` (fold both the query word AND every vocabulary word
   to ASCII, then EXACT match on the folded form) -- high-precision, NOT
   edit-distance-based on purpose. VERIFIED: symspellpy at max_edit_distance=2
   does NOT find "kırmızı" from "kirmizi" (true edit distance is 3) -- this
   needs its own path, edit-distance alone doesn't solve it.

2. GENUINE TYPOS (extra/missing/transposed letters: "telefonn", "pantalon"
   for "pantolon"). Fixed via `symspellpy` edit-distance lookup.

VOCABULARY SOURCE: built from THIS PROJECT'S OWN corpus (item titles +
queries), not a generic downloaded Turkish frequency dictionary -- a domain
vocabulary won't "correct" a real brand name or product-specific term into an
unrelated common word, same conservative philosophy as this file's
SIZE_VOCAB trigger-word gating in features.py and the reverted brand hard-
override (see DESIGN.md lesson 1).
"""
import re
from collections import Counter

from symspellpy import SymSpell, Verbosity

_DEASCII_MAP = str.maketrans({
    "ı": "i", "İ": "i", "I": "i",
    "ş": "s", "Ş": "s",
    "ğ": "g", "Ğ": "g",
    "ü": "u", "Ü": "u",
    "ö": "o", "Ö": "o",
    "ç": "c", "Ç": "c",
})

_WORD_RE = re.compile(r"[a-zçğıöşü]+", re.IGNORECASE)


def deasciify(text: str) -> str:
    """Fold Turkish-specific characters to their ASCII look-alikes. Lossy and
    intentionally so -- this is a LOOKUP KEY, never shown to a user or used
    as a corrected value itself."""
    return text.lower().translate(_DEASCII_MAP)


def tokenize_simple(text: str) -> list:
    """Deliberately simple word splitter for vocabulary building -- use
    text_utils.tokenize for the real pipeline's own tokenization; this is
    just for counting frequency when building the correction vocabulary."""
    if not isinstance(text, str):
        return []
    return _WORD_RE.findall(text.lower())


class TurkishTypoCorrector:
    def __init__(self, max_edit_distance: int = 2):
        self.vocab_counts = Counter()
        self._deascii_index = {}  # deasciified form -> most frequent real form(s)
        self._max_edit_distance = max_edit_distance
        self._sym = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
        self._built = False

    def build_vocab_from_corpus(self, texts):
        """texts: iterable of raw strings (item titles, queries, attributes...).
        Call once with the FULL corpus before using correct()."""
        for t in texts:
            self.vocab_counts.update(tokenize_simple(t))
        self._rebuild_indices()
        return self

    def add_words(self, words_with_counts: dict):
        """Manually seed/extend the vocabulary (e.g. from a curated brand list
        that must never be auto-corrected away) without a full corpus pass."""
        self.vocab_counts.update(words_with_counts)
        self._rebuild_indices()
        return self

    def _rebuild_indices(self):
        self._deascii_index = {}
        for word, count in self.vocab_counts.items():
            key = deasciify(word)
            cur = self._deascii_index.get(key)
            if cur is None or count > cur[1]:
                self._deascii_index[key] = (word, count)
        self._sym = SymSpell(max_dictionary_edit_distance=self._max_edit_distance, prefix_length=7)
        for word, count in self.vocab_counts.items():
            self._sym.create_dictionary_entry(word, count)
        self._built = True

    def correct(self, word: str, min_vocab_count: int = 1) -> str:
        """Returns the best-guess corrected form, or the original word
        unchanged if nothing better is found -- NEVER raises, NEVER returns
        None/empty for non-empty input (same "stable key" contract as
        turkish_morphology.root_of)."""
        if not self._built:
            raise RuntimeError("call build_vocab_from_corpus() or add_words() first")
        if not word or not isinstance(word, str):
            return word

        w = word.lower()
        if self.vocab_counts.get(w, 0) >= min_vocab_count:
            return w  # already a known, real word -- don't touch it

        # Step 1: diacritic-fold exact match (high precision, not edit-distance)
        key = deasciify(w)
        hit = self._deascii_index.get(key)
        if hit is not None:
            return hit[0]

        # Step 2: genuine-typo fallback via edit distance
        suggestions = self._sym.lookup(w, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            return suggestions[0].term

        return w  # nothing found -- leave it alone rather than guess wildly


if __name__ == "__main__":
    corrector = TurkishTypoCorrector()
    corrector.build_vocab_from_corpus([
        "kırmızı ayakkabı", "siyah telefon kılıfı", "kadın elbise",
        "erkek pantolon", "mavi gömlek", "beyaz çanta",
    ])
    tests = ["kirmizi", "ayakkabi", "telefonn", "pantalon", "siyahh",
             "gomlek", "canta", "elbise", "nike"]
    for t in tests:
        print(f"{t:12s} -> {corrector.correct(t)}")
