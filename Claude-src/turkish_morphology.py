"""
Real Turkish morphological analysis: root (lemma) + suffix (morpheme) extraction.

Only imported/used when config.USE_TURKISH_MORPHOLOGY=1 (see features.py) --
this whole module is an optional, flag-gated addition, same pattern as
02_llm_enrichment.py (ADR-02). Ported in from New-Pipeline/ (built and unit-
tested there 2026-07-07 in a separate dev sandbox before touching this proven
pipeline) -- see New-Pipeline/DESIGN.md for the fuller narrative of what was
tried and verified.

WHY THIS EXISTS: text_utils.py's `stem()` is a naive suffix-stripper (pattern-
based, not a real morphological analyzer). "ayakkabılarım" (my shoes) needs to
reduce to the same root as "ayakkabı" (shoe) and "ayakkabıya" (to the shoe)
for lexical-overlap features to see them as related -- a naive stemmer gets
this wrong far more often than a real morphology engine does, because Turkish
is agglutinative (a single root can take a long, ordered chain of suffixes:
ev-ler-i-nden = ev[house]+ler[plural]+i[3rd-person-possessive]+nden[ablative,
"from"] -- four morphemes stacked on one root).

TWO-ENGINE DESIGN (verified working in New-Pipeline's dev sandbox, 2026-07-07):

1. `zeyrek` (pip-installable, pure Python) -- a partial port of Zemberek-NLP's
   morphology to Python. Verified: "ayakkabılarım" -> lemma="ayakkabı",
   morphemes=[Noun, A3pl, P1sg]. Alpha-stage, unmaintained >12 months per its
   PyPI page -- but it WORKS, needs no JVM, no external downloads, no network
   access at runtime. This is the DEFAULT engine.

2. Full `Zemberek-NLP` (Java, via JPype) -- the actively-maintained original,
   with proper morphological disambiguation and a larger lexicon. NOT the
   default because it requires downloading `zemberek-full.jar` (~30-50MB) at
   setup time -- untested end-to-end (the dev sandbox's network allowlist
   blocked every source tried: GitHub release assets, raw.githubusercontent.com,
   Maven Central). Kaggle has full internet access, so this is a viable
   upgrade path there if zeyrek's quality turns out to be the bottleneck, but
   verify it actually works before trusting it -- don't assume.

KNOWN LIMITATIONS (verified, not assumed):
- zeyrek cannot parse ASCII-folded input ("kirmizi") at all -- NO PARSE.
  This is why typo_tolerance.py's correction MUST run before this on query
  text (item titles are professionally written and usually don't need it).
- No disambiguation -- ambiguous words return a first-found parse, not a
  resolved one. Matters less for e-commerce query/title text than narrative
  prose, but is a real gap.
- Prints "APPENDING RESULT: ..." debug lines per parse call. ROOT-CAUSED
  2026-07-07 (real Kaggle run at 345k unique tokens produced a 60k+ line log
  dump of nothing but this): it's not a stdout print at all, it's
  `logger.warning(...)` from `zeyrek/rulebasedanalyzer.py` -- that's exactly
  why `contextlib.redirect_stdout`/`os.dup2` on fd 1 never touched it
  (Python's logging module writes through its own handler, independent of
  whatever fd `sys.stdout` currently points at). Fixed at the source: `init()`
  now sets `logging.getLogger("zeyrek").setLevel(logging.ERROR)` once, which
  silences it via the standard logger hierarchy (zeyrek.rulebasedanalyzer is
  a child logger, so this covers it without needing per-module calls).
- INTEGRATION NOTE (see features.py): this is only ever called ONCE PER
  UNIQUE TOKEN across the whole item/query catalog at LexicalIndex build
  time, never per training/submission ROW -- the catalog's unique vocabulary
  is orders of magnitude smaller than 3.36M rows, which is what makes zeyrek's
  real (unbenchmarked-at-row-scale) per-call speed tolerable here at all.
"""
import logging
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from typing import Optional

# See module docstring's "Prints APPENDING RESULT..." note: zeyrek's noise is
# logging.warning(), not a print -- silence it via the logger hierarchy, not
# stdout/fd tricks (those never worked because they were fixing the wrong
# layer). Set unconditionally at import time, before any analyzer is built.
logging.getLogger("zeyrek").setLevel(logging.ERROR)

_ZEYREK_ANALYZER = None
_ENGINE = None  # set to "zeyrek" or "zemberek_full" once initialized
_ROOT_CACHE = {}  # word -> lemma, populated lazily; see build_root_cache() for the batch entry point


@dataclass
class MorphResult:
    word: str
    lemma: str
    pos: str  # "Noun", "Adj", "Verb", "Adv", "Unk", ...
    morphemes: list = field(default_factory=list)
    parsed: bool = True


def _init_zeyrek():
    global _ZEYREK_ANALYZER, _ENGINE
    if _ZEYREK_ANALYZER is not None:
        return
    import zeyrek
    with redirect_stdout(StringIO()):
        _ZEYREK_ANALYZER = zeyrek.MorphAnalyzer()
    _ENGINE = "zeyrek"


def _init_full_zemberek():
    """Attempt to load real Zemberek-NLP via JPype. Requires `pip install
    jpype1` plus zemberek-full.jar downloaded with ZEMBEREK_JAR_PATH pointed
    at it (see https://github.com/ahmetaa/zemberek-nlp/releases). Raises on
    any failure -- caller decides whether to fall back."""
    import jpype
    global _ZEYREK_ANALYZER, _ENGINE
    jar_path = os.environ.get("ZEMBEREK_JAR_PATH", "")
    if not jar_path or not os.path.exists(jar_path):
        raise FileNotFoundError(
            f"ZEMBEREK_JAR_PATH={jar_path!r} not found. Download zemberek-full.jar "
            "from https://github.com/ahmetaa/zemberek-nlp/releases and set this "
            "env var to its path."
        )
    if not jpype.isJVMStarted():
        jpype.startJVM(classpath=[jar_path])
    TurkishMorphology = jpype.JClass("zemberek.morphology.TurkishMorphology")
    _ZEYREK_ANALYZER = TurkishMorphology.createWithDefaults()
    _ENGINE = "zemberek_full"


def init(prefer_full_zemberek: Optional[bool] = None):
    """Call once at process start. `prefer_full_zemberek` defaults to reading
    USE_FULL_ZEMBEREK env var if not passed explicitly."""
    global _ENGINE
    if _ENGINE is not None:
        return
    if prefer_full_zemberek is None:
        prefer_full_zemberek = os.environ.get("USE_FULL_ZEMBEREK", "0") == "1"
    if prefer_full_zemberek:
        try:
            _init_full_zemberek()
            print("[turkish_morphology] using full Zemberek-NLP (JPype)")
            return
        except Exception as e:
            print(f"[turkish_morphology] WARNING: full Zemberek unavailable ({e!r}) "
                  "-- falling back to zeyrek. This is a REAL quality reduction "
                  "(no disambiguation, smaller lexicon), not a transparent swap.")
    _init_zeyrek()
    print("[turkish_morphology] using zeyrek (pure Python, alpha-stage port)")


def analyze_word(word: str) -> MorphResult:
    """Analyze a single, already-tokenized word. Returns the FIRST candidate
    parse zeyrek finds (no disambiguation available -- see module docstring).
    Prefer `root_of()` with the batch cache for real pipeline use; this
    function always does a fresh (uncached) analysis."""
    if _ENGINE is None:
        init()
    if not word or not isinstance(word, str):
        return MorphResult(word=word, lemma=word, pos="Unk", morphemes=[], parsed=False)

    if _ENGINE == "zeyrek":
        with redirect_stdout(StringIO()):
            analyses = _ZEYREK_ANALYZER._parse(word)
        if not analyses:
            return MorphResult(word=word, lemma=word.lower(), pos="Unk", morphemes=[], parsed=False)
        a = analyses[0]
        morphemes = [m[0].id_ for m in a.morphemes]
        return MorphResult(word=word, lemma=a.dict_item.lemma, pos=a.pos.value,
                            morphemes=morphemes, parsed=True)
    elif _ENGINE == "zemberek_full":
        result = _ZEYREK_ANALYZER.analyzeAndDisambiguate(word)
        best = result.bestAnalysis()
        if best is None or len(best) == 0:
            return MorphResult(word=word, lemma=word.lower(), pos="Unk", morphemes=[], parsed=False)
        analysis = best[0]
        lemma = str(analysis.getDictionaryItem().lemma)
        pos = str(analysis.getDictionaryItem().primaryPos)
        morphemes = [str(m) for m in analysis.getMorphemes()]
        return MorphResult(word=word, lemma=lemma, pos=pos, morphemes=morphemes, parsed=True)
    raise RuntimeError(f"unknown engine state: {_ENGINE!r}")


def root_of(word: str) -> str:
    """Convenience: just the lemma/root, falling back to the lowercased
    original word if no parse was found (never returns None/empty for a
    non-empty input -- callers building overlap features need a stable key).
    Uses the module-level cache -- call build_root_cache() first with the
    full vocabulary for real pipeline use so repeated words are O(1) instead
    of re-invoking zeyrek every time."""
    if word in _ROOT_CACHE:
        return _ROOT_CACHE[word]
    result = analyze_word(word)
    _ROOT_CACHE[word] = result.lemma
    return result.lemma


def build_root_cache(vocab_tokens, verbose=True):
    """Batch-populate the root cache for every token in `vocab_tokens` (an
    iterable of strings, duplicates OK -- this is meant to be called with
    ALL unique tokens across the item+query catalog ONCE, at LexicalIndex
    build time, not per-row). Returns the number of NEW tokens actually
    analyzed (tokens already cached are skipped)."""
    if _ENGINE is None:
        init()
    unique_new = {t for t in vocab_tokens if t and t not in _ROOT_CACHE}
    n = len(unique_new)
    if verbose:
        print(f"[turkish_morphology] analyzing {n:,} new unique tokens "
              f"({len(_ROOT_CACHE):,} already cached)...")
    for i, word in enumerate(unique_new):
        _ROOT_CACHE[word] = analyze_word(word).lemma
        if verbose and n > 5000 and (i + 1) % 5000 == 0:
            print(f"  ...{i+1:,}/{n:,}")
    return n


def engine_name() -> str:
    return _ENGINE or "(not initialized)"


if __name__ == "__main__":
    init()
    test_words = [
        "ayakkabılarım", "kırmızıyı", "evlerinden", "koşarken", "kitaplık",
        "telefonu", "siyah", "pantolonlar", "gömlekleri", "çantasında",
    ]
    print(f"\nengine = {engine_name()}\n")
    for w in test_words:
        r = analyze_word(w)
        print(f"{w:16s} -> lemma={r.lemma!r:14s} pos={r.pos:6s} "
              f"morphemes={r.morphemes} parsed={r.parsed}")
