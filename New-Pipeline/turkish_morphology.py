"""
Real Turkish morphological analysis: root (lemma) + suffix (morpheme) extraction.

WHY THIS EXISTS: Claude-src/text_utils.py's `stem()` is a naive suffix-stripper
(pattern-based, not a real morphological analyzer). "ayakkabılarım" (my shoes)
needs to reduce to the same root as "ayakkabı" (shoe) and "ayakkabıya" (to the
shoe) for lexical-overlap features to see them as related -- a naive stemmer
gets this wrong far more often than a real morphology engine does, because
Turkish is agglutinative (a single root can take a long, ordered chain of
suffixes: ev-ler-i-nden = ev[house]+ler[plural]+i[3rd-person-possessive]+nden
[ablative, "from"] -- four morphemes stacked on one root).

TWO-ENGINE DESIGN (verified working in this sandbox, 2026-07-07):

1. `zeyrek` (pip-installable, pure Python, MIT-style license) -- a partial port
   of Zemberek-NLP's morphology to Python. Verified working here: real root+
   suffix extraction, e.g. "ayakkabılarım" -> lemma="ayakkabı",
   morphemes=[Noun, A3pl, P1sg]. Alpha-stage, unmaintained >12 months per its
   PyPI page -- but it WORKS, needs no JVM, no external downloads, and no
   network access at runtime, which matters because this sandbox's outbound
   network is allowlist-restricted (api.github.com, raw.githubusercontent.com,
   repo1.maven.org all return 403 here) and Kaggle sessions reset packages
   between runs (see Claude-src/DESIGN.md lesson 9b for the exact same class
   of problem with sentence-transformers).

2. Full `Zemberek-NLP` (Java, via JPype) -- the actively-maintained original,
   with proper morphological disambiguation (picks the single correct parse
   in context, not just "first parse found") and a much larger lexicon. NOT
   used as the primary engine here because it requires downloading
   `zemberek-full.jar` (~30-50MB) at setup time, which this sandbox's network
   allowlist blocks (github release assets, raw.githubusercontent.com, and
   Maven Central were all tested and blocked here 2026-07-07). Kaggle
   sessions have full internet access, so this IS a viable upgrade path there
   -- see "Upgrading to full Zemberek" below -- but it could not be verified
   end-to-end in this sandbox, so it is not the default.

DEFAULT: zeyrek. If `USE_FULL_ZEMBEREK=1` env var is set AND jpype+the jar are
available, this module will attempt to use full Zemberek instead and fall
back to zeyrek with a loud warning if that fails -- never silently downgrade
without saying so (see Claude-src/DESIGN.md's whole "Dersler" section for why
this project treats silent fallbacks as a real risk, not a convenience).

KNOWN LIMITATION (verified 2026-07-07): zeyrek's analyzer expects proper
Turkish characters (ı/i, ş/s, ğ/g, ü/u, ö/o, ç/c distinguished). ASCII-typed
queries like "kirmizi" (should be "kırmızı") return NO PARSE at all -- this is
exactly why `typo_tolerance.py` exists as a REQUIRED preprocessing step, not
an optional add-on. Always deasciify-correct before calling this module on
raw user query text; item titles (professionally written) usually don't need
it but query text (user-typed) usually does.

zeyrek also prints noisy "APPENDING RESULT: ..." debug lines on every single
parse call, with no verbosity flag to disable it (checked its source -- it's
a bare `print()`, not gated by logging). VERIFIED 2026-07-07: neither
`contextlib.redirect_stdout` NOR raw fd-level `os.dup2(devnull, 1)` suppress
it -- something in zeyrek's parse path writes through a file handle that
doesn't respect either suppression method (not yet root-caused; both
suppression attempts are still applied below since they're harmless and may
help in some environments, but don't rely on them). At 3.36M rows this WILL
produce unusable log volume. Practical workaround for batch runs on Kaggle:
pipe the whole script's stdout through a filter at the shell level, e.g.
`!python build_turkish_features.py 2>&1 | grep -v "^APPENDING RESULT"`, or
redirect to a file and grep it out afterward if you need other stdout lines
preserved live.
"""
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from typing import Optional

_ZEYREK_ANALYZER = None
_ENGINE = None  # set to "zeyrek" or "zemberek_full" once initialized


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
    """Attempt to load real Zemberek-NLP via JPype. Requires:
    1. `pip install jpype1`
    2. `zemberek-full.jar` downloaded and its path set via ZEMBEREK_JAR_PATH
       env var (get it from https://github.com/ahmetaa/zemberek-nlp releases
       -- on Kaggle, `!wget <release-url> -O zemberek-full.jar` works fine;
       this sandbox's network allowlist blocks that download, so this path
       is untested end-to-end here).
    Raises on any failure -- caller decides whether to fall back."""
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
    """Call once at module/process start. `prefer_full_zemberek` defaults to
    reading USE_FULL_ZEMBEREK env var if not passed explicitly."""
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
    """Analyze a single, already-tokenized word (NOT a full sentence -- do your
    own sentence/word splitting upstream, e.g. Claude-src's text_utils.tokenize).
    Returns the FIRST candidate parse zeyrek finds -- zeyrek does not implement
    Zemberek's disambiguation module, so on genuinely ambiguous words (e.g.
    "kırmızı" can parse as Adj OR as a Noun with an accusative suffix) this is
    a heuristic choice, not a resolved one. For e-commerce query/title text
    this matters less than in general text (product attributes are rarely
    grammatically ambiguous in the way narrative sentences are), but don't
    trust this for anything requiring true disambiguation."""
    if _ZEYREK_ANALYZER is None:
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
    non-empty input -- callers building overlap features need a stable key)."""
    return analyze_word(word).lemma


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
