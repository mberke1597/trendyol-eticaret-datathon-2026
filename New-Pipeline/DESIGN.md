# New-Pipeline: Turkish morphology + typo tolerance experiment

**Status:** foundation built and unit-tested in this dev sandbox (2026-07-07).
NOT yet integrated into `Claude-src/`, NOT yet tested against real competition
data (items.csv/training_pairs.csv aren't in this sandbox). Treat everything
here as verified-in-isolation, not verified-on-real-LB -- same discipline as
the rest of this project: every claim below is scoped to what was actually
run, and the real test is a Kaggle submission, not this document.

## Why this folder exists

Berke asked to seriously try three things surfaced by researching Trendyol's
own public writeup about this exact competition (see `Claude-src/`'s chat
history for the sourced findings): Turkish-specific morphological analysis
(root+suffix understanding, not naive stemming), typo/diacritic tolerance, and
Polars for reliability. This folder is a clean-room space to build and prove
out the Turkish NLP pieces before touching the working 0.891 pipeline in
`Claude-src/` -- consistent with this project's established pattern (see
`Claude-src/ADR.md` ADR-03) of building new capability as standalone,
independently-testable pieces before wiring anything into the proven path.

## What's built

### `turkish_morphology.py` -- root/suffix extraction

Real morphological analysis via `zeyrek` (pip-installable partial Python port
of Zemberek-NLP). Verified working in this sandbox:

```
ayakkabılarım  -> lemma='ayakkabı'  morphemes=['Noun','A3pl','P1sg']
evlerinden     -> lemma='ev'        morphemes=['Noun','A3sg','P3pl','Abl']
kırmızıyı      -> lemma='kırmızı'   morphemes=['Noun','A3sg','Acc']
```

This is a real improvement over `Claude-src/text_utils.py`'s `stem()`, which
is a pattern-based suffix-stripper, not a morphological analyzer -- it will
get agglutinative chains (multiple stacked suffixes) wrong more often,
especially on the possessive/case suffix combinations that are extremely
common in both query text ("ayakkabılarım", "elbisemi") and item titles
("çocuklarınız için").

**Known limitations (verified, not assumed):**
- zeyrek is alpha-stage, unmaintained on PyPI for 12+ months. It works, but
  don't expect fixes if it breaks.
- No disambiguation module (unlike full Zemberek) -- ambiguous words return
  a *first-found* parse, not a resolved one. Checked: this matters less for
  e-commerce query/title text than for narrative prose (product attributes
  are rarely as grammatically ambiguous as full sentences), but it's a real
  gap, not a solved one.
- Cannot parse ASCII-folded input ("kirmizi") at all -- returns no analysis.
  This is WHY `typo_tolerance.py` exists as a mandatory upstream step, not an
  optional nicety.
- Prints debug noise per parse call. ROOT-CAUSED during Claude-src
  integration (2026-07-07, triggered by a real Kaggle run at 345k unique
  tokens producing a 60k+ line log of nothing else): it's `logger.warning()`
  in `zeyrek/rulebasedanalyzer.py`, not a stdout print -- that's exactly why
  `contextlib.redirect_stdout`/`os.dup2` on fd 1 never suppressed it (Python's
  logging module writes through its own handler, independent of `sys.stdout`).
  Fixed via `logging.getLogger("zeyrek").setLevel(logging.ERROR)`, set once at
  `turkish_morphology.py` import time. See `Claude-src/turkish_morphology.py`.

**Upgrade path not yet verified:** full Zemberek-NLP (Java, via JPype) has
real disambiguation and a larger lexicon. The code path exists
(`USE_FULL_ZEMBEREK=1` + `ZEMBEREK_JAR_PATH`) but could NOT be tested
end-to-end here -- this sandbox's network allowlist blocked every source
tried for the required jar (GitHub release assets, raw.githubusercontent.com,
Maven Central all returned 403). Kaggle has full internet access, so this is
worth trying there if zeyrek's quality turns out to be the bottleneck --
but don't assume it works until it's actually been run once.

### `typo_tolerance.py` -- diacritic + typo correction

Two separate mechanisms for two separate problems (conflating them is worse
than solving them separately -- see empirical proof below):

1. **Diacritic dropping** ("kirmizi" for "kırmızı") -- fixed via
   deasciify-then-exact-match against a corpus-built vocabulary, NOT
   edit-distance. Verified this needed its own path: `symspellpy` at
   `max_edit_distance=2` returns NO match for ("kirmizi", "kırmızı") because
   the true edit distance is 3 (three ı/i substitutions) -- see
   `tests/test_typo_tolerance.py::test_edit_distance_does_not_over_correct_diacritic_case`
   for the regression-guarded proof.
2. **Genuine typos** ("telefonn", "pantalon") -- fixed via `symspellpy`
   edit-distance lookup, which IS the right tool for this class of error.

Vocabulary is built from **this project's own corpus**, not a downloaded
generic Turkish dictionary -- partly because generic dictionary downloads
were also blocked by this sandbox's network allowlist, but mainly because a
domain vocabulary is genuinely better here: it won't "correct" a real brand
name into an unrelated common word, mirroring the same conservative,
corpus-grounded philosophy already established in `Claude-src/features.py`'s
`SIZE_VOCAB` trigger-word gating and the reverted brand hard-override (see
`Claude-src/DESIGN.md` lesson 1).

**CORRECTION (found during Claude-src integration, 2026-07-07):** the
corpus must be **item text ONLY, never query text**. Building it from item
titles + query text (as this section originally said) is a real bug: a
query's own typo (e.g. "ayakkabi", missing the dotless-ı) gets counted into
`vocab_counts` just by appearing in the query corpus, so `correct()`'s
fast-path (`vocab_counts.get(w, 0) >= min_vocab_count: return w`) treats it
as an "already known real word" and never reaches the deasciify/edit-distance
logic -- silently disabling correction for exactly the words that need it
most. Item titles are professionally written and typo-free, so they're the
only safe source. See `Claude-src/DESIGN.md` lesson #9 and
`Claude-src/tests/test_features.py::test_turkish_morphology_root_overlap_recovers_typo_match`
for the regression-guarded proof.

Verified end-to-end combined pipeline (typo-correct, THEN extract root):

```
'kirmizi'        -> corrected='kırmızı'       -> root='kırmızı'
'ayakkabilarim'  -> corrected='ayakkabılarım' -> root='ayakkabı'
'siyahh'         -> corrected='siyah'         -> root='siyah'
'gomlekk'        -> corrected='gömlek'        -> root='gömlek'
```

### Tests

16 tests, all passing (`pytest tests/ -v`), covering: correct root extraction
across plural/possessive/ablative/accusative suffix chains, graceful fallback
on unparseable input (never raises, never returns empty), the diacritic-vs-
edit-distance split (with a regression guard proving edit-distance alone
doesn't solve it), and conservative behavior on genuinely unknown words
(doesn't "correct" `"nike"` into an unrelated vocabulary word).

## What's NOT done yet (honest status, not a to-do list dressed as done)

1. **DONE, RESULT NEGATIVE (2026-07-07).** Tested against real data via an
   actual Kaggle leaderboard submission (`TY_USE_TURKISH_MORPHOLOGY=1`,
   correct round-2 pairs file): **0.879 vs. the 0.891 baseline, a real
   -0.012 regression.** The proba array differed meaningfully from baseline
   (corr=0.90), so this was a genuine model-behavior change, not a no-op --
   it just made things worse, not better. Working theory: root-overlap is
   coarser than word/stem overlap and adds lexical-match noise (two
   unrelated products can share a bare root after suffix stripping) rather
   than recall, for this catalog's actual query patterns. See
   `Claude-src/DESIGN.md` lesson #11 for the full account. **Decision:
   `config.USE_TURKISH_MORPHOLOGY` stays off (default); this feature is not
   being pursued further without a more targeted redesign** (e.g. only
   firing root-overlap when word/stem overlap is exactly 0, rather than
   always adding it as an extra column).
2. **DONE (2026-07-07): integrated into `Claude-src/features.py`.**
   `turkish_morphology.py` and `typo_tolerance.py` were copied into
   `Claude-src/`, and `LexicalIndex.__init__` now builds a typo corrector +
   root cache (gated behind `config.USE_TURKISH_MORPHOLOGY`, off by default)
   producing two new features, `root_overlap_n`/`root_recall`, computed the
   same way as `word_overlap_n`/`stem_overlap_n`. The item-vs-query vocab bug
   described above was caught here, before any real run, via a toy-catalog
   integration test. Full existing test suite (52 tests) still passes with
   the flag off -- zero behavior change for anyone who hasn't opted in. See
   `Claude-src/DESIGN.md` lesson #9 for the full account.
3. **Throughput at 3.36M rows was never separately benchmarked** -- moot now
   given item 1's negative result, but for the record: the real run analyzed
   345k unique catalog tokens once (not per-row, per the caching design), and
   completed; the un-suppressable log noise during that run (60k+ lines) was
   root-caused and fixed separately (see `Claude-src/DESIGN.md` lesson #10).
4. **Polars** (the third thing asked for) -- not started. Lower priority than
   the two NLP pieces since it's a reliability/speed lever, not a model-
   quality lever; suggest tackling it only after the morphology/typo work is
   validated to actually move the score, so GPU/Kaggle-session time isn't
   split three ways before any one direction is proven.

## Recommended next steps (in order)

1. Upload a real sample of `items.csv` + `training_pairs.csv` (or run on
   Kaggle directly) and benchmark: (a) `root_of()` throughput on ~10k real
   query/title tokens, (b) how often `typo_tolerance.correct()` actually
   changes a token on real query text (a fire-rate check, same discipline as
   `Claude-src/ILERLEME_PLANI.md`'s recommendation to check `size_conflict`'s
   fire-rate before committing to a full retrain -- don't retrain on a
   feature that rarely fires).
2. If fire-rate and throughput look reasonable, add `root_overlap_n` (and a
   typo-corrected variant of the existing overlap features) to
   `Claude-src/features.py`, following the exact pattern already used for
   `color_match`/`size_match` (a vocab-indexed sparse overlap, see
   `_small_vocab_matrix`/`_sparse_row_overlap` in that file).
3. Rebuild features + retrain + real Kaggle submission, A/B against the
   0.891 baseline. Do not trust OOF for this decision -- same rule as
   everywhere else in this project (`Claude-src/DESIGN.md`'s whole point).
4. Only then consider Polars, and only if the above genuinely helped and
   more Kaggle time is being spent on this direction.
