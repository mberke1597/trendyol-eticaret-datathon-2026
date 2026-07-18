"""
Vectorized feature engineering shared by the train-set builder and the 3.36M-row
submission builder.

Design goal (prompt.md section 3, unchanged from ../src/features.py): no per-pair
Python loop over the hot path. Everything is row-aligned sparse-matrix elementwise
multiplication, batched numpy dot products, or small fixed-vocab indicator
matrices.

NEW vs. ../src/features.py (each grounded in a specific finding -- see
Claude-src/DESIGN.md for the full writeup):

  - Category-path literal overlap (`cat_*` features): 71.8% of positive training
    pairs have a query token overlapping SOME level of the item's category path,
    64.7% at the deepest (most specific) level -- measured directly on this
    competition's data. Only a *semantic* (embedding) category similarity existed
    before; this adds the literal/lexical counterpart, same as title already had.
  - Synonym- and compound-aware overlap (`*_expanded_*` features): a teammate's
    train_stacking.py had a good domain-synonym dictionary and bigram-merge logic
    but applied it with a per-pair Python loop (get_expanded_match called once per
    of up to 3.36M rows). Here the expansion (text_utils.expand_query_tokens) is
    done ONCE PER UNIQUE QUERY when LexicalIndex is built, then run through the
    same vectorized sparse-matrix machinery as plain word_overlap -- richer
    matching, zero extra per-row Python cost.
  - `brand_contradiction`: BrandMatcher (item_meta.py) was fully implemented by
    the same teammate but never wired into their feature pipeline. It's used
    here, with a masked partial loop (only rows whose query actually names a
    brand go through Python -- typically a small fraction of rows, not all
    3.36M) instead of either a full per-row loop or leaving it unused.
  - `is_generic_query`: flags ambiguous queries ("hediye", "dekor", ...) that
    match almost anything by embedding similarity alone.
  - Optional LLM-derived soft features (`llm_gender_soft_match`, etc.): only
    added if 02_llm_enrichment.py was actually run (cache files present) --
    see LexicalIndex.__init__ for the graceful no-op fallback.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from item_meta import (  # noqa: E402
    BrandMatcher,
    COLOR_SYNONYMS,
    MATERIAL_SYNONYMS,
    age_intent_from_tokens,
    expand_with_synonyms,
    extract_query_constraints,
    gender_intent_from_tokens,
    has_color_collocation,
    is_generic_query,
    normalize_age,
    normalize_gender,
    parse_item_constraints,
)
from text_utils import category_levels, char_ngrams, expand_query_tokens, normalize, stem, tokenize  # noqa: E402

AGE_CODES = {"unknown": 0, "yetiskin": 1, "cocuk": 2, "bebek": 3, "genc": 4, "bebek_cocuk": 5}
GENDER_CODES = {"unknown": 0, "kadın": 1, "erkek": 2, "unisex": 3}
INTENT_NONE = -1

# ---- NEW: size / beden matching (self-contained; no item_meta.py dependency) ----
# Mirrors the color/material/pattern constraint path: a small fixed vocab of
# canonical size tokens, an extractor that pulls sizes out of free text, and (in
# compute_batch_features) a query<->item overlap that yields size_match /
# size_conflict exactly analogous to color_match / color_conflict.
#
# Deliberately CONSERVATIVE on purpose (see DESIGN.md lessons 2/4/5 -- every
# constraint feature there was precision-checked on real data before being
# trusted, and the brand hard-override that fired on 27% of rows was reverted):
#   * Letter sizes: only the UNAMBIGUOUS multi-char tokens (xs/xl/xxl/xxxl/2xl/
#     3xl/4xl). Bare "s"/"m"/"l" are NOT treated as sizes on their own -- "m"
#     is metre, "l" litre, etc. They count only when a size trigger word is
#     present in the same text.
#   * Numeric sizes (shoe ~35-46, clothing ~34-52): only counted when a trigger
#     word (beden / numara / no) appears -- a bare "42" or "2'li" is far too
#     noisy to treat as a size otherwise.
# This keeps q_has_size (which gates size_conflict) high-precision, so the
# conflict feature does not misfire the way an over-eager numeric parser would.
SIZE_LETTER_CANON = {
    "xs": "xs", "s": "s", "m": "m", "l": "l",
    "xl": "xl", "xxl": "xxl", "2xl": "xxl", "xxxl": "xxxl", "3xl": "xxxl", "4xl": "xxxxl", "xxxxl": "xxxxl",
}
_SIZE_UNAMBIGUOUS_LETTERS = {"xs", "xl", "xxl", "2xl", "xxxl", "3xl", "4xl", "xxxxl"}
_SIZE_TRIGGER_LETTERS = {"s", "m", "l"}  # only when a trigger word co-occurs
_SIZE_TRIGGER_RE = re.compile(r"\b(beden|numara|no)\b")
_SIZE_NUM_RE = re.compile(r"\b(3[4-9]|4[0-9]|5[0-2])\b")  # plausible clothing/shoe range 34-52
SIZE_VOCAB = sorted(set(SIZE_LETTER_CANON.values()) | {f"n{i}" for i in range(34, 53)})


def extract_sizes(text):
    """Return a set of canonical size tokens found in `text`. See the block
    comment above for why numeric/bare-letter sizes are trigger-gated."""
    if not isinstance(text, str) or not text:
        return frozenset()
    t = text.lower()
    toks = set(re.findall(r"[a-z0-9]+", t))
    out = set()
    for tok in toks:
        if tok in _SIZE_UNAMBIGUOUS_LETTERS:
            out.add(SIZE_LETTER_CANON[tok])
    has_trigger = _SIZE_TRIGGER_RE.search(t) is not None
    if has_trigger:
        for tok in toks:
            if tok in _SIZE_TRIGGER_LETTERS:
                out.add(SIZE_LETTER_CANON[tok])
        for num in _SIZE_NUM_RE.findall(t):
            out.add(f"n{int(num)}")
    return frozenset(out)


def _identity_tokenizer(tokens):
    return tokens


def _fit_binary_vectorizer(token_lists, **kwargs):
    vec = CountVectorizer(
        tokenizer=_identity_tokenizer, preprocessor=lambda x: x, lowercase=False,
        token_pattern=None, binary=True, dtype=np.float32, **kwargs,
    )
    X = vec.fit_transform(token_lists)
    return vec, X


def _transform_binary(vec, token_lists):
    return vec.transform(token_lists).astype(np.float32)


def _small_vocab_matrix(sets_list, vocab):
    vidx = {w: i for i, w in enumerate(vocab)}
    rows, cols = [], []
    for r, s in enumerate(sets_list):
        for w in s:
            j = vidx.get(w)
            if j is not None:
                rows.append(r)
                cols.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(len(sets_list), len(vocab)))


class LexicalIndex:
    """Fitted once on the item catalog; holds sparse word/stem/char/category
    matrices for both items and queries plus the small-vocab constraint matrices."""

    def __init__(self, items_df, terms_df, llm_query_enrichment=None, llm_item_enrichment=None,
                 brand_min_item_count=None):
        """brand_min_item_count: passed straight through to BrandMatcher (None ->
        its production default, BrandMatcher.MIN_ITEM_COUNT=20). Exposed as a
        parameter ONLY so tests can use small synthetic catalogs (where every
        brand appears once or twice, always below the real threshold) with
        min_item_count=1 -- see tests/test_features.py -- without ever touching
        the real default that 04_build_features.py relies on."""
        import gc

        item_text = (
            items_df["title"].fillna("") + " " + items_df["brand"].fillna("") + " "
            + items_df["category"].fillna("")
        ).values
        query_text = terms_df["query"].fillna("").values
        category_text = items_df["category"].fillna("").values
        category_last_text = np.array(
            [(category_levels(c) or [""])[-1] for c in category_text], dtype=object
        )

        item_word_tok = [tokenize(s) for s in item_text]
        query_word_tok = [tokenize(s) for s in query_text]
        query_expanded_tok = [expand_query_tokens(t) for t in query_word_tok]

        self.word_vec, self.item_word_X = _fit_binary_vectorizer(item_word_tok, min_df=1)
        self.query_word_X = _transform_binary(self.word_vec, query_word_tok)
        # expanded (synonym + bigram-merge) query tokens, transformed against the SAME
        # item-fitted vocabulary -- OOV synonyms simply contribute nothing, which is
        # the correct behavior (a synonym only matters if it can appear in item text).
        self.query_word_expanded_X = _transform_binary(self.word_vec, [list(s) for s in query_expanded_tok])

        item_stem_tok = [[stem(t) for t in toks] for toks in item_word_tok]
        query_stem_tok = [[stem(t) for t in toks] for toks in query_word_tok]
        self.stem_vec, self.item_stem_X = _fit_binary_vectorizer(item_stem_tok, min_df=1)
        self.query_stem_X = _transform_binary(self.stem_vec, query_stem_tok)
        del item_stem_tok, query_stem_tok
        gc.collect()

        # ---- NEW: Turkish morphological root overlap (optional, added 2026-07-07) ----
        # OFF by default (config.USE_TURKISH_MORPHOLOGY) -- NOT yet validated
        # against a real Kaggle submission, see New-Pipeline/DESIGN.md for the
        # full rationale/verification status before trusting this. Uses real
        # morphological root extraction (turkish_morphology.py, via zeyrek)
        # instead of the naive stem() above -- "ayakkabılarım" reduces to the
        # correct root "ayakkabı" here, where stem() (pattern-based suffix
        # stripping) is more likely to miss agglutinative suffix chains.
        # Query tokens are typo/diacritic-corrected FIRST (typo_tolerance.py)
        # since zeyrek can't parse ASCII-folded input ("kirmizi") at all --
        # item titles are professionally written and skip this step. Root
        # extraction is cached ONCE PER UNIQUE TOKEN across the whole catalog
        # (build_root_cache), never per-row -- the catalog vocabulary is tiny
        # next to 3.36M training/submission rows, which is what makes zeyrek's
        # real (slow, per-call) speed tolerable here at all.
        self.has_turkish_morphology = config.USE_TURKISH_MORPHOLOGY
        if config.USE_TURKISH_MORPHOLOGY:
            import turkish_morphology as _tm
            from typo_tolerance import TurkishTypoCorrector as _TypoCorrector

            _tm.init()
            # BUG FOUND AND FIXED while integrating (2026-07-07): the typo
            # vocabulary must be built from ITEM text ONLY, never query text.
            # Including query_text seemed reasonable at first but silently
            # defeats the whole mechanism -- a query typo like "ayakkabi"
            # then gets counted as a "real, already-known word" (it appears
            # in the very corpus being vocab-built), so correct()'s fast-path
            # ("already known -- don't touch") returns it unchanged instead of
            # ever reaching the deasciify/edit-distance correction step.
            # Verified via a toy-catalog integration test before this fix:
            # root_overlap_n came out 0 for a query/item pair that obviously
            # should have matched. Item text is professionally written and is
            # the actual target we want queries corrected TOWARD, which is
            # also the more correct framing conceptually, not just the fix
            # for this bug.
            _typo_corrector = _TypoCorrector().build_vocab_from_corpus(list(item_text))
            _query_corrected_tok = [
                [_typo_corrector.correct(t) for t in toks] for toks in query_word_tok
            ]
            _vocab_for_roots = {t for toks in item_word_tok for t in toks}
            _vocab_for_roots.update(t for toks in _query_corrected_tok for t in toks)
            _tm.build_root_cache(_vocab_for_roots)

            item_root_tok = [[_tm.root_of(t) for t in toks] for toks in item_word_tok]
            query_root_tok = [[_tm.root_of(t) for t in toks] for toks in _query_corrected_tok]
            self.root_vec, self.item_root_X = _fit_binary_vectorizer(item_root_tok, min_df=1)
            self.query_root_X = _transform_binary(self.root_vec, query_root_tok)
            del _query_corrected_tok, _vocab_for_roots, item_root_tok, query_root_tok
            gc.collect()

        item_char_tok = [sorted(char_ngrams(s, n=3)) for s in items_df["title"].fillna("").values]
        query_char_tok = [sorted(char_ngrams(s, n=3)) for s in query_text]
        self.char_vec, self.item_char_X = _fit_binary_vectorizer(item_char_tok, min_df=1)
        self.query_char_X = _transform_binary(self.char_vec, query_char_tok)
        del item_char_tok, query_char_tok
        gc.collect()

        # ---- NEW: category-path literal overlap (any level + deepest level) ----
        cat_tok = [tokenize(s) for s in category_text]
        cat_last_tok = [tokenize(s) for s in category_last_text]
        self.cat_vec, self.item_cat_X = _fit_binary_vectorizer(cat_tok, min_df=1)
        self.query_cat_X = _transform_binary(self.cat_vec, query_word_tok)
        self.query_cat_expanded_X = _transform_binary(self.cat_vec, [list(s) for s in query_expanded_tok])
        self.cat_last_vec, self.item_cat_last_X = _fit_binary_vectorizer(cat_last_tok, min_df=1)
        self.query_cat_last_X = _transform_binary(self.cat_last_vec, query_word_tok)
        del cat_tok, cat_last_tok
        gc.collect()

        self.item_word_count = np.asarray(self.item_word_X.sum(axis=1)).ravel()
        self.query_word_count = np.asarray(self.query_word_X.sum(axis=1)).ravel()
        self.item_char_count = np.asarray(self.item_char_X.sum(axis=1)).ravel()
        self.query_char_count = np.asarray(self.query_char_X.sum(axis=1)).ravel()
        self.item_cat_count = np.asarray(self.item_cat_X.sum(axis=1)).ravel()
        self.item_cat_last_count = np.asarray(self.item_cat_last_X.sum(axis=1)).ravel()

        # ---- fixed-vocab constraint matrices (color/material/pattern), now with
        # query-side synonym expansion (altın<->sarı, kot<->denim, ...) ----
        item_colors, item_materials, item_patterns = parse_item_constraints(items_df["attributes"])
        q_colors, q_materials, q_patterns = [], [], []
        q_colors_expanded, q_materials_expanded = [], []
        for q_text, toks in zip(query_text, query_word_tok):
            c, m, p = extract_query_constraints(toks)
            if has_color_collocation(normalize(q_text)):
                c = frozenset()  # "beyaz eşya" etc -- not a real color reference
            q_colors.append(c)
            q_materials.append(m)
            q_patterns.append(p)
            q_colors_expanded.append(expand_with_synonyms(c, COLOR_SYNONYMS))
            q_materials_expanded.append(expand_with_synonyms(m, MATERIAL_SYNONYMS))

        from item_meta import COLOR_VOCAB, MATERIAL_VOCAB, PATTERN_VOCAB

        self.item_color_X = _small_vocab_matrix(item_colors, COLOR_VOCAB)
        self.query_color_X = _small_vocab_matrix(q_colors, COLOR_VOCAB)
        self.query_color_expanded_X = _small_vocab_matrix(q_colors_expanded, COLOR_VOCAB)
        self.item_material_X = _small_vocab_matrix(item_materials, MATERIAL_VOCAB)
        self.query_material_X = _small_vocab_matrix(q_materials, MATERIAL_VOCAB)
        self.query_material_expanded_X = _small_vocab_matrix(q_materials_expanded, MATERIAL_VOCAB)
        self.item_pattern_X = _small_vocab_matrix(item_patterns, PATTERN_VOCAB)
        self.query_pattern_X = _small_vocab_matrix(q_patterns, PATTERN_VOCAB)

        # ---- NEW: size/beden constraint matrices (self-contained, see extract_sizes) ----
        # Item side reads title + attributes (sizes live in both); query side
        # reads the raw query text. Built once per unique item/query here, then
        # matched via the same _sparse_row_overlap machinery as color/pattern.
        item_size_text = (items_df["title"].fillna("") + " " + items_df["attributes"].fillna("")).tolist()
        item_sizes = [extract_sizes(s) for s in item_size_text]
        q_sizes = [extract_sizes(q) for q in query_text]
        self.item_size_X = _small_vocab_matrix(item_sizes, SIZE_VOCAB)
        self.query_size_X = _small_vocab_matrix(q_sizes, SIZE_VOCAB)

        # ---- gender / age: metadata + title-derived fallback ----
        item_gender_meta = items_df["gender"].fillna("unknown").map(normalize_gender).values
        item_age_meta = items_df["age_group"].fillna("unknown").map(normalize_age).values
        item_gender_title = np.array(
            [gender_intent_from_tokens(t) or "unknown" for t in item_word_tok], dtype=object
        )
        item_age_title = np.array(
            [age_intent_from_tokens(t) or "unknown" for t in item_word_tok], dtype=object
        )
        self.item_gender_conflict = (
            (item_gender_meta != "unknown") & (item_gender_title != "unknown") & (item_gender_meta != item_gender_title)
        ).astype(np.float32)
        item_gender_eff = np.where(item_gender_meta != "unknown", item_gender_meta, item_gender_title)
        item_age_eff = np.where(item_age_meta != "unknown", item_age_meta, item_age_title)
        self.item_gender_code = np.array([GENDER_CODES.get(g, 0) for g in item_gender_eff], dtype=np.int8)
        self.item_age_code = np.array([AGE_CODES.get(a, 0) for a in item_age_eff], dtype=np.int8)

        q_gender_intent = [gender_intent_from_tokens(t) for t in query_word_tok]
        q_age_intent = [age_intent_from_tokens(t) for t in query_word_tok]
        self.query_gender_code = np.array(
            [GENDER_CODES.get(g, INTENT_NONE) if g else INTENT_NONE for g in q_gender_intent], dtype=np.int8
        )
        self.query_age_code = np.array(
            [AGE_CODES.get(a, INTENT_NONE) if a else INTENT_NONE for a in q_age_intent], dtype=np.int8
        )

        # ---- NEW: brand contradiction (BrandMatcher, wired in -- see module docstring) ----
        item_brand = items_df["brand"].fillna("").values
        self.item_brand = item_brand
        # pass the FULL (non-deduplicated) brand column, not pd.unique(item_brand) --
        # BrandMatcher needs real per-brand item counts to filter out one-off
        # seller/private-label tags (see item_meta.BrandMatcher docstring for the
        # real-data measurement that found this: pd.unique() here previously
        # caused 58.5% of real queries to be falsely flagged as naming a brand).
        self.brand_matcher = BrandMatcher(item_brand, min_item_count=brand_min_item_count)
        self.query_brand_sets = [self.brand_matcher.query_brands(t) for t in query_word_tok]
        self.query_has_brand_mention = np.array(
            [1.0 if s else 0.0 for s in self.query_brand_sets], dtype=np.float32
        )

        # ---- NEW: is_generic_query (per unique query, gathered by term_idx later) ----
        self.query_is_generic = np.array(
            [1.0 if is_generic_query(t) else 0.0 for t in query_word_tok], dtype=np.float32
        )

        # ---- misc scalar side features ----
        self.item_title_char_len = items_df["title"].fillna("").str.len().values.astype(np.float32)
        self.item_title_word_len = np.array([len(t) for t in item_word_tok], dtype=np.float32)
        self.item_n_attrs = items_df["attributes"].fillna("").str.count(":").values.astype(np.float32)
        self.query_n_tokens = np.array([len(t) for t in query_word_tok], dtype=np.float32)
        self.query_has_digit = np.array(
            [1.0 if re.search(r"\d", s) else 0.0 for s in query_text], dtype=np.float32
        )

        # ---- optional LLM-derived soft signals (see 02_llm_enrichment.py) ----
        self.has_llm_query = llm_query_enrichment is not None
        if self.has_llm_query:
            q_llm = llm_query_enrichment.set_index("term_id")
            self.llm_query_gender = q_llm["gender"].reindex(terms_df["term_id"]).values
            self.llm_query_age = q_llm["age_group"].reindex(terms_df["term_id"]).values
        self.has_llm_item = llm_item_enrichment is not None
        if self.has_llm_item:
            i_llm = llm_item_enrichment.set_index("item_id")
            self.llm_item_gender = i_llm["gender"].reindex(items_df["item_id"]).values
            self.llm_item_age = i_llm["age_group"].reindex(items_df["item_id"]).values


def _row_gather_dot(A, idx_a, B, idx_b):
    a = A[idx_a].astype(np.float32)
    b = B[idx_b].astype(np.float32)
    return np.einsum("ij,ij->i", a, b)


def _sparse_row_overlap(X, idx_x, Y, idx_y):
    prod = X[idx_x].multiply(Y[idx_y])
    return np.asarray(prod.sum(axis=1)).ravel()


def _brand_contradiction_batch(query_brand_sets, has_brand_mention, term_idx, item_brand, item_idx):
    """Masked partial loop: the vast majority of queries don't name a specific
    brand (has_brand_mention == 0), so only the small subset of rows where the
    query DOES name a brand needs a Python-level substring check -- not all
    3.36M rows. See BrandMatcher.check_brand_contradiction for the match rule."""
    n = len(term_idx)
    out = np.zeros(n, dtype=np.float32)
    mask = has_brand_mention[term_idx] > 0
    idxs = np.nonzero(mask)[0]
    for i in idxs:
        brands = query_brand_sets[term_idx[i]]
        ib = item_brand[item_idx[i]]
        if not ib:
            continue
        ib_norm = normalize(ib).replace(" ", "")
        contradicts = True
        for b in brands:
            b_clean = b.replace(" ", "")
            if b_clean in ib_norm or ib_norm in b_clean:
                contradicts = False
                break
        out[i] = 1.0 if contradicts else 0.0
    return out


AGE_COMPAT_UNKNOWN = {0}


def _age_contradiction_vec(q_code, item_code):
    q = q_code.astype(np.int16)
    it = item_code.astype(np.int16)
    no_intent = q == INTENT_NONE
    item_unknown = it == AGE_CODES["unknown"]
    item_bebek_cocuk = it == AGE_CODES["bebek_cocuk"]
    compat_bebek_cocuk = (q == AGE_CODES["bebek"]) | (q == AGE_CODES["cocuk"])
    mismatch = (q != it) & ~item_bebek_cocuk
    mismatch = mismatch | (item_bebek_cocuk & ~compat_bebek_cocuk)
    out = np.where(no_intent | item_unknown, 0.0, mismatch.astype(np.float32))
    return out


def _gender_contradiction_vec(q_code, item_code):
    q = q_code.astype(np.int16)
    it = item_code.astype(np.int16)
    no_intent = q == INTENT_NONE
    item_open = (it == GENDER_CODES["unknown"]) | (it == GENDER_CODES["unisex"])
    q_unisex = q == GENDER_CODES["unisex"]
    mismatch = (q != it).astype(np.float32)
    out = np.where(no_intent | item_open | q_unisex, 0.0, mismatch)
    return out


def compute_batch_features(term_idx, item_idx, lex: LexicalIndex, emb, pop, label=None):
    n = len(term_idx)
    f = {}

    # ---- dense semantic similarity ----
    f["sim_title"] = _row_gather_dot(emb["query_main"], term_idx, emb["item_title"], item_idx)
    cat_rows = emb["item_category_idx"][item_idx]
    f["sim_category"] = _row_gather_dot(emb["query_main"], term_idx, emb["category_emb"], cat_rows)
    f["sim_attr"] = _row_gather_dot(emb["query_tiny"], term_idx, emb["item_attr"], item_idx)
    f["sim_title_minus_cat"] = f["sim_title"] - f["sim_category"]
    f["sim_max_title_cat"] = np.maximum(f["sim_title"], f["sim_category"])
    f["sim_mean_title_cat"] = (f["sim_title"] + f["sim_category"]) / 2.0

    # ---- lexical overlap (word / stem / char n-gram / expanded) ----
    word_overlap = _sparse_row_overlap(lex.query_word_X, term_idx, lex.item_word_X, item_idx)
    expanded_overlap = _sparse_row_overlap(lex.query_word_expanded_X, term_idx, lex.item_word_X, item_idx)
    stem_overlap = _sparse_row_overlap(lex.query_stem_X, term_idx, lex.item_stem_X, item_idx)
    char_overlap = _sparse_row_overlap(lex.query_char_X, term_idx, lex.item_char_X, item_idx)
    q_wc = lex.query_word_count[term_idx]
    it_wc = lex.item_word_count[item_idx]
    q_cc = lex.query_char_count[term_idx]
    it_cc = lex.item_char_count[item_idx]

    eps = 1e-6
    f["word_overlap_n"] = word_overlap
    f["word_jaccard"] = word_overlap / (q_wc + it_wc - word_overlap + eps)
    f["word_recall"] = word_overlap / (q_wc + eps)
    f["expanded_overlap_n"] = expanded_overlap
    f["expanded_recall"] = expanded_overlap / (q_wc + eps)
    f["expanded_minus_word_overlap"] = expanded_overlap - word_overlap  # >0 means synonym/bigram-merge added signal
    f["stem_overlap_n"] = stem_overlap
    f["stem_recall"] = stem_overlap / (q_wc + eps)

    # ---- NEW: Turkish morphological root overlap (optional, see LexicalIndex.__init__) ----
    if getattr(lex, "has_turkish_morphology", False):
        root_overlap = _sparse_row_overlap(lex.query_root_X, term_idx, lex.item_root_X, item_idx)
        f["root_overlap_n"] = root_overlap
        f["root_recall"] = root_overlap / (q_wc + eps)

    f["char_jaccard"] = char_overlap / (q_cc + it_cc - char_overlap + eps)
    f["char_recall"] = char_overlap / (q_cc + eps)
    f["has_any_word_overlap"] = (word_overlap > 0).astype(np.float32)

    # ---- NEW: category-path literal overlap (any level + deepest level) ----
    cat_overlap = _sparse_row_overlap(lex.query_cat_X, term_idx, lex.item_cat_X, item_idx)
    cat_expanded_overlap = _sparse_row_overlap(lex.query_cat_expanded_X, term_idx, lex.item_cat_X, item_idx)
    cat_last_overlap = _sparse_row_overlap(lex.query_cat_last_X, term_idx, lex.item_cat_last_X, item_idx)
    it_cat_wc = lex.item_cat_count[item_idx]
    it_cat_last_wc = lex.item_cat_last_count[item_idx]
    f["cat_word_overlap_n"] = cat_overlap
    f["cat_word_recall"] = cat_overlap / (q_wc + eps)
    f["cat_expanded_overlap_n"] = cat_expanded_overlap
    f["cat_last_level_overlap_n"] = cat_last_overlap
    f["cat_last_level_recall"] = cat_last_overlap / (q_wc + eps)
    f["cat_last_level_jaccard"] = cat_last_overlap / (q_wc + it_cat_last_wc - cat_last_overlap + eps)
    f["has_any_cat_overlap"] = (cat_overlap > 0).astype(np.float32)

    # ---- constraint matching (color/material with synonym expansion, pattern) ----
    color_ov = _sparse_row_overlap(lex.query_color_X, term_idx, lex.item_color_X, item_idx)
    color_ov_expanded = _sparse_row_overlap(lex.query_color_expanded_X, term_idx, lex.item_color_X, item_idx)
    material_ov = _sparse_row_overlap(lex.query_material_X, term_idx, lex.item_material_X, item_idx)
    material_ov_expanded = _sparse_row_overlap(lex.query_material_expanded_X, term_idx, lex.item_material_X, item_idx)
    pattern_ov = _sparse_row_overlap(lex.query_pattern_X, term_idx, lex.item_pattern_X, item_idx)
    q_has_color = np.asarray(lex.query_color_X[term_idx].sum(axis=1)).ravel() > 0
    q_has_material = np.asarray(lex.query_material_X[term_idx].sum(axis=1)).ravel() > 0
    q_has_pattern = np.asarray(lex.query_pattern_X[term_idx].sum(axis=1)).ravel() > 0
    f["color_match"] = (color_ov > 0).astype(np.float32)
    f["color_match_expanded"] = (color_ov_expanded > 0).astype(np.float32)
    f["color_conflict"] = (q_has_color & (color_ov_expanded == 0)).astype(np.float32)
    f["material_match"] = (material_ov > 0).astype(np.float32)
    f["material_match_expanded"] = (material_ov_expanded > 0).astype(np.float32)
    f["material_conflict"] = (q_has_material & (material_ov_expanded == 0)).astype(np.float32)
    f["pattern_match"] = (pattern_ov > 0).astype(np.float32)
    f["pattern_conflict"] = (q_has_pattern & (pattern_ov == 0)).astype(np.float32)

    # ---- NEW: size/beden match & conflict (analogous to color/pattern above) ----
    size_ov = _sparse_row_overlap(lex.query_size_X, term_idx, lex.item_size_X, item_idx)
    q_has_size = np.asarray(lex.query_size_X[term_idx].sum(axis=1)).ravel() > 0
    f["size_match"] = (size_ov > 0).astype(np.float32)
    f["size_conflict"] = (q_has_size & (size_ov == 0)).astype(np.float32)

    f["gender_contradiction"] = _gender_contradiction_vec(lex.query_gender_code[term_idx], lex.item_gender_code[item_idx])
    f["age_contradiction"] = _age_contradiction_vec(lex.query_age_code[term_idx], lex.item_age_code[item_idx])
    f["item_gender_meta_title_conflict"] = lex.item_gender_conflict[item_idx]
    f["query_has_gender_word"] = (lex.query_gender_code[term_idx] != INTENT_NONE).astype(np.float32)
    f["query_has_age_word"] = (lex.query_age_code[term_idx] != INTENT_NONE).astype(np.float32)

    # ---- NEW: brand contradiction (masked partial loop, see helper docstring) ----
    f["brand_contradiction"] = _brand_contradiction_batch(
        lex.query_brand_sets, lex.query_has_brand_mention, term_idx, lex.item_brand, item_idx
    )
    f["query_has_brand_mention"] = lex.query_has_brand_mention[term_idx]

    # ---- NEW: generic/ambiguous query flag ----
    f["is_generic_query"] = lex.query_is_generic[term_idx]

    # ---- side / popularity features ----
    f["query_n_tokens"] = lex.query_n_tokens[term_idx]
    f["query_has_digit"] = lex.query_has_digit[term_idx]
    f["item_title_char_len"] = lex.item_title_char_len[item_idx]
    f["item_title_word_len"] = lex.item_title_word_len[item_idx]
    f["item_n_attrs"] = lex.item_n_attrs[item_idx]
    f["len_ratio"] = lex.query_n_tokens[term_idx] / (lex.item_title_word_len[item_idx] + eps)

    loo = label.astype(np.float32) if label is not None else 0.0
    item_click_count = np.clip(pop["item_click_count"][item_idx].astype(np.float32) - loo, 0, None)
    brand_click_count = np.clip(pop["brand_click_count"][item_idx].astype(np.float32) - loo, 0, None)
    f["item_click_log"] = np.log1p(item_click_count)
    f["item_click_cat_rel"] = f["item_click_log"] - pop["item_cat_mean_log"][item_idx]
    f["brand_click_log"] = np.log1p(brand_click_count)

    # ---- optional LLM-derived soft signals (additive, never overrides the
    # rule-based hard gender_contradiction/age_contradiction above) ----
    if lex.has_llm_query and lex.has_llm_item:
        llm_q_gender = lex.llm_query_gender[term_idx]
        llm_i_gender = lex.llm_item_gender[item_idx]
        llm_q_age = lex.llm_query_age[term_idx]
        llm_i_age = lex.llm_item_age[item_idx]
        has_both_gender = pd.notna(llm_q_gender) & pd.notna(llm_i_gender)
        has_both_age = pd.notna(llm_q_age) & pd.notna(llm_i_age)
        f["llm_gender_conflict"] = np.where(
            has_both_gender, (llm_q_gender != llm_i_gender).astype(np.float32), 0.0
        )
        f["llm_age_conflict"] = np.where(
            has_both_age, (llm_q_age != llm_i_age).astype(np.float32), 0.0
        )

    return pd.DataFrame(f)
