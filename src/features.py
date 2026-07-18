"""
Vectorized feature engineering shared by the train-set builder and the 3.36M-row
submission builder.

Design goal ("fast feature engineering... combined with GBDTs", prompt.md section 3):
no per-pair Python loop over the hot path. Lexical/char-ngram overlap is computed
as *row-aligned sparse-matrix elementwise multiplication* (scipy, vectorized C code)
instead of per-row set intersections; embedding similarity is batched numpy dot
products; gender/age contradiction and color/material/pattern overlap are done on
integer-coded arrays / small fixed-vocab sparse indicator matrices. This is what
makes it feasible to featurize 3.36M pairs without a cross-encoder.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from item_meta import (  # noqa: E402
    age_intent_from_tokens,
    extract_query_constraints,
    gender_intent_from_tokens,
    normalize_age,
    normalize_gender,
    parse_item_constraints,
)
from text_utils import char_ngrams, stem, tokenize  # noqa: E402

AGE_CODES = {"unknown": 0, "yetiskin": 1, "cocuk": 2, "bebek": 3, "genc": 4, "bebek_cocuk": 5}
GENDER_CODES = {"unknown": 0, "kadın": 1, "erkek": 2, "unisex": 3}
INTENT_NONE = -1


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
    """sets_list: list[frozenset[str]] -> sparse (n, len(vocab)) binary indicator matrix."""
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
    """Fitted once on the item catalog; holds sparse word/stem/char matrices for
    both items and queries plus the small-vocab constraint matrices."""

    def __init__(self, items_df, terms_df):
        import gc

        item_text = (
            items_df["title"].fillna("") + " " + items_df["brand"].fillna("") + " "
            + items_df["category"].fillna("")
        ).values
        query_text = terms_df["query"].fillna("").values

        # item_word_tok is kept alive (needed later for gender/age title-fallback and
        # title-length features); stem/char token lists are transient (~GB-scale for
        # ~1M items) and freed right after their vectorizer is fit.
        item_word_tok = [tokenize(s) for s in item_text]
        query_word_tok = [tokenize(s) for s in query_text]
        self.word_vec, self.item_word_X = _fit_binary_vectorizer(item_word_tok, min_df=1)
        self.query_word_X = _transform_binary(self.word_vec, query_word_tok)

        item_stem_tok = [[stem(t) for t in toks] for toks in item_word_tok]
        query_stem_tok = [[stem(t) for t in toks] for toks in query_word_tok]
        self.stem_vec, self.item_stem_X = _fit_binary_vectorizer(item_stem_tok, min_df=1)
        self.query_stem_X = _transform_binary(self.stem_vec, query_stem_tok)
        del item_stem_tok, query_stem_tok
        gc.collect()

        item_char_tok = [sorted(char_ngrams(s, n=3)) for s in items_df["title"].fillna("").values]
        query_char_tok = [sorted(char_ngrams(s, n=3)) for s in query_text]
        self.char_vec, self.item_char_X = _fit_binary_vectorizer(item_char_tok, min_df=1)
        self.query_char_X = _transform_binary(self.char_vec, query_char_tok)
        del item_char_tok, query_char_tok
        gc.collect()

        self.item_word_count = np.asarray(self.item_word_X.sum(axis=1)).ravel()
        self.query_word_count = np.asarray(self.query_word_X.sum(axis=1)).ravel()
        self.item_char_count = np.asarray(self.item_char_X.sum(axis=1)).ravel()
        self.query_char_count = np.asarray(self.query_char_X.sum(axis=1)).ravel()

        # ---- fixed-vocab constraint matrices (color/material/pattern) ----
        item_colors, item_materials, item_patterns = parse_item_constraints(items_df["attributes"])
        q_colors, q_materials, q_patterns = [], [], []
        for toks in query_word_tok:
            c, m, p = extract_query_constraints(toks)
            q_colors.append(c)
            q_materials.append(m)
            q_patterns.append(p)

        from item_meta import COLOR_VOCAB, MATERIAL_VOCAB, PATTERN_VOCAB

        self.item_color_X = _small_vocab_matrix(item_colors, COLOR_VOCAB)
        self.query_color_X = _small_vocab_matrix(q_colors, COLOR_VOCAB)
        self.item_material_X = _small_vocab_matrix(item_materials, MATERIAL_VOCAB)
        self.query_material_X = _small_vocab_matrix(q_materials, MATERIAL_VOCAB)
        self.item_pattern_X = _small_vocab_matrix(item_patterns, PATTERN_VOCAB)
        self.query_pattern_X = _small_vocab_matrix(q_patterns, PATTERN_VOCAB)

        # ---- gender / age: metadata + title-derived fallback (noisy-metadata robustness) ----
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

        # ---- misc scalar side features ----
        self.item_title_char_len = items_df["title"].fillna("").str.len().values.astype(np.float32)
        self.item_title_word_len = np.array([len(t) for t in item_word_tok], dtype=np.float32)
        self.item_n_attrs = items_df["attributes"].fillna("").str.count(":").values.astype(np.float32)
        self.query_n_tokens = np.array([len(t) for t in query_word_tok], dtype=np.float32)
        self.query_has_digit = np.array(
            [1.0 if re.search(r"\d", s) else 0.0 for s in query_text], dtype=np.float32
        )


def _row_gather_dot(A, idx_a, B, idx_b):
    """cosine sim for L2-normalized fp16 embeddings, upcast to fp32 per-batch."""
    a = A[idx_a].astype(np.float32)
    b = B[idx_b].astype(np.float32)
    return np.einsum("ij,ij->i", a, b)


def _sparse_row_overlap(X, idx_x, Y, idx_y):
    """count of shared columns between row idx_x of X and row idx_y of Y (both binary)."""
    prod = X[idx_x].multiply(Y[idx_y])
    return np.asarray(prod.sum(axis=1)).ravel()


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
    """
    term_idx, item_idx: int arrays, same length (one row per pair)
    emb: dict with query_main, query_tiny, item_title, item_attr, category_emb, item_category_idx
    pop: dict with item_click_count, item_cat_mean_log, brand_click_count (raw, aligned to item_idx)
    label: optional 0/1 array. When given (i.e. building the TRAIN set), the row's own
        contribution is subtracted from its item's/brand's click count before log-transform
        (leave-one-out). Without this, a positive row's popularity features would count its
        *own* click -- for the ~92% of items clicked exactly once, that makes item_click_log
        an almost perfect label proxy during training that doesn't exist at inference time.
    """
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

    # ---- lexical overlap (word / stem / char n-gram) ----
    word_overlap = _sparse_row_overlap(lex.query_word_X, term_idx, lex.item_word_X, item_idx)
    stem_overlap = _sparse_row_overlap(lex.query_stem_X, term_idx, lex.item_stem_X, item_idx)
    char_overlap = _sparse_row_overlap(lex.query_char_X, term_idx, lex.item_char_X, item_idx)
    q_wc = lex.query_word_count[term_idx]
    it_wc = lex.item_word_count[item_idx]
    q_cc = lex.query_char_count[term_idx]
    it_cc = lex.item_char_count[item_idx]

    eps = 1e-6
    f["word_overlap_n"] = word_overlap
    f["word_jaccard"] = word_overlap / (q_wc + it_wc - word_overlap + eps)
    f["word_recall"] = word_overlap / (q_wc + eps)  # fraction of query covered by item text
    f["stem_overlap_n"] = stem_overlap
    f["stem_recall"] = stem_overlap / (q_wc + eps)
    f["char_jaccard"] = char_overlap / (q_cc + it_cc - char_overlap + eps)
    f["char_recall"] = char_overlap / (q_cc + eps)
    f["has_any_word_overlap"] = (word_overlap > 0).astype(np.float32)

    # ---- constraint matching ----
    color_ov = _sparse_row_overlap(lex.query_color_X, term_idx, lex.item_color_X, item_idx)
    material_ov = _sparse_row_overlap(lex.query_material_X, term_idx, lex.item_material_X, item_idx)
    pattern_ov = _sparse_row_overlap(lex.query_pattern_X, term_idx, lex.item_pattern_X, item_idx)
    q_has_color = np.asarray(lex.query_color_X[term_idx].sum(axis=1)).ravel() > 0
    q_has_material = np.asarray(lex.query_material_X[term_idx].sum(axis=1)).ravel() > 0
    q_has_pattern = np.asarray(lex.query_pattern_X[term_idx].sum(axis=1)).ravel() > 0
    f["color_match"] = (color_ov > 0).astype(np.float32)
    f["color_conflict"] = (q_has_color & (color_ov == 0)).astype(np.float32)
    f["material_match"] = (material_ov > 0).astype(np.float32)
    f["material_conflict"] = (q_has_material & (material_ov == 0)).astype(np.float32)
    f["pattern_match"] = (pattern_ov > 0).astype(np.float32)
    f["pattern_conflict"] = (q_has_pattern & (pattern_ov == 0)).astype(np.float32)

    f["gender_contradiction"] = _gender_contradiction_vec(lex.query_gender_code[term_idx], lex.item_gender_code[item_idx])
    f["age_contradiction"] = _age_contradiction_vec(lex.query_age_code[term_idx], lex.item_age_code[item_idx])
    f["item_gender_meta_title_conflict"] = lex.item_gender_conflict[item_idx]
    f["query_has_gender_word"] = (lex.query_gender_code[term_idx] != INTENT_NONE).astype(np.float32)
    f["query_has_age_word"] = (lex.query_age_code[term_idx] != INTENT_NONE).astype(np.float32)

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

    return pd.DataFrame(f)
