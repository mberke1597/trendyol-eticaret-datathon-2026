"""
Stage 60 — attribute-contradiction post-filter (high precision, 1->0 only).

Motivation (from a manual audit of the 0.891 submission): the lexical+embedding
models assign HIGH surface similarity to query/item pairs that share words and
category but violate exactly one hard constraint, and mark them relevant. A
careful reader catches these instantly. This filter encodes those catches as
rules and flips ONLY clear contradictions from relevant(1) -> not-relevant(0).

It never creates positives (no 1<-0), so it can only raise precision — it can't
blow up recall the way the Qwen/CE hard-overrides did. Verified on 891: flips
~2,431 / 924,980 positives (0.26%), every sampled flip a genuine error, e.g.:
  ayt edebiyat        -> "ayt tarih soru bankası"        (book subject)
  195/55 r16          -> "205/55r16 ..."                 (tire size)
  kışlık motor eldiven-> "... yazlık eldiven ..."         (season)
  erkek uzun kollu    -> "... kısa kollu ..."             (sleeve length)
  erkek çorap         -> gender=kadın                     (gender)

Rules are deliberately conservative: fire only on EXPLICIT opposite tokens, so a
missing attribute never triggers a flip (e.g. a winter tire whose title omits
"kış" is left untouched).

TRAIN-VALIDATED (2026-07-14). Each rule was measured on the 250k RELEVANT
(label=1) training pairs: what fraction of genuinely-relevant pairs have the
attribute MISMATCHING? That is the false-flip rate. Principle: keep a rule only
if that rate is < ~5%.
  SAFE (kept):    cinsiyet 0.2% (n=35342) · kol_boyu 0.0% · sezon 0.7% ·
                  lastik ~0% · kitap 0.0% · iphone 3.0% · galaxy 0.0% ·
                  numara 0.0% · kapasite_gb 0.0% · kişilik 1.2% (n=425) ·
                  watt 0.0% · hacim_ml 0.0%
  BORDERLINE:     sinif 7.5% (multi-grade books leak) — kept but weakest.
  REJECTED (unsafe, would delete true positives):
                  RENK 21.4% · ekran_inç 29.0% · litre 7.8% · adet 6.5% ·
                  cm_ölçü 6.7% · (materyal — variant-level, same failure mode).
  Why: intrinsic specs (subject/model/size/gender/capacity) are tight in
  relevant pairs; VARIANT descriptors (colour/material/screen-size) are loose —
  an item page spans many variants, so a relevant item often "mismatches" them.
  This is exactly why the colour/material/type submissions LOST on the LB.

Usage:
  python 60_attribute_contradiction_filter.py --in <submission.csv> --out <filtered.csv>
  # restrict with e.g. --rules kol_boyu,sezon,lastik,kitap,cinsiyet,kisilik
"""
import argparse, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from config import DATA_DIR

BOOK_SUBJECTS = ["fizik", "kimya", "biyoloji", "matematik", "geometri", "tarih",
                 "coğrafya", "edebiyat", "türkçe", "ingilizce", "almanca", "felsefe"]
TIRE_RE = re.compile(r"(\d{3})[\s/](\d{2})\s*r\s?(\d{2})")


def _has(arr, pat):
    rx = re.compile(pat)
    return np.fromiter((bool(rx.search(x)) for x in arr), bool, len(arr))


def _tire(arr):
    return [(m.groups() if m else None) for m in (TIRE_RE.search(x) for x in arr)]


def _subjects(arr):
    return [set(s for s in BOOK_SUBJECTS if s in x) for x in arr]


def _one(arr, pat):
    rx = re.compile(pat)
    return [(m.group(1) if m else None) for m in (rx.search(x) for x in arr)]


def _allset(arr, pat):
    rx = re.compile(pat)
    return [set(rx.findall(x)) for x in arr]


def _numeric_contradiction(qvals, tsets):
    """Query specifies a numeric spec that is NOT among the item's values.
    Robust to multi-value items (e.g. a book covering '3.sınıf, 4.sınıf')."""
    return np.fromiter((v is not None and len(s) > 0 and v not in s
                        for v, s in zip(qvals, tsets)), bool, len(qvals))


_CAP_WORDS = {"tek": "1", "tekli": "1", "çift": "2", "ikili": "2",
              "üçlü": "3", "dörtlü": "4"}


def _capacity_set(arr):
    """All capacity values found: '\\d kişilik' + word forms (tek/çift/...)."""
    out = []
    for s in arr:
        caps = set(re.findall(r"(\d)\s?kişilik", s))
        for w, nn in _CAP_WORDS.items():
            if re.search(r"\b" + w + r"\b", s):
                caps.add(nn)
        out.append(caps)
    return out


def _capacity(arr):
    """Same as _capacity_set (query side); kept separate for readability."""
    return _capacity_set(arr)


def contradiction_mask(query, title, category, gender, rules):
    """Boolean mask over rows: True == hard query/item contradiction (flip 1->0).
    All inputs are lowercase string arrays of equal length."""
    n = len(query)
    flip = np.zeros(n, bool)
    reason = np.empty(n, object); reason[:] = ""

    def add(m, name):
        nonlocal flip
        nw = m & ~flip
        reason[nw] = name; flip[nw] = True

    if "kol_boyu" in rules:      # sleeve length
        add((_has(query, r"uzun kol") & _has(title, r"k[ıi]sa kol|kolsuz")) |
            (_has(query, r"k[ıi]sa kol") & _has(title, r"uzun kol")), "kol_boyu")

    if "sezon" in rules:         # season (explicit opposite words only)
        qk = _has(query, r"kış|kışlık"); qy = _has(query, r"\byaz\b|yazlık|yazlik")
        tk = _has(title, r"kış|kışlık"); ty = _has(title, r"\byaz\b|yazlık|yazlik")
        add((qk & ty) | (qy & tk), "sezon")

    if "lastik" in rules:        # tire size mismatch
        qt = _tire(query); it = _tire(np.char.add(np.char.add(title, " "), category))
        add(np.fromiter(((a and b and a != b) for a, b in zip(qt, it)), bool, n), "lastik_ebadi")

    if "kitap" in rules:         # book subject mismatch
        qs = _subjects(query); ts = _subjects(title); bk = _has(category, r"kitap")
        add(np.fromiter((len(a) > 0 and len(b) > 0 and a.isdisjoint(b)
                         for a, b in zip(qs, ts)), bool, n) & bk, "kitap_konusu")

    if "cinsiyet" in rules:      # gender field contradiction
        qe = _has(query, r"erkek") & ~_has(query, r"kad[ıi]n|unisex")
        qd = _has(query, r"kad[ıi]n") & ~_has(query, r"erkek|unisex")
        add((qe & (gender == "kadın")) | (qd & (gender == "erkek")), "cinsiyet")

    if "sinif" in rules:         # school grade mismatch (books); multi-grade safe
        add(_numeric_contradiction(_one(query, r"(\d+)\.?\s?sınıf"),
                                   _allset(title, r"(\d+)\.?\s?sınıf")) & _has(category, r"kitap"),
            "sinif")

    if "kapasite" in rules:      # storage capacity mismatch (GB)
        add(_numeric_contradiction(_one(query, r"(\d+)\s?gb"),
                                   _allset(title, r"(\d+)\s?gb")), "kapasite_gb")

    if "telefon" in rules:       # iPhone model number mismatch
        add(_numeric_contradiction(_one(query, r"iphone\s?(\d+)"),
                                   _allset(title, r"iphone\s?(\d+)")), "iphone_modeli")

    if "galaxy" in rules:        # Samsung Galaxy S / A model mismatch
        add(_numeric_contradiction(_one(query, r"galaxy\s?s\s?(\d+)"),
                                   _allset(title, r"galaxy\s?s\s?(\d+)")), "galaxy_s")
        add(_numeric_contradiction(_one(query, r"galaxy\s?a\s?(\d+)"),
                                   _allset(title, r"galaxy\s?a\s?(\d+)")), "galaxy_a")

    if "numara" in rules:        # size number mismatch (e.g. diaper size, shoe size)
        add(_numeric_contradiction(_one(query, r"(\d+)\s?numara"),
                                   _allset(title, r"(\d+)\s?numara")), "numara")

    if "hacim" in rules:         # volume mismatch (ml)  [train mismatch 0.0%]
        add(_numeric_contradiction(_one(query, r"(\d+)\s?ml"),
                                   _allset(title, r"(\d+)\s?ml")), "hacim_ml")

    if "kisilik" in rules:       # seating/bedding capacity  [train mismatch 1.2%, n=425]
        # "çift kişilik" vs "tek kişilik" nevresim/yatak, "3 kişilik" vs "2 kişilik" koltuk
        it = np.char.add(np.char.add(title, " "), category)
        qcap = _capacity(query); tcap = _capacity_set(it)
        add(np.fromiter((len(a) == 1 and len(b) > 0 and a.isdisjoint(b)
                         for a, b in zip(qcap, tcap)), bool, n), "kisilik")

    if "watt" in rules:          # appliance wattage mismatch  [train mismatch 0.0%]
        # guard: skip motor-oil viscosity like "10w-40"
        oil = _has(query, r"\d+w-\d")
        add(_numeric_contradiction(_one(query, r"(\d+)\s?w(?:att)?\b"),
                                   _allset(title, r"(\d+)\s?w(?:att)?\b")) & ~oil, "watt")

    return flip, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="submission csv (id,prediction)")
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--rules",  # all train-validated <5% (sinif 7.5% borderline kept)
                    default="kol_boyu,sezon,lastik,kitap,cinsiyet,sinif,kapasite,telefon,galaxy,numara,hacim,kisilik,watt")
    args = ap.parse_args()
    rules = set(args.rules.split(","))

    sub = pd.read_csv(args.inp)
    items = pd.read_csv(f"{DATA_DIR}/items.csv", usecols=["item_id", "title", "category", "gender"])
    terms = pd.read_csv(f"{DATA_DIR}/terms.csv")
    pairs = pd.read_csv(f"{DATA_DIR}/submission_pairs.csv")

    pos_ids = sub.loc[sub.prediction == 1, "id"]
    pos = pairs.merge(pos_ids.to_frame(), on="id").merge(terms, on="term_id").merge(items, on="item_id")
    for c in ["query", "title", "category", "gender"]:
        pos[c] = pos[c].fillna("").str.lower()

    flip, reason = contradiction_mask(pos["query"].values, pos["title"].values,
                                      pos["category"].values, pos["gender"].values, rules)
    flagged = set(pos.loc[flip, "id"])
    print(f"[60] positives={len(pos):,}  contradictions flipped 1->0={len(flagged):,} "
          f"({100*len(flagged)/max(len(pos),1):.2f}%)")
    vc = pd.Series(reason[flip]).value_counts()
    print(vc.to_string())

    m = sub["id"].isin(flagged)
    sub.loc[m, "prediction"] = 0
    sub.to_csv(args.out, index=False)
    print(f"[60] wrote {args.out}  (pos {int((sub.prediction==1).sum()):,})")


if __name__ == "__main__":
    main()
