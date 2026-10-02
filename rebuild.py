"""Rebuild analysis tables from the committed corpus snapshot (no pipeline needed).

Regenerates, using ONLY the installed nepali_readability library plus the stdlib:
  data/processed/features_v1_ngt_akshara.csv   (frozen combo: nepali_grammar + akshara)
  data/processed/features_v2_ngt_akshara_hun.csv (+ baseline/Hunspell columns)
  data/processed/features_v3_ngt_akshara_hun.csv (+ n_poly3, n_nospace)
  data/processed/scores_{method}_v1.csv + stats_{method}_v1.csv (all 10 methods)

Reads: data/corpus/nepali-textbook-sentences.csv.gz, data/external/baseline_words_v1.txt,
  data/external/ne_NP.{dic,aff}. All method params equal the library defaults
  (verified: every configs/method_*.yaml params dict matches code defaults), so no
  config files are needed. Deterministic. Expect ~15-30 min (Hunspell lookups).

  Known deviation: chapter_title is written empty (per-grade interim source files
  are local-only and not committed; verified no live consumer reads the column —
  the v2 split groups by book, not chapter). stable_id (content hash, see
  run_merge.stable_id_of) IS populated here identically. sentence_index is
  recovered exactly from the id suffix (A-..-{si} / B-..-{si}), verified 100%
  against the canonical table.

Does NOT rebuild: the frozen split (committed), model artifacts (need scikit-learn,
  a research dependency — see docs), or the manuscript figures. Run from repo root:
  python rebuild.py
"""

import csv
import gzip
import hashlib
import pathlib
import re
import statistics
import sys
import unicodedata

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from nepali_readability import formulas  # noqa: E402
from nepali_readability import halant as H  # noqa: E402
from nepali_readability import script as S  # noqa: E402
from nepali_readability.baseline_words import (  # noqa: E402
    AffixStripper, BaselineWords, apply_mode,
)

P = ROOT / "data" / "processed"

V1_FIELDS = ["id", "grade", "subject", "source_corpus", "chapter_title",
             "sentence_index", "stable_id", "sentence_text", "char_count", "word_count",
             "sentence_length_words", "syllables", "syllables_per_word",
             "matras", "matras_long", "matras_short", "viramas", "conjuncts",
             "cons_raw", "cons_surface", "cons_merged", "vowels_total",
             "ratio_vc_raw", "ratio_vc_surface", "anusvara", "visarga",
             "independent_vowels", "dev_ratio"]
V2_NEW = ["bl_cov", "bl_oob_n", "bl_oob", "dict_cov", "morph_n_surface",
          "morph_n_stems", "morph_n_stemsuf", "morph_stems_oob_n"]
V3_NEW = ["n_poly3", "n_nospace"]


def words_of(text):
    cleaned = S.detach_punct(S.strip_residue(text))
    return cleaned, [w for w in S.segment_words(cleaned, method="nepali_grammar")
                     if w.strip()]


def featurize_v1(text):
    cleaned, words = words_of(text)
    spw = S.count_syllables_per_word(words, method="akshara")
    counts = S.count_codepoints(cleaned)
    hf = H.halant_features(cleaned)
    nw = len(words)
    return {
        "char_count": len(text), "word_count": nw, "sentence_length_words": nw,
        "syllables": sum(spw),
        "syllables_per_word": round(sum(spw) / nw, 4) if nw else 0.0,
        "matras": counts["matras"], "matras_long": counts["matras_long"],
        "matras_short": counts["matras_short"], "viramas": counts["viramas"],
        "conjuncts": counts["conjuncts"], "cons_raw": hf["cons_raw"],
        "cons_surface": hf["cons_surface"], "cons_merged": hf["cons_merged"],
        "vowels_total": hf["vowels_total"], "ratio_vc_raw": hf["ratio_vc_raw"],
        "ratio_vc_surface": hf["ratio_vc_surface"],
        "anusvara": counts["anusvara"], "visarga": counts["visarga"],
        "independent_vowels": counts["independent_vowels"],
        "dev_ratio": round(S.devanagari_ratio(text), 4),
    }


def main():
    print("reading corpus snapshot...", flush=True)
    with gzip.open(ROOT / "data" / "corpus" / "nepali-textbook-sentences.csv.gz",
                   "rt", encoding="utf-8") as f:
        sents = list(csv.DictReader(f))
    print(f"snapshot rows: {len(sents)}", flush=True)

    print("stage v1 (tokenize + akshara + halant)...", flush=True)
    rows = []
    for s in sents:
        g = int(s["grade"])
        # Self-contained copy of run_merge.stable_id_of (rebuild cannot import
        # experiments/, absent on fresh clones): sha1 of NFC + whitespace-collapse.
        canon = re.sub(r"\s+", " ", unicodedata.normalize("NFC", s["sentence_text"])).strip()
        sid = hashlib.sha1(canon.encode("utf-8")).hexdigest()[:16]
        r = {"id": s["id"], "grade": g, "subject": s["subject"],
             "source_corpus": s["source_corpus"], "chapter_title": "",
             "sentence_index": int(s["id"].rsplit("-", 1)[1]),
             "stable_id": sid, "sentence_text": s["sentence_text"]}
        r.update(featurize_v1(s["sentence_text"]))
        rows.append(r)
    with open(P / "features_v1_ngt_akshara.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=V1_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in V1_FIELDS})

    print("stage v2 (baseline + Hunspell)...", flush=True)
    from spylls.hunspell import Dictionary
    bw = BaselineWords()
    st = AffixStripper()
    dic = Dictionary.from_files(str(ROOT / "data" / "external" / "ne_NP"))
    stem_cache, dict_cache = {}, {}

    def stemmed(w):
        v = stem_cache.get(w)
        if v is None:
            v = st.stem(w)
            stem_cache[w] = v
        return v

    def in_dict(w):
        v = dict_cache.get(w)
        if v is None:
            try:
                v = bool(dic.lookup(w))
            except Exception:
                v = False
            dict_cache[w] = v
        return v

    for r in rows:
        _, words = words_of(r["sentence_text"])
        triples = [(x,) + stemmed(x) for x in words]
        stems = apply_mode(words, triples, "stems-only")
        stemsuf = apply_mode(words, triples, "stems-plus-suffixes")
        cov, oob = bw.coverage(words)
        letters = [x for x in words
                   if any("\u0904" <= c <= "\u0939" or "\u0958" <= c <= "\u0960"
                          for c in x)]
        scov, _ = bw.coverage(stems)
        r.update({
            "bl_cov": cov, "bl_oob_n": len(oob), "bl_oob": "|".join(oob[:20]),
            "dict_cov": round(sum(1 for x in letters if in_dict(x))
                              / len(letters), 4) if letters else 1.0,
            "morph_n_surface": len(words), "morph_n_stems": len(stems),
            "morph_n_stemsuf": len(stemsuf),
            "morph_stems_oob_n": len(stems) - round(scov * len(stems))})
    with open(P / "features_v2_ngt_akshara_hun.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=V1_FIELDS + V2_NEW)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in V1_FIELDS + V2_NEW})

    print("stage v3 (n_poly3, n_nospace)...", flush=True)
    for r in rows:
        _, words = words_of(r["sentence_text"])
        spw = S.count_syllables_per_word(words, method="akshara")
        cleaned = S.detach_punct(S.strip_residue(r["sentence_text"]))
        r["n_poly3"] = sum(1 for s in spw if s >= 3)
        r["n_nospace"] = len(cleaned.replace(" ", ""))
    with open(P / "features_v3_ngt_akshara_hun.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=V1_FIELDS + V2_NEW + V3_NEW)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in V1_FIELDS + V2_NEW + V3_NEW})

    print("stage scores+stats (10 methods)...", flush=True)
    str_rows = [{k: ("" if v is None else str(v)) for k, v in r.items()}
                for r in rows]
    for m in formulas.METHODS:
        fn = formulas.DISPATCH[m]
        per_grade = {}
        with open(P / f"scores_{m}_v1.csv", "w", newline="",
                  encoding="utf-8") as fo:
            w = csv.writer(fo)
            w.writerow(["id", "grade", "raw", "ease"])
            for r, sr in zip(rows, str_rows):
                out = fn(sr, {})
                g = int(r["grade"])
                w.writerow([r["id"], g, round(out["raw"], 4),
                            round(out["ease"], 2)])
                per_grade.setdefault(g, []).append(out["ease"])
        with open(P / f"stats_{m}_v1.csv", "w", newline="",
                  encoding="utf-8") as fo:
            w = csv.writer(fo)
            w.writerow(["grade", "n", "mean_ease", "std_ease"])
            for g in sorted(per_grade):
                v = per_grade[g]
                w.writerow([g, len(v), round(sum(v) / len(v), 2),
                            round(statistics.pstdev(v), 2) if len(v) > 1
                            else 0.0])
        print(f"  {m}: {sum(map(len, per_grade.values()))} scored", flush=True)
    print("done. Verify with: python -m pytest tests/ -q", flush=True)


if __name__ == "__main__":
    main()
