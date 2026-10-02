"""ReadabilityScorer: .score(text), .compare(texts), .batch_score(texts).

Same public shape as hindi_readability.ReadabilityScorer. Single-text scoring computes
live features with the frozen §4.4 combo (nepali_grammar + akshara) plus the v2
baseline/Hunspell columns; bulk work reads data/processed CSVs instead (§7).
"""
from nepali_readability import formulas, script as S
from nepali_readability.baseline_words import AffixStripper, BaselineWords, apply_mode

_ngl_bands = formulas.ngl_from_nrs


class ReadabilityScorer:
    def __init__(self, config=None):
        self.config = config or {}
        self._bw = None
        self._st = None
        self._dic = None

    def _lazy(self):
        if self._bw is None:
            self._bw = BaselineWords(
                self.config.get("baseline_path"))
            self._st = AffixStripper(self.config.get("aff_path"))
            try:
                from spylls.hunspell import Dictionary
                self._dic = Dictionary.from_files(
                    self.config.get("dic_path") or
                    str(__import__("pathlib").Path(__file__).resolve().parents[1]
                        / "data" / "external" / "ne_NP"))
            except Exception:
                self._dic = None

    def _featurize_text(self, text):
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")
        self._lazy()
        cleaned = S.detach_punct(S.strip_residue(text))
        words = [w for w in S.segment_words(cleaned, method="nepali_grammar") if w.strip()]
        spw = S.count_syllables_per_word(words, method="akshara")
        counts = S.count_codepoints(cleaned)
        from nepali_readability import halant as H
        hf = H.halant_features(cleaned)
        triples = [(w,) + self._st.stem(w) for w in words]
        stems = apply_mode(words, triples, "stems-only")
        stemsuf = apply_mode(words, triples, "stems-plus-suffixes")
        cov, _ = self._bw.coverage(words)
        nw = len(words)
        n_nospace = len(cleaned.replace(" ", ""))
        n_poly3 = sum(1 for s in spw if s >= 3)
        feats = {
            "sentence_length_words": nw, "word_count": nw,
            "syllables_per_word": sum(spw) / nw if nw else 0.0,
            "conjuncts": counts["conjuncts"],
            "matras": counts["matras"], "matras_long": counts["matras_long"],
            "vowels_total": hf["vowels_total"], "cons_surface": hf["cons_surface"],
            "bl_cov": cov, "n_poly3": n_poly3, "n_nospace": n_nospace,
            "morph_n_surface": nw, "morph_n_stems": len(stems),
            "morph_n_stemsuf": len(stemsuf),
        }
        return feats, words

    def score(self, text):
        feats, _ = self._featurize_text(text)
        cfgs = self.config.get("methods", {})
        out = formulas.score_all(feats, cfgs)
        nrs = out["nrs"]["ease"]
        out["ngl"] = _ngl_bands(nrs, cfgs.get("ngl", {}))
        return out

    def compare(self, texts):
        scored = [(t, self.score(t)) for t in texts]
        scored.sort(key=lambda kv: kv[1]["nrs"]["ease"], reverse=True)
        return [{"text": t[:60], **s} for t, s in scored]

    def batch_score(self, texts):
        return self.compare(texts)
