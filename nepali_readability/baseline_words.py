"""Baseline/easy-word list + Hunspell morphology (§4.3) — two distinct uses.

Use 1 — baseline coverage: BaselineWords loads data/external/baseline_words_v1.txt
(Grade 1–3 frequency >= 10, Dale-Chall analog). coverage(words) -> (fraction, oob_list).
Use 2 — morphology: Morphology strips affixes using the MAINTAINED ne_NP .aff SFX/PFX
rules (parsed, not hand-written). One suffix pass then one prefix pass; longest-append
match wins; NO dictionary gating (roots like गर- are allomorphs absent from the .dic,
so gating would veto correct strips — documented simplification vs full Hunspell
generate-and-check). spylls lookup is used only for the dict_cov diagnostic column.
Modes (§4.3 toggles in configs/method_hunspell_morphology.yaml): surface-as-is |
stems-only | stems-plus-suffixes.
"""
import pathlib
import re

LETTER = re.compile(r"[\u0904-\u0939\u0958-\u0960]")
ROOT = pathlib.Path(__file__).resolve().parents[1]


class AffixStripper:
    """Suffix/prefix stripper driven by a parsed Hunspell .aff file."""

    def __init__(self, aff_path=None):
        self.sfx = []  # (append, strip, cond_re_or_literal, is_regex)
        self.pfx = []
        with open(aff_path or ROOT / "data" / "external" / "ne_NP.aff",
                   encoding="utf-8") as f:
            for ln in f:
                parts = ln.split()
                # Hunspell: SFX <flag> <strip> <append>[/x-flags] <condition>
                if len(parts) < 5 or parts[0] not in ("SFX", "PFX"):
                    continue
                if parts[2] in ("Y", "N"):  # header line, not a rule
                    continue
                kind, strip = parts[0], parts[2]
                append, cond = parts[3].split("/")[0], parts[4]
                rule = (append, "" if strip == "0" else strip, cond)
                (self.sfx if kind == "SFX" else self.pfx).append(rule)
        self.sfx.sort(key=lambda r: -len(r[0]))
        self.pfx.sort(key=lambda r: -len(r[0]))

    @staticmethod
    def _cond_ok(cond, stem):
        if cond == "." or not stem:
            return cond == "."
        if cond.startswith("["):
            try:
                return re.search(cond + "$", stem) is not None
            except re.error:
                return False
        return stem.endswith(cond)

    def stem(self, word):
        """-> (stem, affix). Stripped prefixes are marked with a leading '<'."""
        if not word or not LETTER.search(word):
            return word, ""
        for append, strip, cond in self.sfx:
            if not append or append == "0" or not word.endswith(append):
                continue
            base = word[:-len(append)]
            if not base:
                continue
            if self._cond_ok(cond, base + strip if strip else base):
                return base + strip, append
        for append, strip, cond in self.pfx:
            if not append or append == "0" or not word.startswith(append):
                continue
            base = word[len(append):]
            if base and self._cond_ok(cond, base):
                return (strip + base) if strip else base, "<" + append
        return word, ""


class BaselineWords:
    def __init__(self, path=None):
        with open(path or ROOT / "data" / "external" / "baseline_words_v1.txt",
                   encoding="utf-8") as f:
            self.words = {ln.strip() for ln in f if ln.strip()}

    def coverage(self, words):
        """-> (fraction_in_baseline, [out-of-baseline words]). Empty input -> (1.0, [])."""
        words = [w for w in words if w.strip()]
        if not words:
            return 1.0, []
        oob = [w for w in words if w not in self.words]
        return round(1 - len(oob) / len(words), 4), oob


MODES = ("surface-as-is", "stems-only", "stems-plus-suffixes")


def apply_mode(words, stems, mode):
    """Expand pre-stemmed (word, stem, affix) triples under one §4.3 counting mode."""
    if mode == "stems-only":
        return [s for _, s, _ in stems]
    if mode == "stems-plus-suffixes":
        out = []
        for _, s, a in stems:
            out.append(s)
            if a:
                out.append(a)
        return out
    return list(words)  # surface-as-is
