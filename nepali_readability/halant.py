"""Halant/virama-deletion feature family (§4.2).

Not one number but a family of related features around the virama (् U+094D),
each toggled independently via configs/method_halant_deletion.yaml:

- virama_raw: raw virama/halant codepoint count.
- conjuncts: virama-joined consonant clusters (C + virama + C; explicit word-final
  halant excluded) — same count as script.count_codepoints()["conjuncts"].
- cons_raw: raw consonant codepoints (underlying phoneme inventory).
- cons_surface: consonant CLUSTER count after halant-triggered vowel suppression —
  each virama-joined run (C(virama+C)+) renders as ONE surface unit. The raw-vs-surface
  contrast is the candidate difficulty signal: more merging = heavier orthography.
- vowels_total: independent vowels + matras (unaffected by virama deletion).
- ratio_vc_raw / ratio_vc_surface: vowels/consonants before vs after deletion —
  logged as their own columns per §4.2, not intermediates.
"""
import unicodedata

from nepali_readability.script import (
    VIRAMA, CONSONANTS, INDEPENDENT_VOWELS, MATRAS, count_codepoints,
)


def _chars(text):
    return [c for c in unicodedata.normalize("NFC", text) if not c.isspace()]


def consonant_clusters(text):
    """Consonant surface units: maximal C(virama+C)* runs each count once."""
    chs = _chars(text)
    n, i = 0, 0
    while i < len(chs):
        if chs[i] in CONSONANTS:
            n += 1
            i += 1
            while i + 1 < len(chs) and chs[i] == VIRAMA and chs[i + 1] in CONSONANTS:
                i += 2  # swallowed into the same surface unit
        else:
            i += 1
    return n


def halant_features(text, config=None):
    """Full §4.2 family for one text span. Config toggles each variant."""
    cfg = config or {}
    base = count_codepoints(text)
    chs = _chars(text)
    vowels = base["independent_vowels"] + base["matras"]
    cons_raw = base["consonants"]
    cons_surface = consonant_clusters(text)
    feats = {
        "virama_raw": base["viramas"],
        "conjuncts": base["conjuncts"],
        "cons_raw": cons_raw,
        "cons_surface": cons_surface,
        "cons_merged": cons_raw - cons_surface,  # consonants absorbed into clusters
        "vowels_total": vowels,
        "ratio_vc_raw": round(vowels / cons_raw, 4) if cons_raw else 0.0,
        "ratio_vc_surface": round(vowels / cons_surface, 4) if cons_surface else 0.0,
    }
    if cfg.get("use_raw_virama") is False:
        feats.pop("virama_raw")
    if cfg.get("use_conjunct_count") is False:
        feats.pop("conjuncts")
    if cfg.get("use_deletion_contrast") is False:
        for k in ("cons_raw", "cons_surface", "cons_merged"):
            feats.pop(k)
    if cfg.get("log_vowel_consonant_ratios") is False:
        feats.pop("ratio_vc_raw")
        feats.pop("ratio_vc_surface")
    return feats
