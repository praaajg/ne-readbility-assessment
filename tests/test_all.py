"""Unit tests — Phase 3b/4: script.py / halant.py / baseline_words.py.

All expectations are hand-derived from Unicode codepoints (§4, §4.2, §4.3); akshara
counts were hand-validated (10/10) against a naive baseline (9/10).
RULE: Devanagari virama/joiner expectations use \\u escapes, never literals
(lesson of the 3a U+0941-lookalike baseline bug).
"""
from nepali_readability import script as S
from nepali_readability import halant as H
from nepali_readability.baseline_words import (
    AffixStripper, BaselineWords, apply_mode,
)

VIR = "\u094d"  # halant, escape-only rule
ZWNJ = "\u200c"  # zero-width non-joiner


def test_codepoint_constants():
    assert ord(S.VIRAMA) == 0x94D  # never the U+0941 lookalike (see 3a baseline bug)
    assert {ord(c) for c in S.LONG_MATRAS} == {0x93E, 0x940, 0x942, 0x948, 0x94C, 0x947, 0x94B}
    assert {ord(c) for c in S.INDEPENDENT_VOWELS} == set(range(0x0904, 0x0915))
    assert set(range(0x0915, 0x093A)) <= {ord(c) for c in S.CONSONANTS}


def test_codepoints_jnana():
    # ज्ञान = ज + ् + ञ + ा + न
    c = S.count_codepoints("ज्ञान")
    assert (c["consonants"], c["viramas"], c["conjuncts"]) == (3, 1, 1)
    assert (c["matras"], c["matras_long"], c["matras_short"]) == (1, 1, 0)
    assert c["independent_vowels"] == 0 and c["total_chars"] == 5


def test_codepoints_sentence():
    # राम स्कुल जान्छ । — cons: र म स क ल ज न छ (8); viramas: स्क, न्छ (2)
    c = S.count_codepoints("राम स्कुल जान्छ ।")
    assert (c["consonants"], c["viramas"], c["conjuncts"]) == (8, 2, 2)
    assert (c["matras"], c["matras_long"], c["matras_short"]) == (3, 2, 1)


def test_explicit_final_halant_not_a_conjunct():
    # छ + ् + न + ् (escapes, never literals) — only the first virama joins consonants
    c = S.count_codepoints("\u091b\u094d\u0928\u094d")
    assert (c["viramas"], c["conjuncts"], c["consonants"]) == (2, 1, 2)


def test_halant_family_jnana():
    f = H.halant_features("ज्ञान")
    assert (f["virama_raw"], f["conjuncts"]) == (1, 1)
    assert (f["cons_raw"], f["cons_surface"], f["cons_merged"]) == (3, 2, 1)
    assert f["vowels_total"] == 1
    assert f["ratio_vc_raw"] == round(1 / 3, 4) and f["ratio_vc_surface"] == 0.5


def test_halant_config_toggles():
    f = H.halant_features("ज्ञान", {"use_raw_virama": False, "use_conjunct_count": False,
                                    "use_deletion_contrast": False,
                                    "log_vowel_consonant_ratios": False})
    assert f == {"vowels_total": 1}


def test_dev_ratio():
    assert S.devanagari_ratio("राम") == 1.0
    assert S.devanagari_ratio("abc") == 0.0
    assert S.devanagari_ratio("Fez राम") == 0.5


def test_strip_residue():
    assert S.strip_residue("जोडेर\\") == "जोडेर"


def test_strip_residue_zwnj():
    assert S.strip_residue("पर्दछन्" + ZWNJ) == "पर्दछन्"
    assert S.strip_residue("a" + ZWNJ + "b") == "ab"


def test_syllable_methods_differ_on_halant_final():
    assert S.count_syllables_per_word(["जान्छिन्"], method="akshara") == [3]
    assert S.count_syllables_per_word(["जान्छिन्"], method="__bogus_fallback__") == [2]
    assert S.count_syllables_per_word(["विद्यालय", "ज्ञान"], method="akshara") == [4, 2]


def test_spoken_syllables():
    cases = {"विद्यालय": 4, "ज्ञान": 2, "विज्ञान": 3, "संरचना": 4, "क्षेत्र": 2,
             "त्रिभुवन": 4, "स्कुल": 2, "कक्षा": 2, "शिक्षक": 3, "जान्छिन्": 3,
             "\u091b\u094d\u0928\u094d": 1,
             "राम": 2, "२०७८": 0, "Fez": 0}
    for w, exp in cases.items():
        assert S.spoken_syllables(w) == exp, w


def test_analyse_word_count():
    assert S.analyse("सीता पनि जान्छिन्")["words"] == 3


def test_affix_stripper_counts():
    st = AffixStripper()
    assert (len(st.sfx), len(st.pfx)) == (507, 1)


def test_affix_stems():
    st = AffixStripper()
    assert st.stem("गरेर") == ("गर" + VIR, "ेर")
    assert st.stem("किताबहरू") == ("किताब", "हरू")
    assert st.stem("विद्यालय") == ("विद्यालय", "")
    assert st.stem("२०७८") == ("२०७८", "")  # no letters: untouched


def test_baseline_coverage():
    bw = BaselineWords()
    assert len(bw.words) == 3104
    frac, oob = bw.coverage(["राम", "स्कुल", "जान्छ"])
    assert frac == round(2 / 3, 4) and oob == ["राम"]
    assert BaselineWords().coverage([]) == (1.0, [])


def test_morphology_modes():
    triples = [("गरेर", "गर" + VIR, "ेर"), ("विद्यालय", "विद्यालय", "")]
    assert apply_mode(["गरेर", "विद्यालय"], triples, "surface-as-is") == ["गरेर", "विद्यालय"]
    assert apply_mode(["गरेर", "विद्यालय"], triples, "stems-only") == ["गर" + VIR, "विद्यालय"]
    assert apply_mode(["गरेर", "विद्यालय"], triples, "stems-plus-suffixes") == [
        "गर" + VIR, "ेर", "विद्यालय"]
