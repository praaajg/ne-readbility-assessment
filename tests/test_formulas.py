import pytest

from nepali_readability import formulas as F
from nepali_readability.scorer import ReadabilityScorer

FEATS = {
    "sentence_length_words": 10, "word_count": 10, "syllables_per_word": 2.0,
    "conjuncts": 5, "matras": 10, "matras_long": 3,
    "vowels_total": 3, "cons_surface": 6, "bl_cov": 0.6,
    "n_poly3": 2, "n_nospace": 50,
}


def test_contract_keys():
    out = F.score_all(FEATS, {})
    assert set(out) == set(F.METHODS)
    for v in out.values():
        assert set(v) == {"raw", "ease"} and 0.0 <= v["ease"] <= 100.0


def test_nrs_hand():
    # 206 - 60*2 - 1.8*10 - 70*0.5 - 8*0.3 = 30.6
    r = F.nrs(FEATS, {})
    assert r == {"raw": 30.6, "ease": 30.6}


def test_flesch_np_hand():
    # 206.835 - 1.015*10 - 84.6*2 = 27.485
    assert F.flesch_np(FEATS, {})["raw"] == pytest.approx(27.485)


def test_fog_np_hand():
    # 0.4*(10 + 100*0.2) = 12.0 ; ease = 100-72
    r = F.fog_np(FEATS, {})
    assert r["raw"] == pytest.approx(12.0) and r["ease"] == pytest.approx(28.0)


def test_ari_np_hand():
    # 4.71*5 + 0.5*10 - 21.43 = 7.12 ; ease = 100-42.72
    r = F.ari_np(FEATS, {})
    assert r["raw"] == pytest.approx(7.12) and r["ease"] == pytest.approx(57.28)


def test_dale_np_hand():
    # pdw=40: 0.1579*40 + 0.0496*10 + 3.6365 = 10.4485
    r = F.dale_np(FEATS, {})
    assert r["raw"] == pytest.approx(10.4485)


def test_nci_hand():
    # v1: 0.35*0.4 + 0.35*(10/30) + 0.30*0.5 + 0.0*0.3 = 0.40667
    r = F.nci(FEATS, {})
    assert r["raw"] == pytest.approx(0.40667, abs=1e-4)
    assert r["ease"] == pytest.approx(59.333, abs=1e-2)


def test_ngl_bands():
    assert F.ngl_from_nrs(88.4, {}) == {"grade": 1, "band": "Grade 1-2"}
    assert F.ngl_from_nrs(0.0, {})["grade"] == 10


def test_nepali_specific():
    assert F.halant_ratio(FEATS, {})["ease"] == pytest.approx(100 * 3 / 9)
    assert F.hunspell_cov(FEATS, {})["ease"] == pytest.approx(60.0)
    assert F.matra_ease(FEATS, {})["ease"] == pytest.approx(70.0)
    assert F.conj_ease(FEATS, {})["ease"] == pytest.approx(50.0)


def test_scorer_smoke_and_empty():
    rs = ReadabilityScorer()
    out = rs.score("राम स्कुल जान्छ ।")
    assert set(out) == set(F.METHODS) | {"ngl"}
    assert all(0.0 <= out[m]["ease"] <= 100.0 for m in F.METHODS)
    with pytest.raises(ValueError):
        rs.score("   ")
    cmp_ = rs.compare(["मेरो नाम सिता हो ।",
                       "पाठ्यक्रम विकास केन्द्रको लिखित स्वीकृतिबिना व्यापारिक प्रयोजनका लागि यसको पुरै वा आंशिक भाग हुबहु प्रकाशन गर्न पाइने छैन ।"])
    assert cmp_[0]["nrs"]["ease"] >= cmp_[1]["nrs"]["ease"]
