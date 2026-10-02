"""Every readability formula, one function each, same call signature (Phase 5, §4.1).

Contract: f(features, params) -> {"raw": float, "ease": float 0..100 (higher=easier)}.
features = a row of data/processed/features_v3_*.csv (string or numeric values).
params   = the method's configs/method_*.yaml (weights versioned, never hardcoded).

Coefficient policy (§15): classic/Hindi weights are v0 STARTING points (uncalibrated for
Nepali — stated per function). Calibration happens in Phase 6 as NEW config variants.
§8 regression of grade on score is AFFINE-INVARIANT, so each method's raw->ease display
map cannot affect rankings — it is cosmetic and documented as such.

Method set (grid §4.1): flesch_np, fog_np, ari_np, dale_np (classic analogs);
nrs, ngl_from_nrs (Nepali family: Hindi-calibrated v0 weights verbatim from
hindi-readability 0.3.0 CODE: HRS=206-60*spw-1.8*wps-70*conj/word-8*matra — note
its docstring describes a different variant); nci (Nepali-refit v1, simplex grid
on train grades, adopted §10: 0.35*syllable+0.35*sentence+0.30*conjunct+0.0*matra);
halant_ratio, hunspell_cov, matra_ease, conj_ease (Nepali-specific).
"""
import math

METHODS = ["flesch_np", "fog_np", "ari_np", "dale_np", "nrs", "nci",
           "halant_ratio", "hunspell_cov", "matra_ease", "conj_ease"]


def _f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _clip(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def _wps(f):
    return max(_f(f, "sentence_length_words"), 1.0)


def _spw(f):
    return _f(f, "syllables_per_word")


def _conj_pw(f):
    return _f(f, "conjuncts") / max(_f(f, "word_count"), 1.0)


def _matra_complex(f):
    m = _f(f, "matras")
    return _f(f, "matras_long") / m if m else 0.0


def _grade_ease(raw, params):
    """Display-only affine map grade-like raw -> 0..100 ease (default: grade 0->100)."""
    scale = float(params.get("grade_ease_scale", 6.0))
    return _clip(100.0 - scale * raw)


def flesch_np(features, params):
    p = {"wps": 1.015, "spw": 84.6, "intercept": 206.835, **params}
    raw = p["intercept"] - p["wps"] * _wps(features) - p["spw"] * _spw(features)
    return {"raw": raw, "ease": _clip(raw)}


def fog_np(features, params):
    p = {"intercept": 0.0, "w_wps": 0.4, "w_complex": 0.4, **params}
    words = max(_f(features, "word_count"), 1.0)
    complex_frac = _f(features, "n_poly3") / words
    raw = p["intercept"] + p["w_wps"] * _wps(features) + p["w_complex"] * 100.0 * complex_frac
    return {"raw": raw, "ease": _grade_ease(raw, params)}


def ari_np(features, params):
    p = {"intercept": -21.43, "w_cpw": 4.71, "w_wps": 0.5, **params}
    words = max(_f(features, "word_count"), 1.0)
    cpw = _f(features, "n_nospace") / words
    raw = p["intercept"] + p["w_cpw"] * cpw + p["w_wps"] * _wps(features)
    return {"raw": raw, "ease": _grade_ease(raw, params)}


def dale_np(features, params):
    p = {"w_pdw": 0.1579, "w_wps": 0.0496, "hard_bonus": 3.6365,
         "intercept": 0.0, **params}
    pdw = 100.0 * (1.0 - _f(features, "bl_cov"))
    raw = p["intercept"] + p["w_pdw"] * pdw + p["w_wps"] * _wps(features)
    if pdw > 5.0:
        raw += p["hard_bonus"]
    return {"raw": raw, "ease": _grade_ease(raw, params)}


def nrs(features, params):
    """Nepali Readability Score (ease 0-100, higher=easier).

    v0 weights are the Hindi HRS weights verbatim (uncalibrated for Nepali).
    """
    p = {"spw": 60.0, "wps": 1.8, "conj": 70.0, "matra": 8.0,
         "intercept": 206.0, **params}
    raw = (p["intercept"] - p["spw"] * _spw(features) - p["wps"] * _wps(features)
           - p["conj"] * _conj_pw(features) - p["matra"] * _matra_complex(features))
    return {"raw": raw, "ease": _clip(raw)}


def nci(features, params):
    """Nepali Complexity Index (0-1, lower=easier). v1 = Nepali refit, adopted §10
    (simplex grid step 0.05 maximizing train Pearson grade-vs-blend; matra→0)."""
    p = {"syl": 0.35, "sent": 0.35, "conj": 0.30, "matra": 0.0,
         "syl_cap": 5.0, "sent_cap": 30.0, "conj_cap": 1.0, **params}
    words = max(_f(features, "word_count"), 1.0)
    nci = (p["syl"] * min(_spw(features), p["syl_cap"]) / p["syl_cap"]
           + p["sent"] * min(words / 1.0, p["sent_cap"]) / p["sent_cap"]
           + p["conj"] * min(_conj_pw(features), p["conj_cap"])
           + p["matra"] * _matra_complex(features))
    nci = max(0.0, min(1.0, nci))
    return {"raw": nci, "ease": _clip(100.0 * (1.0 - nci))}


def ngl_from_nrs(nrs, params):
    """Hindi-calibrated NRS->grade map (UNCALIBRATED for Nepali; CDC bands, not CBSE)."""
    p = {"intercept": 10.083, "slope": 0.1088, **params}
    grade = max(1, min(13, round(p["intercept"] - p["slope"] * nrs)))
    if grade <= 2:
        band = "Grade 1-2"
    elif grade <= 5:
        band = "Grade 3-5"
    elif grade <= 8:
        band = "Grade 6-8"
    elif grade <= 10:
        band = "Grade 9-10"
    elif grade <= 12:
        band = "Grade 11-12"
    else:
        band = "College+"
    return {"grade": grade, "band": band}


def halant_ratio(features, params):
    v = _f(features, "vowels_total")
    c = _f(features, "cons_surface")
    ease = 100.0 * v / (v + c) if (v + c) else 50.0
    return {"raw": ease, "ease": _clip(ease)}


def hunspell_cov(features, params):
    ease = 100.0 * _f(features, "bl_cov")
    return {"raw": ease, "ease": _clip(ease)}


def matra_ease(features, params):
    ease = 100.0 * (1.0 - _matra_complex(features))
    return {"raw": ease, "ease": _clip(ease)}


def conj_ease(features, params):
    ease = 100.0 * (1.0 - min(_conj_pw(features), 1.0))
    return {"raw": ease, "ease": _clip(ease)}


DISPATCH = {m: globals()[m] for m in METHODS}


def score_all(features, configs):
    """Score one feature row with every method. configs: {method: params}."""
    return {m: DISPATCH[m](features, configs.get(m, {})) for m in METHODS}
