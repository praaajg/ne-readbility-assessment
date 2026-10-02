"""Minimal usage: score Nepali sentences with the frozen method set.

Run from the repo root:  python examples/score_nepali.py
Needs only the pip-installed package + committed data/external files (offline-safe).
"""
from nepali_readability import ReadabilityScorer

SENTS = [
    "मेरो नाम सिता हो ।",
    "विद्यालयमा ज्ञान र विज्ञानको संरचना छ ।",
    "पाठ्यक्रम विकास केन्द्रको लिखित स्वीकृतिबिना व्यापारिक प्रयोजनका लागि यसको पुरै वा आंशिक भाग हुबहु प्रकाशन गर्न पाइने छैन ।",
]

rs = ReadabilityScorer()
for s in SENTS:
    r = rs.score(s)
    print(f"{s[:48]}")
    print(f"  NRS-ease={r['nrs']['ease']:.1f}  grade~{r['ngl']['grade']} "
          f"({r['ngl']['band']})  coverage-ease={r['hunspell_cov']['ease']:.1f}")
print()
print("Easiest-first ranking:")
for r in rs.compare(SENTS):
    print(f"  NRS={r['nrs']['ease']:5.1f}  {r['text'][:40]}")
