"""Devanagari/Nepali unicode analyzer (§4, Phase 3b).

Tokenizer + syllable-counter choice is config-driven (§4.4 decision: nepali_grammar +
akshara; see configs/tokenizer_nepali_grammar.yaml, configs/syllable_akshara.yaml).
`analyse()` takes pre-tokenized words/syllable counts so experiment runners control the
combo; convenience wrappers default to the standardized combo with naive fallbacks.
"""
import re
import unicodedata

VIRAMA = "्"          # U+094D halant — the conjunct-former
ANUSVARA = "ं"         # U+0902
CHANDRABINDU = "ँ"      # U+0901
VISARGA = "ः"           # U+0903
AVAGRAHA = "ऽ"          # U+093D
DANDA = "।"             # U+0964
DOUBLE_DANDA = "॥"      # U+0965

INDEPENDENT_VOWELS = frozenset(chr(c) for c in range(0x0904, 0x0915))
CONSONANTS = frozenset(chr(c) for c in range(0x0915, 0x093A)) | \
    frozenset(chr(c) for c in range(0x0958, 0x0960))  # + nukta variants
MATRAS = frozenset(chr(c) for c in range(0x093E, 0x094D)) | \
    {chr(0x094E), chr(0x094F)} | frozenset(chr(c) for c in range(0x0955, 0x0958))

# Long ("heavy"/guru) matras — harder to read (same split as hindi-readability).
LONG_MATRAS = frozenset(["ा", "ी", "ू", "ै", "ौ", "े", "ो"])

DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
PUNCT_DETACH_RE = re.compile(r"([(),;:\"'“”‘’\[\]{}|/\\])")


def devanagari_ratio(text):
    """Fraction of non-space characters in the Devanagari block — quality flag (§1)."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if "ऀ" <= c <= "ॿ") / len(chars)


def strip_residue(text):
    """3b preprocessing (+4: also drop ZWNJ/ZWJ joiners lingering in OCR text)."""
    text = text.replace("\\", " ")
    text = text.replace("\u200c", "").replace("\u200d", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detach_punct(text):
    """3b preprocessing: detach punctuation so counts don't glue tokens (§3a finding)."""
    return PUNCT_DETACH_RE.sub(r" \1 ", text)


def count_codepoints(text):
    """Raw script-level counts (unicode rules only, no tokenizer involved)."""
    text = unicodedata.normalize("NFC", text)
    counts = {"total_chars": 0, "consonants": 0, "independent_vowels": 0,
              "matras": 0, "matras_long": 0, "matras_short": 0, "viramas": 0,
              "conjuncts": 0, "anusvara": 0, "visarga": 0}
    prev = ""
    for ch in text:
        if ch.isspace():
            prev = ch
            continue
        counts["total_chars"] += 1
        if ch in CONSONANTS:
            counts["consonants"] += 1
        elif ch in INDEPENDENT_VOWELS:
            counts["independent_vowels"] += 1
        elif ch in MATRAS:
            counts["matras"] += 1
            if ch in LONG_MATRAS:
                counts["matras_long"] += 1
            else:
                counts["matras_short"] += 1
        elif ch == VIRAMA:
            counts["viramas"] += 1
        elif ch == ANUSVARA:
            counts["anusvara"] += 1
        elif ch == VISARGA:
            counts["visarga"] += 1
        prev = ch
    # Correct conjuncts with lookahead: virama counts iff followed by a consonant.
    conj = 0
    chs = [c for c in unicodedata.normalize("NFC", text) if not c.isspace()]
    for i, ch in enumerate(chs):
        if ch == VIRAMA and i > 0 and chs[i - 1] in CONSONANTS \
                and i + 1 < len(chs) and chs[i + 1] in CONSONANTS:
            conj += 1
    counts["conjuncts"] = conj
    return counts


def segment_words(text, method="nepali_grammar"):
    """Word tokenizer dispatch (§4.4). Falls back to whitespace split."""
    if method == "nepali_grammar":
        try:
            from nepali_tokenizer import NepaliTokenizer
            return [x.text for x in NepaliTokenizer().segment_words(text)]
        except Exception:
            pass
    elif method == "indic":
        try:
            from indicnlp.tokenize import trivial_tokenize
            return trivial_tokenize.tokenize(text, lang="hi")
        except Exception:
            pass
    return text.split()


def count_syllables_per_word(words, method="akshara"):
    """Syllable counter dispatch (§4.4). Falls back to the naive estimator."""
    if method == "akshara":
        try:
            from akshara_tokenizer import count_aksharas
            return [count_aksharas(w) for w in words]
        except Exception:
            pass
    return [_naive_syllables(w) for w in words]


def _naive_syllables(word):
    chs = list(unicodedata.normalize("NFC", word))
    n = 0
    for i, ch in enumerate(chs):
        if ch in INDEPENDENT_VOWELS:
            n += 1
        elif ch in CONSONANTS:
            nxt = chs[i + 1] if i + 1 < len(chs) else ""
            if nxt != VIRAMA:
                n += 1
    return n


def spoken_syllables(word):
    """Spoken-form syllable estimate with silent-halant handling.

    Counts vowel nuclei: independent vowels + matras + consonant clusters whose
    inherent /a/ survives (i.e. the cluster is NOT followed by a matra). A
    maximal C(virama+C)* run counts once — including an explicit word-final
    halant cluster (छन् -> 1), which the naive estimator scores 0.
    Non-Devanagari tokens score 0.
    """
    chs = list(unicodedata.normalize("NFC", word))
    n, i, L = 0, 0, len(chs)
    while i < L:
        ch = chs[i]
        if ch in INDEPENDENT_VOWELS or ch in MATRAS:
            n += 1
            i += 1
        elif ch in CONSONANTS:
            j = i + 1
            while j + 1 < L and chs[j] == VIRAMA and chs[j + 1] in CONSONANTS:
                j += 2
            if j < L and chs[j] in MATRAS:
                i = j  # land on the matra; it counts next iteration
            else:
                n += 1
                i = j if j > i else i + 1
        else:
            i += 1
    return n


def analyse(text, words=None, syllables_per_word=None,
            word_method="nepali_grammar", syllable_method="akshara"):
    """Full per-text analysis. Pass words/syllables to pin the §4.4 combo explicitly."""
    cleaned = detach_punct(strip_residue(text))
    counts = count_codepoints(cleaned)
    if words is None:
        words = [w for w in segment_words(cleaned, method=word_method) if w.strip()]
    if syllables_per_word is None:
        syllables_per_word = count_syllables_per_word(words, method=syllable_method)
    counts["words"] = len(words)
    counts["syllables"] = sum(syllables_per_word)
    counts["dev_ratio"] = round(devanagari_ratio(text), 4)
    return counts
