# CREDITS

- HF dataset: `dineshkarki/nepali-textbooks-corpus` (Grades 1–12 CDC textbooks, 5,634 segments; Apache-2.0).
- Local raw OCR textbook files, Grades 1–10 (secondary corpus, provided outside the HF corpus).
- `Erprabhat8423/hindi-readability` (PyPI `hindi-readability` 0.3.0) — **structural reference only**: package layout (`__init__/script/formulas/scorer` + tests/data split) and API shape (`score/compare/batch_score`) informed ours; **no code copied**. Its HRS/HCI *weights* were reused verbatim solely as cited, uncalibrated starting points for our NRS/NCI. The Nepali refit has since been run: NRS kept v0 (no gain), NCI v1 (0.35/0.35/0.30/0.0) adopted as the shipped default.
- Hunspell `ne_NP` dictionary (LibreOffice/dictionaries `ne_NP/`, Madan Puraskar Pustakalaya, Rel. 1.1 2006, LGPL 2.1) + tokenizer/syllable packages (`nepali-grammar-tokenizer`, `akshara-tokenizer`, `indic-nlp-library` — pinned in `pyproject.toml`). Engine: `spylls` (pure-Python Hunspell lookup).
- Evaluated but NOT adopted: `npltk` (torch-weight import, no counting edge), `Nepali_nlp` Tokenizer (splits on । only, destroys abbreviations, PyPI sdist uninstallable).
- Papers: L16-1038 (OSMAN/Arabic); arXiv 2502.13520 (BAREC); ArabicNLP 2024-1.5 (SAMER readability); ICON 2014 File32-p92 (Hindi SVM/SVR); COLING 2012 C12-2111 (Hindi/Bangla models).
- Compute stack: numpy, scipy (see `pyproject.toml`).
