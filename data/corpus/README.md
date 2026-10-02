# Corpus snapshot

`nepali-textbook-sentences.csv.gz` (8.8 MB, 288,115 rows) — the merged working
corpus behind the reported results: one row per sentence.

Columns: `id` (stable sentence id) · `grade` (textbook-assigned grade 1–12;
working proxy label, not measured difficulty — see README) · `subject` ·
`source_corpus` (A = Hugging Face `dineshkarki/nepali-textbooks-corpus`,
Apache-2.0; B = raw OCR files) · `sentence_text` (cleaned, boilerplate-filtered).

Grades 1–10 are the A∪B deduplicated working set; Grades 11–12 are A-only
(extrapolation set). Sentences are cleaned and boilerplate-filtered; the
per-sentence feature table behind the reported scores is derived from them.
