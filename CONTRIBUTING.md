# Contributing

## Adding a readability method
1. Add the function to `nepali_readability/formulas.py` with the shared
   `features, params -> {raw, ease}` signature (weights as `params` defaults,
   documented as calibrated or uncalibrated).
2. Expose it through `ReadabilityScorer` if it should ship in the public API.
3. Add unit tests with hand-computed expectations in `tests/`.
4. Show a before/after example in `examples/` where it clarifies usage.

New weights/coefficients are added as new defaults alongside the old, never silent
edits — committed result tables must stay reproducible from the shipped code.

## Rules that apply to every change
- Never hardcode an absolute local path (Windows or otherwise) in any tracked file.
  The pre-commit check is a repo-wide grep for machine-specific path fragments —
  it must come back clean.
- Devanagari codepoints that are visually confusable or invisible (virama U+094D vs
  U+0941, ZWNJ/ZWJ) use `\u` escapes in code and test expectations, never literals.
- Every CSV in `data/` has a header. Large artifacts (>~10MB) stay gitignored —
  commit the small tables, never the bulk files.
- Run the suite before pushing: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q`.

## Questions adjudicated by linguists
Anything touching ground-truth assumptions or linguistic correctness goes through
linguist review first — code the mechanism only after the decision is recorded.
