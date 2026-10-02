# `data/processed/` — naming and versioning convention

All committed result artifacts are lowercase `snake_case` CSVs (plus versioned
model pickles under `models/`), named `{family}_{detail}_v1.csv`:

- `scores_{method}_v1.csv` / `stats_{method}_v1.csv` — per-method outputs.
- `formula_*_v1.csv` — cross-method analysis tables (CIs, cliffs, correlations,
  test metrics).
- `comparison_ranking_v1.csv`, `pairwise_significance_v1.csv`, `refit_comparison_v1.csv`
- `anomaly_clusters_v1.csv`, `cross_grade_dupes_v1.csv`, `domain_nci_v1.csv`.
- `models/{name}_v1.{pkl,_metrics.csv,_importance.csv}`.
- `splits/frozen_v2.csv` — the active frozen split (v2 = strict 1–10, book-grouped);
  `splits/frozen_v1_retired_chapter.csv` — the retired v1 split, kept for history.

The `_v1` suffix is artifact-lineage versioning (first committed lineage), decoupled
from the split version: v1-named tables were regenerated in place onto the v2 split
during the 2026-09-30 re-freeze rather than forked into parallel `_v2` files.
Regenerating runners live under `experiments/` (local-only); committed CSVs are
their outputs, checked by `experiments/10_replicability/verify.py`. The large
intermediates (`features_v*`, `scores_*`) are gitignored but rebuildable from the
committed snapshot via root `rebuild.py` (only `chapter_title` differs, unread
downstream). A `stable_id` content-hash column populates automatically on the next
end-to-end `rebuild.py` run (computed inline from snapshot text — no extra input);
current files predate it (see `docs/07_data_merge.md`).
