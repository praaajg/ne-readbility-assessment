"""Unit tests — Phase 6: metrics.py statistics + split invariants."""
import numpy as np
import pytest

from nepali_readability.metrics import (
    acc_within_k,
    bootstrap_ci,
    cliffs_delta,
    grade_stratified_group_split,
    holm_bonferroni,
    mae,
    ols_fit,
    ols_predict,
    paired_bootstrap_diff,
    pearson,
    spearman,
    wilcoxon_paired,
)


def test_rank_metrics():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 1, 1], [2, 4, 6]) == 0.0  # zero variance guard
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert mae([1, 2, 3], [1, 2, 4]) == pytest.approx(1 / 3)
    assert acc_within_k([1, 2, 5], [1, 3, 5], 1) == pytest.approx(1.0)


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(0)
    x = rng.normal(5, 1, 200)
    mean, lo, hi = bootstrap_ci(np.mean, x, n_boot=200)
    assert lo <= mean <= hi and lo <= 5 <= hi


def test_cliffs_delta():
    assert cliffs_delta([3, 3, 3], [1, 1, 1]) == pytest.approx(1.0)
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_paired_diff_detects_gap():
    y = np.arange(50, dtype=float)
    mean, lo, hi = paired_bootstrap_diff(mae, y, y, y + 0.5, n_boot=200)
    assert mean == pytest.approx(-0.5) and hi < 0  # constant gap excludes zero


def test_holm_bonferroni():
    assert holm_bonferroni([0.001, 0.04, 0.5]) == [True, False, False]
    assert wilcoxon_paired([1, 2, 3], [1, 2, 3]) == 1.0  # identical guard


def test_ols_fit_exact():
    import numpy as np
    X = [[1.0], [2.0], [3.0], [4.0]]
    y = [3.0, 5.0, 7.0, 9.0]  # y = 1 + 2x exactly
    assert ols_fit(X, y) == pytest.approx([1.0, 2.0])
    assert list(ols_predict([1.0, 2.0], [[5.0]])) == pytest.approx([11.0])
    with pytest.raises(ValueError):
        ols_fit([[1.0], [1.0]], [2.0, 3.0])  # constant column


def test_split_invariants():
    rows = [{"id": f"g{g}c{c}s{i}", "grade": g, "group": f"g{g}c{c}"}
            for g in (1, 2) for c in range(10) for i in range(10)]
    assign = grade_stratified_group_split(rows, "grade", "group", test_frac=0.2,
                                          n_folds=5, seed=42)
    assert set(assign) == {r["id"] for r in rows}
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], set()).add(assign[r["id"]][0])
    assert all(len(v) == 1 for v in groups.values())  # no group spans splits
    folds = {assign[r["id"]][1] for r in rows if assign[r["id"]][0] == "train"}
    assert folds == {0, 1, 2, 3, 4}
