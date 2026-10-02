"""Shared evaluation statistics (§8.1, §9).

All resampling is seeded and deterministic. Conventions:
- bootstrap_ci: percentile CI from resampling with replacement (default 1000).
- cliffs_delta: rank-based effect size in [-1, 1]; |d|<0.147 negligible, <0.33 small,
  <0.474 medium, else large (Vargha-Delaney thresholds).
- holm_bonferroni: adjusted reject decisions for m p-values at level alpha.
- paired_bootstrap_diff: CI on (metric(A) - metric(B)) over joint resamples — the
  §8.1-mandated paired comparison (excludes zero => significant difference).
- ols_fit: ordinary least squares weights (with intercept) for refitting formula
  coefficients against labels. Deterministic, closed-form, no seed needed.
"""
import numpy as np
from scipy import stats as _ss

RNG_SEED = 42


def _rng(seed=RNG_SEED):
    return np.random.default_rng(seed)


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return float(_ss.spearmanr(x, y).statistic)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def acc_within_k(y_true, y_pred, k=1):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= k))


def bootstrap_ci(stat_fn, data, n_boot=1000, ci=95, seed=RNG_SEED):
    rng = _rng(seed)
    data = np.asarray(data)
    n = len(data)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = stat_fn(data[idx])
    lo, hi = np.percentile(vals, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(vals.mean()), float(lo), float(hi)


def mean_ci(x, n_boot=1000, seed=RNG_SEED):
    return bootstrap_ci(np.mean, x, n_boot=n_boot, seed=seed)


def cliffs_delta(a, b):
    """P(X_a > X_b) - P(X_a < X_b) via broadcasting (exact, no sampling)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    return float((gt - lt) / (len(a) * len(b)))


def paired_bootstrap_diff(metric_fn, y_true, pred_a, pred_b, n_boot=1000,
                          seed=RNG_SEED):
    """CI on metric(y_true, pred_a) - metric(y_true, pred_b) over joint resamples.

    The single metric_fn form (not a pre-differenced lambda) is deliberate:
    passing the difference as metric_fn with duplicated args silently yields
    identically-zero intervals — this signature makes that misuse impossible.
    """
    rng = _rng(seed)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric_fn(y_true[idx], pred_a[idx]) - metric_fn(y_true[idx], pred_b[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def holm_bonferroni(p_values, alpha=0.05):
    """-> list[bool] reject decisions in input order."""
    m = len(p_values)
    order = np.argsort(p_values)
    reject = np.zeros(m, dtype=bool)
    for rank, idx in enumerate(order):
        if p_values[idx] <= alpha / (m - rank):
            reject[idx] = True
        else:
            break
    return reject.tolist()


def wilcoxon_paired(a, b):
    """Two-sided Wilcoxon signed-rank p-value on paired errors (non-parametric)."""
    try:
        return float(_ss.wilcoxon(a, b, zero_method="wilcox").pvalue)
    except ValueError:
        return 1.0


def grade_stratified_group_split(frame_rows, grade_key, group_key, test_frac=0.15,
                                 n_folds=5, seed=RNG_SEED):
    """Assign chapters to test (grade-stratified greedy) + train folds (round-robin).

    frame_rows: iterable of dicts with grade_key/group_key/id. Returns {id: (split, fold)}
    with fold=-1 for test. Same chapter never spans splits (§9 leakage rule).
    """
    rng = np.random.default_rng(seed)
    by_grade = {}
    for r in frame_rows:
        by_grade.setdefault(r[grade_key], []).append(r)
    assign = {}
    for grade in sorted(by_grade):
        chapters = {}
        for r in by_grade[grade]:
            chapters.setdefault(r[group_key], []).append(r["id"])
        names = list(chapters)
        rng.shuffle(names)
        sizes = {c: len(chapters[c]) for c in names}
        target = test_frac * sum(sizes.values())
        test, acc = set(), 0
        for c in sorted(names, key=lambda c: -sizes[c]):
            if acc < target:
                test.add(c)
                acc += sizes[c]
        folds = [[] for _ in range(n_folds)]
        rest = [c for c in names if c not in test]
        for i, c in enumerate(rest):
            folds[i % n_folds].append(c)
        fold_of = {c: i for i, fs in enumerate(folds) for c in fs}
        for c, ids in chapters.items():
            for i in ids:
                assign[i] = ("test", -1) if c in test else ("train", fold_of[c])
    return assign


def ols_fit(X, y):
    """Closed-form OLS weights with intercept first: returns [b0, b1, ...].

    Used to refit formula coefficients against labels (e.g. grade proxy).
    Deterministic; raises on degenerate (all-constant) input.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    A = np.column_stack([np.ones(len(X)), X])
    coef, residuals, rank, _ = np.linalg.lstsq(A, y, rcond=None)
    if rank < A.shape[1]:
        raise ValueError("degenerate design matrix (constant column?)")
    return [float(c) for c in coef]


def ols_predict(coef, X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return coef[0] + X @ np.array(coef[1:])
