"""Report Table 15: leakage, shuffled labels, signal vs no signal, reproducibility."""

import numpy as np
import pytest

from sklearn.metrics import roc_auc_score

from rad.validation import (
    bootstrap_ci,
    compute_metrics,
    expected_calibration_error,
    holm,
    net_benefit,
    paired_bootstrap_auc,
    permutation_test,
    run_nested_cv,
    summarize,
)
from tests.conftest import make_cohort


def _auc(cfg, X, y, model):
    result = run_nested_cv(X, y, model, cfg)
    summary, pooled = summarize(y, result.oof)
    return summary["auc"], pooled, result


@pytest.mark.parametrize("model", ["logreg", "svm", "rf"])
def test_signal_is_found(config, cohort, model):
    X, y = cohort
    auc, _, _ = _auc(config, X, y, model)
    assert auc > 0.75


def test_shuffled_labels_give_chance_level(config):
    # many noise features + selection: any leakage of labels into selection would lift this above 0.5
    X, y = make_cohort(n=120, n_noise=300, signal=0.0, seed=3)
    shuffled = np.random.default_rng(1).permutation(y)
    auc, _, _ = _auc(config, X, shuffled, "logreg")
    assert 0.38 < auc < 0.62


def test_preprocessing_only_sees_training_rows(config, cohort):
    X, y = cohort
    result = run_nested_cv(X, y, "logreg", config)
    for fold in result.folds:
        assert set(fold["fit_index"]).isdisjoint(fold["test_index"])
        assert len(fold["fit_index"]) + len(fold["test_index"]) == len(X)
    assert not np.isnan(result.oof).any()
    for fold in result.folds:
        radiomic = [c for c in fold["selected"] if not c.startswith("clin__")]
        assert 1 <= len(radiomic) <= 15


def test_same_seed_same_result(config, cohort):
    X, y = cohort
    assert np.array_equal(run_nested_cv(X, y, "logreg", config).oof, run_nested_cv(X, y, "logreg", config).oof)


def test_prior_baseline_is_chance(config, cohort):
    X, y = cohort
    auc, _, _ = _auc(config, X, y, "prior")
    assert auc == pytest.approx(0.5, abs=0.02)


def test_metrics_are_sane():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.0, 0.1, 0.9, 1.0])
    m = compute_metrics(y, perfect)
    assert m["auc"] == 1.0 and m["balanced_accuracy"] == 1.0 and m["sensitivity"] == 1.0
    assert expected_calibration_error(y, perfect) < 0.15


def test_xgboost_is_usable_when_installed(config, cohort):
    try:
        import xgboost  # noqa: F401
    except Exception as exc:  # package missing, or its OpenMP runtime missing (macOS: brew install libomp)
        pytest.skip(f"xgboost is not usable here: {type(exc).__name__}")
    X, y = cohort
    auc, _, _ = _auc(config, X, y, "xgboost")
    assert auc > 0.7


def test_bootstrap_interval_contains_the_estimate(config, cohort):
    X, y = cohort
    _, pooled, _ = _auc(config, X, y, "logreg")
    lo, hi = bootstrap_ci(y, pooled, n=300)
    assert lo < roc_auc_score(y, pooled) < hi


def test_paired_bootstrap_separates_signal_from_noise():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    good = y * 0.8 + rng.normal(size=300) * 0.5
    bad = rng.normal(size=300)
    diff, lo, hi, p = paired_bootstrap_auc(y, good, bad, n=400)
    assert diff > 0.2 and lo > 0 and p < 0.05
    _, _, _, p_same = paired_bootstrap_auc(y, good, good + 1e-9, n=200)
    assert p_same >= 0.0


def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert max(adj) <= 1.0


def test_permutation_test_p_value_ranges(config):
    X, y = make_cohort(n=100, n_noise=30, signal=1.5, seed=5)
    config["validation"].update(repeats=1, outer_folds=3)
    auc, _, _ = _auc(config, X, y, "logreg")
    p, null = permutation_test(X, y, "logreg", config, None, auc, n_perm=6, seed=1)
    assert p <= 2 / 7 and 0.3 < null.mean() < 0.7


def test_net_benefit_of_perfect_model_beats_treat_all():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    model, all_ = net_benefit(y, y.astype(float), [0.2, 0.5])
    assert all(m >= a for m, a in zip(model, all_))
