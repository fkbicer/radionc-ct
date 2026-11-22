"""Report Table 15: leakage, shuffled labels, signal vs no signal, reproducibility."""

import numpy as np
import pytest

from rad.validation import compute_metrics, expected_calibration_error, run_nested_cv, summarize
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
