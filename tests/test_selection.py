import numpy as np
import pandas as pd

from rad.selection import (
    CorrelationFilter,
    ElasticNetSelector,
    MissingRateFilter,
    NearConstantFilter,
    StabilityFilter,
    build_preprocessor,
)
from tests.conftest import SELECTION, make_cohort


def test_cleaning_filters_drop_what_they_should():
    X, y = make_cohort()
    assert "gtv__mostly_missing" not in MissingRateFilter(0.2).fit(X).columns_
    assert "gtv__constant" not in NearConstantFilter().fit(X.fillna(0)).columns_


def test_stability_filter_uses_icc_table_and_keeps_unlisted():
    X = pd.DataFrame({"a": [1.0, 2], "b": [1.0, 2], "c": [1.0, 2], "d": [1.0, 2]})
    icc = pd.Series({"a": 0.9, "b": 0.4, "c": np.nan})
    kept = StabilityFilter(icc, 0.75).fit(X).columns_
    assert kept == ["a", "d"]
    assert StabilityFilter(None).fit(X).columns_ == list(X.columns)


def test_correlated_pair_collapses_to_the_more_informative_one():
    X, y = make_cohort()
    Xs = X[["gtv__informative_0", "gtv__informative_0_copy", "gtv__noise_0", "gtv__noise_1"]]
    kept = CorrelationFilter(0.9).fit(Xs, y).columns_
    assert sum(c.startswith("gtv__informative_0") for c in kept) == 1


def test_elasticnet_selector_respects_k_max_and_finds_signal():
    X, y = make_cohort()
    Xn = X[[c for c in X.columns if c.startswith("gtv__") and c not in ("gtv__constant", "gtv__mostly_missing")]]
    Xn = (Xn - Xn.mean()) / Xn.std()
    sel = ElasticNetSelector(C=0.1, l1_ratio=0.5, k_max=10).fit(Xn, y)
    assert 1 <= len(sel.selected_) <= 10
    assert any(c.startswith("gtv__informative") for c in sel.selected_)


def test_elasticnet_fallback_when_everything_is_zeroed():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 30)), columns=[f"f{i}" for i in range(30)])
    y = rng.integers(0, 2, size=60)
    sel = ElasticNetSelector(C=1e-4, l1_ratio=1.0, fallback_k=4).fit(X, y)
    assert sel.used_fallback_ and len(sel.selected_) == 4


def test_full_preprocessor_reduces_to_a_handful_and_keeps_clinical():
    X, y = make_cohort()
    prep = build_preprocessor(SELECTION).fit(X, y)
    out = prep.transform(X)
    radiomic = [c for c in out.columns if not c.startswith("clin__")]
    assert 1 <= len(radiomic) <= 15
    assert {"clin__age", "clin__male"} <= set(out.columns)
    assert "gtv__constant" not in out.columns and "gtv__mostly_missing" not in out.columns
    assert not out.isna().any().any()


def test_preprocessor_works_with_clinical_only_and_radiomic_only():
    X, y = make_cohort()
    only_clin = build_preprocessor(SELECTION).fit(X[["clin__age", "clin__male"]], y)
    assert list(only_clin.transform(X[["clin__age", "clin__male"]]).columns) == ["clin__age", "clin__male"]
    only_rad = build_preprocessor(SELECTION).fit(X[[c for c in X.columns if c.startswith("gtv__")]], y)
    assert only_rad.transform(X[[c for c in X.columns if c.startswith("gtv__")]]).shape[1] >= 1
