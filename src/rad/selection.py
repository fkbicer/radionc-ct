"""Fold-internal cleaning and feature selection (report 4.6).

Everything here is a scikit-learn transformer that learns only in ``fit``. Put
inside a Pipeline that is fitted on a training fold, nothing about the test
fold (or the labels outside the fold) reaches the selection.

Radiomic columns go through: missing-rate filter -> median imputation ->
near-constant filter -> stability (ICC) filter -> z-score -> correlation
clusters -> elastic-net selection (about 10-15 features). Clinical columns
(``clin__*``) are only imputed and scaled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _sklearn_older_than(major, minor):
    parts = sklearn.__version__.split(".")
    return (int(parts[0]), int(parts[1])) < (major, minor)


def make_enet_logreg(C, l1_ratio, seed=0, max_iter=5000):
    """Elastic-net logistic regression across scikit-learn versions.

    ``penalty`` is deprecated from 1.8 (l1_ratio alone selects elastic-net) but
    required before it.
    """
    kwargs = dict(solver="saga", C=C, l1_ratio=l1_ratio, class_weight="balanced",
                  max_iter=max_iter, random_state=seed)
    if _sklearn_older_than(1, 8):
        kwargs["penalty"] = "elasticnet"
    return LogisticRegression(**kwargs)


class MissingRateFilter(BaseEstimator, TransformerMixin):
    def __init__(self, max_missing=0.2):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        rate = X.isna().mean()
        self.columns_ = list(X.columns[(rate <= self.max_missing) & (rate < 1.0)])
        return self

    def transform(self, X):
        return X[self.columns_]


class NearConstantFilter(BaseEstimator, TransformerMixin):
    def __init__(self, variance=1e-8):
        self.variance = variance

    def fit(self, X, y=None):
        keep = (X.nunique() > 1) & (X.var() > self.variance)
        self.columns_ = list(X.columns[keep])
        return self

    def transform(self, X):
        return X[self.columns_]


class StabilityFilter(BaseEstimator, TransformerMixin):
    """Drop features with ICC below ``min_icc``. ``icc`` is a Series indexed by feature
    name computed from mask perturbations (no labels). With ``icc=None`` it is a no-op;
    a feature absent from the table is kept."""

    def __init__(self, icc=None, min_icc=0.75):
        self.icc = icc
        self.min_icc = min_icc

    def fit(self, X, y=None):
        if self.icc is None:
            self.columns_ = list(X.columns)
        else:
            lookup = self.icc.to_dict()
            self.columns_ = [
                c for c in X.columns
                if c not in lookup or (np.isfinite(lookup[c]) and lookup[c] >= self.min_icc)
            ]
        return self

    def transform(self, X):
        return X[self.columns_]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """One representative per cluster of features with |Spearman rho| > threshold.

    Features are visited from the most to the least associated with the label
    (ANOVA F, training fold only); a feature is kept unless it is too correlated
    with one already kept.
    """

    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y):
        if X.shape[1] <= 1:
            self.columns_ = list(X.columns)
            return self
        scores, _ = f_classif(X.to_numpy(dtype=float), y)
        scores = np.nan_to_num(scores, nan=0.0)
        order = np.argsort(-scores, kind="stable")
        corr = np.abs(np.corrcoef(X.rank().to_numpy(dtype=float), rowvar=False))
        corr = np.nan_to_num(corr, nan=0.0)
        kept = []
        for i in order:
            if all(corr[i, j] <= self.threshold for j in kept):
                kept.append(i)
        self.columns_ = [X.columns[i] for i in sorted(kept)]
        return self

    def transform(self, X):
        return X[self.columns_]


class ElasticNetSelector(BaseEstimator, TransformerMixin):
    """Keep the features an elastic-net logistic regression gives non-zero weight,
    at most ``k_max`` (largest |coef|). If it zeroes everything, fall back to the
    ``fallback_k`` features with the largest ANOVA F."""

    def __init__(self, C=0.1, l1_ratio=0.5, k_max=15, fallback_k=5, seed=0):
        self.C = C
        self.l1_ratio = l1_ratio
        self.k_max = k_max
        self.fallback_k = fallback_k
        self.seed = seed

    def fit(self, X, y):
        model = make_enet_logreg(self.C, self.l1_ratio, self.seed)
        model.fit(X.to_numpy(dtype=float), y)
        weight = np.abs(model.coef_).max(axis=0)
        chosen = np.flatnonzero(weight > 1e-8)
        self.used_fallback_ = False
        if len(chosen) > self.k_max:
            chosen = chosen[np.argsort(-weight[chosen], kind="stable")[: self.k_max]]
        elif len(chosen) == 0:
            scores, _ = f_classif(X.to_numpy(dtype=float), y)
            chosen = np.argsort(-np.nan_to_num(scores, nan=0.0), kind="stable")[: self.fallback_k]
            self.used_fallback_ = True
        self.selected_ = [X.columns[i] for i in sorted(chosen)]
        return self

    def transform(self, X):
        return X[self.selected_]


class BlockPreprocessor(BaseEstimator, TransformerMixin):
    """Radiomic columns -> ``radiomic_pipe``; ``clin__*`` columns -> ``clinical_pipe``."""

    def __init__(self, radiomic_pipe=None, clinical_pipe=None, clinical_prefix="clin__"):
        self.radiomic_pipe = radiomic_pipe
        self.clinical_pipe = clinical_pipe
        self.clinical_prefix = clinical_prefix

    def _split(self, X):
        clinical = [c for c in X.columns if c.startswith(self.clinical_prefix)]
        radiomic = [c for c in X.columns if c not in set(clinical)]
        return radiomic, clinical

    def fit(self, X, y):
        radiomic, clinical = self._split(X)
        self.radiomic_columns_, self.clinical_columns_ = radiomic, clinical
        self.fit_index_ = list(X.index)             # lets tests prove only training rows were seen
        self.radiomic_ = clone(self.radiomic_pipe).fit(X[radiomic], y) if radiomic else None
        self.clinical_ = clone(self.clinical_pipe).fit(X[clinical], y) if clinical else None
        self.selected_features_ = list(self.transform(X.iloc[:1]).columns)
        return self

    def transform(self, X):
        parts = []
        if self.radiomic_ is not None:
            parts.append(self.radiomic_.transform(X[self.radiomic_columns_]))
        if self.clinical_ is not None:
            parts.append(self.clinical_.transform(X[self.clinical_columns_]))
        return pd.concat(parts, axis=1)


def build_preprocessor(sel_cfg, icc=None, seed=0):
    radiomic = Pipeline([
        ("missing", MissingRateFilter(sel_cfg["max_missing"])),
        ("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("constant", NearConstantFilter(sel_cfg["near_constant_variance"])),
        ("stability", StabilityFilter(icc, sel_cfg["min_icc"])),
        ("scale", StandardScaler().set_output(transform="pandas")),
        ("correlation", CorrelationFilter(sel_cfg["corr_threshold"])),
        ("enet", ElasticNetSelector(
            C=sel_cfg["enet"]["C"], l1_ratio=sel_cfg["enet"]["l1_ratio"],
            k_max=sel_cfg["k_max"], fallback_k=sel_cfg["fallback_k"], seed=seed)),
    ])
    clinical = Pipeline([
        ("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("scale", StandardScaler().set_output(transform="pandas")),
    ])
    return BlockPreprocessor(radiomic, clinical)
