"""Model explanation (report 4.9, FR11).

The performance numbers come from nested CV; this module refits ONE model on all
patients (inner CV picks its hyperparameters) only to read off what it uses.
Linear models: coefficients. Trees: SHAP if installed, impurity importance otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .clinical import columns_for
from .models import build_model
from .selection import build_preprocessor


def fit_for_explanation(X, y, model_name, cfg, icc=None):
    y = np.asarray(y)
    seed = cfg["seed"]
    estimator, grid = build_model(model_name, seed)
    pipe = Pipeline([("prep", build_preprocessor(cfg["selection"], icc, seed)), ("model", estimator)])
    search = GridSearchCV(
        pipe, {f"model__{k}": vals for k, vals in grid.items()}, scoring="roc_auc", refit=True,
        cv=StratifiedKFold(cfg["validation"]["inner_folds"], shuffle=True, random_state=seed), n_jobs=1,
    )
    search.fit(X, y)
    return search.best_estimator_


def _linear_weights(model):
    if hasattr(model, "coef_"):
        return np.ravel(model.coef_)
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        inner = calibrated[0].estimator
        if hasattr(inner, "coef_"):
            return np.ravel(inner.coef_)
    return None


def feature_importance(best, X):
    """DataFrame(feature, importance, kind) for a fitted prep+model pipeline."""
    transformed = best.named_steps["prep"].transform(X)
    model = best.named_steps["model"]
    names = list(transformed.columns)

    weights = _linear_weights(model)
    if weights is not None:
        return pd.DataFrame({"feature": names, "importance": weights, "kind": "coefficient"})

    try:
        import shap

        values = shap.TreeExplainer(model).shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[..., -1]
        return pd.DataFrame({"feature": names, "importance": np.abs(values).mean(axis=0), "kind": "mean|SHAP|"})
    except ImportError:
        pass
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({"feature": names, "importance": model.feature_importances_, "kind": "impurity"})
    raise ValueError(f"Cannot explain {type(model).__name__}")


def explain(cfg, X, y, icc, set_name, model_name, out_dir):
    spec = cfg["ladder"][set_name]
    Xs = X[columns_for(X, spec["groups"])]
    best = fit_for_explanation(Xs, y, model_name, cfg, icc)
    table = feature_importance(best, Xs)
    table = table.reindex(table["importance"].abs().sort_values(ascending=False).index)
    path = out_dir / f"explain_{set_name}_{model_name}.csv"
    table.to_csv(path, index=False)
    return table, path
