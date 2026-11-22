"""Repeated, stratified, nested cross-validation and basic metrics (report 4.8).

The outer loop only estimates performance. Inside each outer training set an
inner loop picks hyperparameters; scaling, imputation, feature selection and the
model are all re-fitted there. Splitting is by patient (one row per patient).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from .models import build_model
from .selection import build_preprocessor


@dataclass
class CVResult:
    oof: np.ndarray          # (repeats, n) out-of-fold probability of class 1
    folds: list              # one dict per outer fold


def _fit_fold(X, y, model_name, cfg, icc, seed, repeat, fold, train, test):
    v = cfg["validation"]
    estimator, grid = build_model(model_name, seed)
    steps = [("model", estimator)]
    if model_name != "prior":
        steps.insert(0, ("prep", build_preprocessor(cfg["selection"], icc, seed)))
    pipe = Pipeline(steps)

    y_train = y[train]
    inner_k = int(min(v["inner_folds"], np.bincount(y_train).min()))
    if grid and inner_k >= 2:
        search = GridSearchCV(
            pipe,
            {f"model__{k}": values for k, values in grid.items()},
            scoring="roc_auc",
            cv=StratifiedKFold(inner_k, shuffle=True, random_state=seed + 1000 + repeat * 100 + fold),
            refit=True,
            n_jobs=1,
            error_score="raise",
        )
        search.fit(X.iloc[train], y_train)
        best, params = search.best_estimator_, search.best_params_
    else:
        best, params = pipe.fit(X.iloc[train], y_train), {}

    proba = best.predict_proba(X.iloc[test])[:, 1]
    prep = best.named_steps.get("prep")
    return {
        "repeat": repeat, "fold": fold, "test": test, "proba": proba,
        "selected": list(prep.selected_features_) if prep is not None else [],
        "fit_index": list(prep.fit_index_) if prep is not None else list(X.index[train]),
        "test_index": list(X.index[test]),
        "params": params,
    }


def run_nested_cv(X, y, model_name, cfg, icc=None, seed=None, repeats=None):
    """Out-of-fold probabilities for every repeat, plus per-fold bookkeeping."""
    v = cfg["validation"]
    seed = cfg["seed"] if seed is None else seed
    repeats = v["repeats"] if repeats is None else repeats
    y = np.asarray(y)
    splits = []
    for r in range(repeats):
        outer = StratifiedKFold(v["outer_folds"], shuffle=True, random_state=seed + r)
        for f, (train, test) in enumerate(outer.split(np.zeros(len(y)), y)):
            splits.append((r, f, train, test))
    results = Parallel(n_jobs=v["n_jobs"])(
        delayed(_fit_fold)(X, y, model_name, cfg, icc, seed, r, f, tr, te) for r, f, tr, te in splits
    )
    oof = np.full((repeats, len(y)), np.nan)
    for out in results:
        oof[out["repeat"], out["test"]] = out["proba"]
    return CVResult(oof=oof, folds=results)


# --- metrics ----------------------------------------------------------------

def expected_calibration_error(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def compute_metrics(y, p, threshold=0.5):
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "auc": float(roc_auc_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, average="macro")),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
    }


def summarize(y, oof):
    """Mean and sd over repeats of each metric, plus the pooled (repeat-averaged) predictions."""
    per_repeat = [compute_metrics(y, row) for row in oof]
    summary = {}
    for key in per_repeat[0]:
        values = np.array([m[key] for m in per_repeat], dtype=float)
        summary[key] = float(np.nanmean(values))
        summary[f"{key}_sd"] = float(np.nanstd(values))
    return summary, np.nanmean(oof, axis=0)
