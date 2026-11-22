"""The classifiers of the comparison (report 4.7) and a prior-only baseline.

Logistic regression comes first; SVM, random forest and XGBoost are added one per PR.

``build_model`` returns ``(estimator, grid)``. Grids are small on purpose: with a
few hundred patients a big search overfits the inner loop.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier

from .selection import make_enet_logreg

MODEL_NAMES = ("logreg",)


def build_model(name, seed=0):
    if name == "logreg":
        return make_enet_logreg(C=0.1, l1_ratio=0.5, seed=seed), {
            "C": [0.01, 0.1, 1.0],
            "l1_ratio": [0.2, 0.5, 0.8],
        }
    if name == "prior":
        return DummyClassifier(strategy="prior"), {}
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
