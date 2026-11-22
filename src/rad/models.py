"""The classifiers of the comparison (report 4.7) and a prior-only baseline.

Logistic regression and a linear SVM so far; random forest and XGBoost are added one per PR.

``build_model`` returns ``(estimator, grid)``. Grids are small on purpose: with a
few hundred patients a big search overfits the inner loop.
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

from .selection import make_enet_logreg

MODEL_NAMES = ("logreg", "svm")


def build_model(name, seed=0):
    if name == "logreg":
        return make_enet_logreg(C=0.1, l1_ratio=0.5, seed=seed), {
            "C": [0.01, 0.1, 1.0],
            "l1_ratio": [0.2, 0.5, 0.8],
        }
    if name == "svm":
        # SVC(probability=True) is deprecated (scikit-learn 1.9); calibrate explicitly instead.
        base = SVC(kernel="linear", class_weight="balanced", random_state=seed)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3, ensemble=False), {
            "estimator__C": [0.01, 0.1, 1.0],
        }
    if name == "prior":
        return DummyClassifier(strategy="prior"), {}
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
