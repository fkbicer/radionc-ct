"""The classifiers of the comparison (report 4.7) and a prior-only baseline.

Logistic regression, a linear SVM, a random forest and XGBoost.

``build_model`` returns ``(estimator, grid)``. Grids are small on purpose: with a
few hundred patients a big search overfits the inner loop.
"""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from .selection import make_enet_logreg

MODEL_NAMES = ("logreg", "svm", "rf", "xgboost")


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
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", n_jobs=1, random_state=seed
        ), {"max_depth": [3, 5], "min_samples_leaf": [3, 8]}
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("xgboost is required for the 'xgboost' model (pip install xgboost)") from exc
        return XGBClassifier(
            n_estimators=200, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
            eval_metric="logloss", tree_method="hist", n_jobs=1, random_state=seed, verbosity=0,
        ), {"max_depth": [2, 3], "learning_rate": [0.03, 0.1]}
    if name == "prior":
        return DummyClassifier(strategy="prior"), {}
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
