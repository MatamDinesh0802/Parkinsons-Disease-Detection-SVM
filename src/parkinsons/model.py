"""Model definitions — SVM headline + baselines."""
from __future__ import annotations

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from .config import RANDOM_STATE


def build_models() -> dict:
    return {
        "svm_linear": SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
        "svm_rbf": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
    }
