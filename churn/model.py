"""Model building, cross-validation, threshold tuning and evaluation.

Three classifiers are compared on the same engineered features:

* **Logistic Regression** - a transparent linear baseline.
* **Random Forest** - a bagged tree ensemble.
* **XGBoost** - gradient-boosted trees, the strongest model here.

All three handle the ~27 % / 73 % class imbalance (``class_weight`` /
``scale_pos_weight``) and are wrapped in a single sklearn ``Pipeline`` together
with their preprocessing so nothing leaks between train and test.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn import config, features

# Class imbalance of the Telco dataset (negatives / positives).
_SCALE_POS_WEIGHT = 73.5 / 26.5


def build_models() -> dict[str, Pipeline]:
    """Return the three model pipelines keyed by display name."""
    return {
        "Logistic Regression": Pipeline(
            [
                ("prep", features.build_preprocessor(scale_numeric=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("prep", features.build_preprocessor(scale_numeric=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=8,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=config.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("prep", features.build_preprocessor(scale_numeric=False)),
                (
                    "clf",
                    XGBClassifier(
                        # Tuned with RandomizedSearchCV (5-fold ROC-AUC) on the
                        # training split - see scripts/tune_xgboost.py.
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.02,
                        subsample=0.85,
                        colsample_bytree=0.9,
                        min_child_weight=3,
                        reg_lambda=4.0,
                        gamma=0.1,
                        scale_pos_weight=_SCALE_POS_WEIGHT,
                        eval_metric="auc",
                        random_state=config.RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def make_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on the engineered frame."""
    X, y = features.split_xy(df)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )


def cross_validate_models(
    models: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> pd.DataFrame:
    """5-fold stratified CV, returning mean +/- std for the key metrics."""
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE
    )
    scoring = ["roc_auc", "f1", "precision", "recall"]
    rows = []
    for name, model in models.items():
        # Threading backend (vs the default loky) avoids the noisy semaphore
        # resource-tracker warnings on macOS; native training releases the GIL.
        with joblib.parallel_backend("threading"):
            scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def oof_proba(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> np.ndarray:
    """Out-of-fold churn probabilities, used to tune the threshold without
    ever touching the held-out test set."""
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE
    )
    return cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]


@dataclass
class ThresholdResult:
    """Best decision threshold plus the precision/recall curve behind it."""

    threshold: float
    precision: float
    recall: float
    f1: float
    curve: pd.DataFrame


def tune_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> ThresholdResult:
    """Pick the probability threshold that maximises F1 on the churn class.

    The default 0.5 cut-off is rarely optimal on an imbalanced target. Here the
    class weights already push the scores up, so the F1-optimal cut-off lands
    above 0.5: tuning gives up a little recall for noticeably better precision,
    i.e. fewer retention offers wasted on customers who would have stayed.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns one extra point with no threshold.
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best = int(np.argmax(f1))
    curve = pd.DataFrame(
        {"threshold": thresholds, "precision": precision, "recall": recall, "f1": f1}
    )
    return ThresholdResult(
        threshold=float(thresholds[best]),
        precision=float(precision[best]),
        recall=float(recall[best]),
        f1=float(f1[best]),
        curve=curve,
    )


def evaluate(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Score a set of predicted probabilities at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": threshold,
    }
