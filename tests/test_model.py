"""Tests for model building, cross-validation, threshold tuning and scoring."""

import numpy as np
from sklearn.metrics import roc_auc_score

from churn import model


def test_build_models_returns_three_pipelines():
    models = model.build_models()
    assert set(models) == {"Logistic Regression", "Random Forest", "XGBoost"}


def test_pipeline_predicts_valid_probabilities(fitted_xgb):
    pipe, X_test, _ = fitted_xgb
    proba = pipe.predict_proba(X_test)[:, 1]
    assert proba.shape[0] == len(X_test)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_cross_validation_is_reasonable(split):
    X_train, _, y_train, _ = split
    # A quick 3-fold pass; the dataset is small enough for this to be fast.
    cv = model.cross_validate_models(model.build_models(), X_train, y_train, n_splits=3)
    assert cv.loc["XGBoost", "roc_auc_mean"] > 0.80


def test_threshold_tuning_returns_valid_threshold(fitted_xgb):
    pipe, X_test, y_test = fitted_xgb
    proba = pipe.predict_proba(X_test)[:, 1]
    tuned = model.tune_threshold(y_test.to_numpy(), proba)
    assert 0.0 < tuned.threshold < 1.0
    assert tuned.f1 > 0.0
    # The tuned threshold should not score worse on F1 than the default 0.5.
    default_f1 = model.evaluate(y_test, proba, threshold=0.5)["f1"]
    tuned_f1 = model.evaluate(y_test, proba, threshold=tuned.threshold)["f1"]
    assert tuned_f1 + 1e-9 >= default_f1


def test_evaluate_reports_all_metrics(fitted_xgb):
    pipe, X_test, y_test = fitted_xgb
    proba = pipe.predict_proba(X_test)[:, 1]
    scores = model.evaluate(y_test, proba, threshold=0.5)
    assert set(scores) == {"roc_auc", "precision", "recall", "f1", "threshold"}
    assert all(np.isfinite(v) for v in scores.values())


def test_calibrator_matches_churn_rate_and_keeps_ranking(split):
    X_train, _, y_train, _ = split
    pipe = model.build_models()["XGBoost"]
    oof = model.oof_proba(pipe, X_train, y_train, n_splits=3)
    calibrator = model.fit_calibrator(y_train.to_numpy(), oof)
    calibrated = calibrator.transform(oof)
    # The class weights inflate the raw scores; calibration brings the average
    # back to the real churn rate...
    assert oof.mean() > y_train.mean() + 0.05
    assert abs(calibrated.mean() - y_train.mean()) < 0.01
    # ...without changing the order of customers (so ROC-AUC is unchanged).
    order = np.argsort(oof)
    assert (np.diff(calibrated[order]) >= 0).all()
    assert roc_auc_score(y_train, calibrated) == roc_auc_score(y_train, oof)
