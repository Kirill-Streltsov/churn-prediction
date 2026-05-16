"""Tests for the SHAP explainability helpers."""

from churn import explain


def test_shap_explanation_shape_matches_test_set(fitted_xgb):
    pipe, X_test, _ = fitted_xgb
    exp = explain.shap_explanation(pipe, X_test)
    assert exp.values.shape[0] == len(X_test)
    # Feature-name prefixes should be stripped for readability.
    assert not any(name.startswith(("num__", "cat__")) for name in exp.feature_names)


def test_contract_and_tenure_are_top_drivers(fitted_xgb):
    pipe, X_test, _ = fitted_xgb
    grouped = explain.grouped_importance(explain.shap_explanation(pipe, X_test))
    top_two = list(grouped.head(2).index)
    assert top_two[0] == "Contract"
    assert "tenure" in top_two
