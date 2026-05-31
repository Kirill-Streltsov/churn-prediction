"""Tests for the SHAP explainability helpers."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

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


def test_waterfall_output_axis_is_hidden(fitted_xgb):
    # Regression test for the garbled f(x) label bug: after the fix, no *visible*
    # axis should carry SHAP's duplicated output-value tick labels.
    pipe, X_test, _ = fitted_xgb
    exp = explain.shap_explanation(pipe, X_test.iloc[[0]])
    shap.plots.waterfall(exp[0], show=False)
    fig = plt.gcf()
    explain.hide_waterfall_output_axis(fig)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert not (
            len(labels) == 2 and labels[0] == labels[1] and labels[0].startswith("$ =")
        )
    plt.close(fig)
