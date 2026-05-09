"""SHAP explanations for the tuned XGBoost model.

TreeExplainer runs on the fitted booster after the pipeline's preprocessing,
so the SHAP feature names line up with the one-hot encoded columns the model
actually sees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def transformed_frame(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Run X through the pipeline's preprocessor and keep the feature names."""
    prep = pipeline.named_steps["prep"]
    matrix = prep.transform(X)
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return pd.DataFrame(matrix, columns=prep.get_feature_names_out(), index=X.index)


def shap_explanation(pipeline: Pipeline, X: pd.DataFrame) -> shap.Explanation:
    """Return a SHAP ``Explanation`` for the booster inside ``pipeline``.

    The ``num__`` / ``cat__`` ColumnTransformer prefixes are stripped from the
    feature names so the plots read in plain business terms.
    """
    transformed = transformed_frame(pipeline, X)
    explainer = shap.TreeExplainer(pipeline.named_steps["clf"])
    explanation = explainer(transformed)
    explanation.feature_names = [
        name.split("__", 1)[-1] for name in explanation.feature_names
    ]
    return explanation


def mean_abs_importance(explanation: shap.Explanation) -> pd.Series:
    """Global feature importance = mean absolute SHAP value per feature."""
    importance = np.abs(explanation.values).mean(axis=0)
    return pd.Series(importance, index=explanation.feature_names).sort_values(
        ascending=False
    )


def grouped_importance(explanation: shap.Explanation) -> pd.Series:
    """Collapse one-hot columns back to their original feature for reporting.

    ``cat__Contract_Two year`` and ``cat__Contract_One year`` both roll up into
    ``Contract`` so the headline importances read in business terms. Numeric
    features (which have no category suffix) are kept whole, so engineered
    columns like ``monthly_to_total_ratio`` are not truncated.
    """
    from churn import features

    raw = mean_abs_importance(explanation)
    groups: dict[str, float] = {}
    for name, value in raw.items():
        base = name.split("__", 1)[-1]  # drop the "num__"/"cat__" prefix
        group = base
        for cat in features.CATEGORICAL_FEATURES:
            if base == cat or base.startswith(f"{cat}_"):
                group = cat
                break
        groups[group] = groups.get(group, 0.0) + float(value)
    return pd.Series(groups).sort_values(ascending=False)
