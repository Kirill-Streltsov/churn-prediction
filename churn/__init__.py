"""Churn prediction on the Telco Customer Churn dataset.

The package is intentionally small and modular so the same code powers the
notebooks, the test-suite and the Streamlit demo:

    config    -> paths, seed and column groups
    data      -> load and clean the raw CSV
    features  -> feature engineering + preprocessing pipeline
    model     -> build / cross-validate / tune the three classifiers
    explain   -> SHAP values for the tuned XGBoost model

Import the submodules you need directly, e.g. ``from churn import data, model``.
"""

__version__ = "0.1.0"
