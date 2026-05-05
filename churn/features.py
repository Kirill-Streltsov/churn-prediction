"""Feature engineering and the preprocessing pipeline.

Three families of engineered features, matching the project write-up:

* **Contract-duration categories** - ``tenure`` bucketed into lifecycle stages
  (new customers churn very differently from long-tenured ones).
* **Charge ratios** - how a customer's spend relates to their tenure, which
  separates high-value loyal customers from expensive newcomers.
* **Missing-value flags** - a binary column marking the rows where
  ``TotalCharges`` was blank in the raw data.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config

# Buckets for the engineered ``tenure_group`` feature (months).
TENURE_BINS = [-0.1, 12, 24, 48, 60, 72]
TENURE_LABELS = ["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"]

# Column groups *after* engineering, used to wire up the ColumnTransformer.
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_charges_per_month",
    "monthly_vs_lifetime",
    "TotalCharges_missing",
]
CATEGORICAL_FEATURES = [*config.RAW_CATEGORICAL, "tenure_group"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features and impute the flagged missing values."""
    df = df.copy()

    # Missing-value flag has to be captured *before* we impute.
    df["TotalCharges_missing"] = df["TotalCharges"].isna().astype(int)
    # A blank TotalCharges only happens at tenure 0, so 0 is the right fill.
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Contract-duration categories.
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
    ).astype(str)

    # Charge ratios. tenure 0 -> fall back to the monthly charge so the ratio
    # stays finite and meaningful for brand-new customers.
    #
    # avg_charges_per_month : lifetime average monthly spend.
    # monthly_vs_lifetime   : how the *current* monthly charge compares to that
    #                         average - a value >1 flags a recent price rise,
    #                         which is a churn signal in its own right.
    df["avg_charges_per_month"] = (df["TotalCharges"] / df["tenure"]).where(
        df["tenure"] > 0, df["MonthlyCharges"]
    )
    df["monthly_vs_lifetime"] = df["MonthlyCharges"] / df[
        "avg_charges_per_month"
    ].replace(0.0, 1.0)

    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split an engineered frame into the feature matrix X and target y."""
    return df[FEATURE_COLUMNS].copy(), df[config.TARGET].copy()


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """One-hot encode categoricals; optionally standard-scale numerics.

    Scaling only matters for the linear model (logistic regression), so the
    tree ensembles pass their numeric columns straight through.
    """
    numeric = (
        Pipeline([("scale", StandardScaler())]) if scale_numeric else "passthrough"
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
