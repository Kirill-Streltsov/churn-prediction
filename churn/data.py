"""Load and clean the raw Telco Customer Churn CSV.

The raw file has three quirks worth handling explicitly:

* ``TotalCharges`` is stored as text and contains blank strings for the 11
  brand-new customers whose ``tenure`` is 0 -> these become genuine missing
  values (feature engineering later adds a flag for them).
* ``Churn`` is ``Yes`` / ``No`` -> mapped to 1 / 0.
* ``SeniorCitizen`` is 0 / 1 while every other binary column is Yes / No -> we
  normalise it to Yes / No so all categoricals look the same.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn import config


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the raw CSV exactly as shipped, no transformations."""
    return pd.read_csv(path or config.DATA_RAW)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a typed copy of the raw frame ready for feature engineering."""
    df = df.copy()

    # Blank strings -> NaN, then a proper numeric dtype.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True), errors="coerce"
    )

    # Make SeniorCitizen look like the other yes/no flags.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # Target -> integer.
    df[config.TARGET] = df[config.TARGET].map({"No": 0, "Yes": 1}).astype(int)

    return df


def load_clean(path: str | Path | None = None) -> pd.DataFrame:
    """Convenience wrapper: load the raw file and clean it in one call."""
    return clean(load_raw(path))
