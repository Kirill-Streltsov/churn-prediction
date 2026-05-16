"""Tests for loading and cleaning the raw CSV."""

import pandas as pd

from churn import config


def test_loads_expected_shape(clean_df):
    assert clean_df.shape[0] == 7043
    assert config.TARGET in clean_df.columns


def test_target_is_binary_int(clean_df):
    assert set(clean_df[config.TARGET].unique()) == {0, 1}
    assert pd.api.types.is_integer_dtype(clean_df[config.TARGET])


def test_totalcharges_is_numeric_with_missing(clean_df):
    assert pd.api.types.is_float_dtype(clean_df["TotalCharges"])
    # The 11 blank rows (tenure 0) should now be NaN.
    assert clean_df["TotalCharges"].isna().sum() == 11


def test_seniorcitizen_normalised_to_yes_no(clean_df):
    assert set(clean_df["SeniorCitizen"].unique()) == {"Yes", "No"}
