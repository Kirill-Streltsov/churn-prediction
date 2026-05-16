"""Tests for feature engineering and the preprocessing pipeline."""

import numpy as np

from churn import features


def test_engineered_columns_exist(engineered_df):
    for col in [
        "TotalCharges_missing",
        "tenure_group",
        "avg_charges_per_month",
        "monthly_vs_lifetime",
    ]:
        assert col in engineered_df.columns


def test_missing_flag_matches_raw_blanks(engineered_df):
    # 11 customers had a blank TotalCharges in the raw file.
    assert engineered_df["TotalCharges_missing"].sum() == 11
    # After flagging, the column itself is imputed to 0 (no NaN left).
    assert engineered_df["TotalCharges"].isna().sum() == 0


def test_feature_matrix_has_no_missing(engineered_df):
    X = engineered_df[features.FEATURE_COLUMNS]
    assert not X.isna().any().any()


def test_tenure_group_labels(engineered_df):
    assert set(engineered_df["tenure_group"].unique()) <= set(features.TENURE_LABELS)


def test_charge_ratios_are_finite(engineered_df):
    for col in ["avg_charges_per_month", "monthly_vs_lifetime"]:
        assert np.isfinite(engineered_df[col]).all()


def test_preprocessor_produces_dense_numeric_matrix(engineered_df):
    prep = features.build_preprocessor(scale_numeric=False)
    X = engineered_df[features.FEATURE_COLUMNS]
    encoded = prep.fit_transform(X)
    assert encoded.shape[0] == len(engineered_df)
    # More columns than raw features because of one-hot encoding.
    assert encoded.shape[1] > len(features.FEATURE_COLUMNS)
