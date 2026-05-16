"""Shared, session-scoped fixtures so the (slightly heavier) model fixtures are
built only once across the whole test run."""

import pytest

from churn import data, features, model


@pytest.fixture(scope="session")
def clean_df():
    return data.load_clean()


@pytest.fixture(scope="session")
def engineered_df(clean_df):
    return features.engineer(clean_df)


@pytest.fixture(scope="session")
def split(engineered_df):
    return model.make_split(engineered_df)


@pytest.fixture(scope="session")
def fitted_xgb(split):
    X_train, X_test, y_train, y_test = split
    pipe = model.build_models()["XGBoost"]
    pipe.fit(X_train, y_train)
    return pipe, X_test, y_test
