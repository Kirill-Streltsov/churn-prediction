"""Reproduce the XGBoost hyperparameter search.

    python scripts/tune_xgboost.py

A 40-iteration randomized search (5-fold ROC-AUC) over the XGBoost
hyperparameters. The winning configuration is what ``churn.model.build_models``
hard-codes for the ``XGBoost`` pipeline, so results stay reproducible without
re-running the (slower) search every time.
"""

from __future__ import annotations

from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from churn import config, data, features, model


def main() -> None:
    df = features.engineer(data.load_clean())
    X_train, _, y_train, _ = model.make_split(df)

    prep = features.build_preprocessor(scale_numeric=False)
    X = prep.fit_transform(X_train)

    spw = (y_train == 0).sum() / (y_train == 1).sum()
    param_dist = {
        "n_estimators": randint(200, 900),
        "max_depth": randint(2, 6),
        "learning_rate": uniform(0.01, 0.09),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 12),
        "reg_lambda": uniform(0.0, 5.0),
        "gamma": uniform(0.0, 0.5),
    }
    search = RandomizedSearchCV(
        XGBClassifier(
            eval_metric="auc", scale_pos_weight=spw, random_state=config.RANDOM_STATE
        ),
        param_dist,
        n_iter=40,
        scoring="roc_auc",
        cv=StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE),
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X, y_train)

    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print("Best params:")
    for key, value in sorted(search.best_params_.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
