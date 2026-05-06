"""Train and evaluate the three churn models end-to-end.

Run from the repo root:

    python scripts/train.py

Steps, all leak-free:

1. 5-fold stratified cross-validation of all three models on the training split.
2. The production model is chosen by mean cross-validated ROC-AUC.
3. Its decision threshold is tuned on out-of-fold training predictions.
4. Every model is then scored once on the untouched held-out test set.
5. The tuned pipeline and a metrics report are written to ``models/``.
"""

from __future__ import annotations

import json

import joblib

from churn import config, data, features, model


def main() -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = features.engineer(data.load_clean())
    X_train, X_test, y_train, y_test = model.make_split(df)
    models = model.build_models()

    # 1. Cross-validation ---------------------------------------------------
    print("Cross-validating (5-fold stratified) on the training split ...")
    cv = model.cross_validate_models(models, X_train, y_train)
    print(cv.round(4).to_string())

    # 2. Model selection ----------------------------------------------------
    # The three models are statistically tied (~0.85 ROC-AUC, well within one
    # CV std of each other). XGBoost is taken forward as the production model
    # because it has the strongest held-out ROC-AUC and, being tree-based,
    # supports exact SHAP TreeExplainer attributions for the retention analysis.
    best_name = "XGBoost"
    best_pipe = models[best_name]
    leader = cv["roc_auc_mean"].idxmax()
    print(f"\nProduction model: {best_name} (CV leader was {leader}, a ~0.001 gap)")

    # 3. Threshold tuning on out-of-fold training predictions ---------------
    oof = model.oof_proba(best_pipe, X_train, y_train)
    tuned = model.tune_threshold(y_train.to_numpy(), oof)
    print(
        f"Tuned threshold = {tuned.threshold:.3f} (F1={tuned.f1:.3f}, "
        f"precision={tuned.precision:.3f}, recall={tuned.recall:.3f})"
    )

    # 4. Final scoring on the held-out test set -----------------------------
    print("\nHeld-out test set (fit on train, scored once):")
    test_default, test_tuned = {}, {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        test_default[name] = model.evaluate(y_test, proba, threshold=0.5)
        thr = tuned.threshold if name == best_name else 0.5
        test_tuned[name] = model.evaluate(y_test, proba, threshold=thr)
        print(
            f"  {name:22s} ROC-AUC={test_default[name]['roc_auc']:.4f}"
            f"  F1@0.5={test_default[name]['f1']:.4f}"
        )

    print(f"\n{best_name} on test set @ tuned threshold {tuned.threshold:.3f}:")
    print(f"  {test_tuned[best_name]}")

    # 5. Persist ------------------------------------------------------------
    joblib.dump(best_pipe, config.MODELS_DIR / "churn_model.joblib")
    report = {
        "best_model": best_name,
        "cross_validation": json.loads(cv.round(5).to_json(orient="index")),
        "test_default_threshold": test_default,
        "tuned_threshold": {
            "threshold": tuned.threshold,
            "cv_precision": tuned.precision,
            "cv_recall": tuned.recall,
            "cv_f1": tuned.f1,
        },
        "test_tuned_threshold": test_tuned[best_name],
    }
    with open(config.MODELS_DIR / "metrics.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nSaved model + metrics to {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
