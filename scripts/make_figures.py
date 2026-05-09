"""Render the report figures used in the README and notebooks.

    python scripts/make_figures.py

Writes PNGs to ``reports/figures/``. Everything is derived from the same
``churn`` package the app and tests use, so the figures never drift from the
code.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from churn import config, data, explain, features, model

sns.set_theme(style="whitegrid", palette="deep")
config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CHURN = config.TARGET


def _rate_plot(df, column, filename, title, order=None):
    rate = df.groupby(column, observed=True)[CHURN].mean().sort_values()
    if order is not None:
        rate = rate.reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=rate.values * 100, y=rate.index, ax=ax, color="#d1495b")
    ax.axvline(df[CHURN].mean() * 100, ls="--", c="gray", label="overall churn")
    ax.set_xlabel("Churn rate (%)")
    ax.set_ylabel("")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=120)
    plt.close(fig)


def main() -> None:
    df = features.engineer(data.load_clean())

    # --- EDA -----------------------------------------------------------------
    _rate_plot(df, "Contract", "churn_by_contract.png", "Churn rate by contract type")
    _rate_plot(
        df,
        "tenure_group",
        "churn_by_tenure.png",
        "Churn rate by tenure group",
        order=features.TENURE_LABELS,
    )
    _rate_plot(
        df, "PaymentMethod", "churn_by_payment.png", "Churn rate by payment method"
    )

    # --- model comparison ----------------------------------------------------
    X_train, X_test, y_train, y_test = model.make_split(df)
    models = model.build_models()
    cv = model.cross_validate_models(models, X_train, y_train)
    fig, ax = plt.subplots(figsize=(7, 4))
    cv[["roc_auc_mean", "f1_mean"]].plot.bar(ax=ax, color=["#0b6e4f", "#e8985e"], rot=0)
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("score (5-fold CV mean)")
    ax.set_title("Model comparison")
    ax.legend(["ROC-AUC", "F1 (churn class)"])
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "model_comparison.png", dpi=120)
    plt.close(fig)

    # --- precision/recall + tuned threshold ----------------------------------
    xgb = models["XGBoost"]
    oof = model.oof_proba(xgb, X_train, y_train)
    tuned = model.tune_threshold(y_train.to_numpy(), oof)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tuned.curve["recall"], tuned.curve["precision"], color="#0b6e4f")
    ax.scatter(
        [tuned.recall],
        [tuned.precision],
        color="#d1495b",
        zorder=5,
        label=f"tuned threshold = {tuned.threshold:.2f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("XGBoost precision/recall trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "pr_curve.png", dpi=120)
    plt.close(fig)

    # --- SHAP ----------------------------------------------------------------
    xgb.fit(X_train, y_train)
    exp = explain.shap_explanation(xgb, X_test)

    import shap

    fig = plt.figure()
    shap.plots.beeswarm(exp, max_display=12, show=False)
    plt.title("SHAP summary - what drives churn")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    grouped = explain.grouped_importance(exp).head(10)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=grouped.values, y=grouped.index, ax=ax, color="#0b6e4f")
    ax.set_xlabel("mean |SHAP value| (summed over one-hot columns)")
    ax.set_ylabel("")
    ax.set_title("Top churn drivers (grouped by feature)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "shap_importance.png", dpi=120)
    plt.close(fig)

    print(f"Wrote figures to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
