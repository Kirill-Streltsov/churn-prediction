"""Interactive churn-prediction demo (Streamlit).

Run locally:

    streamlit run app.py

Four tabs: score a single customer (with a personal SHAP explanation and a
retention recommendation), explore the data, compare the models, and read the
global explainability story. The production XGBoost model is trained once and
cached, so the app is responsive after the first load.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from churn import config, data, explain, features, model

st.set_page_config(page_title="Telco Churn Prediction", page_icon="📉", layout="wide")

PRIMARY = "#0b6e4f"
DANGER = "#d1495b"


# --------------------------------------------------------------------------- #
# Cached data / model
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_frame() -> pd.DataFrame:
    return features.engineer(data.load_clean())


@st.cache_resource(show_spinner="Training the model (one-off) ...")
def load_bundle():
    df = load_frame()
    X_train, X_test, y_train, y_test = model.make_split(df)
    pipe = model.build_models()["XGBoost"]
    pipe.fit(X_train, y_train)

    oof = model.oof_proba(pipe, X_train, y_train)
    tuned = model.tune_threshold(y_train.to_numpy(), oof)

    proba_test = pipe.predict_proba(X_test)[:, 1]
    test_metrics = model.evaluate(y_test, proba_test, threshold=tuned.threshold)
    explanation = explain.shap_explanation(pipe, X_test)
    return {
        "pipe": pipe,
        "threshold": tuned.threshold,
        "tuned": tuned,
        "test_metrics": test_metrics,
        "explanation": explanation,
        "X_test": X_test,
    }


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    path = config.MODELS_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("📉 Telco Customer Churn Prediction")
st.markdown(
    "Predicts which telecom customers are likely to cancel, and shows why using "
    "SHAP. The model is a tuned XGBoost trained on the Telco dataset "
    "(7,043 customers)."
)

frame = load_frame()
bundle = load_bundle()
metrics = load_metrics()

roc = bundle["test_metrics"]["roc_auc"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{len(frame):,}")
c2.metric("Overall churn rate", f"{frame[config.TARGET].mean() * 100:.1f}%")
c3.metric("Test ROC-AUC", f"{roc:.3f}")
c4.metric("Decision threshold", f"{bundle['threshold']:.2f}")

tab_predict, tab_eda, tab_models, tab_explain = st.tabs(
    ["🔮 Predict", "📊 Data & EDA", "🤖 Models", "🧠 Explainability"]
)


# --------------------------------------------------------------------------- #
# Predict tab
# --------------------------------------------------------------------------- #
def choices(column: str) -> list[str]:
    return sorted(frame[column].dropna().unique().tolist())


with tab_predict:
    st.subheader("Score a single customer")
    st.caption(
        "Enter a customer profile. The model returns a churn probability and, "
        "if it clears the tuned threshold, flags them for a retention offer."
    )

    left, right = st.columns(2)
    with left:
        contract = st.selectbox("Contract", choices("Contract"))
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.slider("Monthly charges ($)", 15.0, 120.0, 70.0, step=0.5)
        internet = st.selectbox("Internet service", choices("InternetService"))
        payment = st.selectbox("Payment method", choices("PaymentMethod"))
    with right:
        online_security = st.selectbox("Online security", choices("OnlineSecurity"))
        tech_support = st.selectbox("Tech support", choices("TechSupport"))
        paperless = st.selectbox("Paperless billing", choices("PaperlessBilling"))
        senior = st.selectbox("Senior citizen", choices("SeniorCitizen"))
        dependents = st.selectbox("Dependents", choices("Dependents"))

    # Start from the dataset's most common profile, then override the inputs the
    # user actually set, so every feature the pipeline needs has a value.
    row = frame.drop(columns=[config.TARGET]).mode().iloc[0].to_dict()
    row.update(
        Contract=contract,
        tenure=tenure,
        MonthlyCharges=monthly,
        TotalCharges=monthly * tenure,
        InternetService=internet,
        PaymentMethod=payment,
        OnlineSecurity=online_security,
        TechSupport=tech_support,
        PaperlessBilling=paperless,
        SeniorCitizen=senior,
        Dependents=dependents,
    )
    x = features.engineer(pd.DataFrame([row]))[features.FEATURE_COLUMNS]

    proba = float(bundle["pipe"].predict_proba(x)[:, 1][0])
    threshold = bundle["threshold"]
    will_churn = proba >= threshold

    st.divider()
    m1, m2 = st.columns([1, 2])
    with m1:
        st.metric("Churn probability", f"{proba * 100:.1f}%")
        if will_churn:
            st.error("High churn risk. Worth flagging for a retention offer.", icon="⚠️")
        else:
            st.success("Likely to stay.", icon="✅")
    with m2:
        st.progress(min(proba, 1.0))
        st.caption(
            f"Flagged when probability ≥ tuned threshold ({threshold:.2f}). "
            "The threshold was tuned on out-of-fold training data to balance "
            "precision and recall on the churn class."
        )

    st.markdown("**Why this prediction?**")
    single = explain.shap_explanation(bundle["pipe"], x)
    shap.plots.waterfall(single[0], max_display=10, show=False)
    fig = plt.gcf()
    explain.hide_waterfall_output_axis(fig)
    fig.set_size_inches(8, 4.2)
    st.pyplot(fig, clear_figure=True)

    # Simple rule-of-thumb retention suggestions tied to the biggest levers.
    tips = []
    if contract == "Month-to-month":
        tips.append("Offer an incentive to move to a one or two year contract.")
    if tenure <= 12:
        tips.append("New customer, so onboarding and regular check-ins help.")
    if payment == "Electronic check":
        tips.append("Nudge towards auto-pay (electronic-check users churn more).")
    if tech_support == "No" and internet != "No":
        tips.append(
            "Bundle tech support or online security to make switching less likely."
        )
    if will_churn and tips:
        st.markdown("**Some things to try:**")
        for tip in tips:
            st.markdown(f"- {tip}")


# --------------------------------------------------------------------------- #
# EDA tab
# --------------------------------------------------------------------------- #
with tab_eda:
    st.subheader("Who churns?")
    st.caption(
        "Churn concentrates in month-to-month contracts, short-tenured customers "
        "and electronic-check payers. Those are the same signals the model uses."
    )

    def rate_by(column: str, order=None) -> pd.Series:
        rate = frame.groupby(column, observed=True)[config.TARGET].mean() * 100
        return rate.reindex(order) if order else rate.sort_values()

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**Churn rate by contract type (%)**")
        st.bar_chart(rate_by("Contract"), color=DANGER, horizontal=True)
        st.markdown("**Churn rate by tenure group (%)**")
        st.bar_chart(rate_by("tenure_group", features.TENURE_LABELS), color=DANGER)
    with g2:
        st.markdown("**Churn rate by payment method (%)**")
        st.bar_chart(rate_by("PaymentMethod"), color=DANGER, horizontal=True)
        st.markdown("**Churn rate by internet service (%)**")
        st.bar_chart(rate_by("InternetService"), color=DANGER, horizontal=True)

    with st.expander("Peek at the data"):
        st.dataframe(frame.head(50), width="stretch")


# --------------------------------------------------------------------------- #
# Models tab
# --------------------------------------------------------------------------- #
with tab_models:
    st.subheader("Model comparison")
    st.caption(
        "Three models on the same features, compared with 5-fold cross-validation. "
        "They land close together (around 0.85 ROC-AUC). I ship XGBoost for its "
        "held-out score and because it works well with SHAP."
    )
    if metrics:
        cv = pd.DataFrame(metrics["cross_validation"]).T
        show = cv[["roc_auc_mean", "f1_mean", "precision_mean", "recall_mean"]]
        show.columns = ["ROC-AUC", "F1 (churn)", "Precision", "Recall"]
        st.dataframe(
            show.style.format("{:.3f}").highlight_max(axis=0, color="#123f2b"),
            width="stretch",
        )
        pr_path = config.FIGURES_DIR / "pr_curve.png"
        if pr_path.exists():
            st.image(str(pr_path), caption="XGBoost precision/recall trade-off")
    else:
        st.info("Run `python scripts/train.py` to generate the metrics report.")

    tm = bundle["test_metrics"]
    st.markdown("**XGBoost on the held-out test set, at the tuned threshold:**")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ROC-AUC", f"{tm['roc_auc']:.3f}")
    k2.metric("F1 (churn)", f"{tm['f1']:.3f}")
    k3.metric("Precision", f"{tm['precision']:.3f}")
    k4.metric("Recall", f"{tm['recall']:.3f}")


# --------------------------------------------------------------------------- #
# Explainability tab
# --------------------------------------------------------------------------- #
with tab_explain:
    st.subheader("What drives churn? (global SHAP)")
    grouped = explain.grouped_importance(bundle["explanation"]).head(10)
    st.bar_chart(grouped.sort_values(), color=PRIMARY, horizontal=True)

    with st.expander("SHAP beeswarm (per-feature detail)"):
        shap.plots.beeswarm(bundle["explanation"], max_display=12, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

    st.markdown(
        """
        **What the model keys on**

        - Contract type and tenure dominate: month-to-month customers and
          newcomers churn much more often.
        - Electronic-check payment and having no tech support or online security
          push the risk up.
        - High monthly charges on a short tenure flag a price-sensitive newcomer.

        **Ideas for retention**

        1. Move month-to-month customers onto annual contracts. This is the
           biggest single lever.
        2. Put effort into the first 12 months: onboarding, support, loyalty perks.
        3. Bundle tech support or online security to make switching less appealing.
        4. Move electronic-check payers to auto-pay to cut passive churn.
        """
    )

st.divider()
st.caption("Built with pandas, scikit-learn, XGBoost, SHAP and Streamlit.")
