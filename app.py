import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import json
from features import engineer_features

# Load the calibrated model (this is the one we use for real decisions)
model = joblib.load("models/xgb_calibrated_model.pkl")

# Load the raw model too — SHAP explanations typically need the raw model,
# not the calibrated wrapper, since CalibratedClassifierCV wraps the estimator
raw_model = joblib.load("models/xgb_final_model.pkl")

# Load metadata (for threshold, feature list, etc.)
with open("models/model_metadata.json", "r") as f:
    metadata = json.load(f)

st.title("Credit Risk Assessment")
st.write("Models loaded successfully.")
st.write("Metadata keys:", list(metadata.keys()))

st.divider()
st.header("Applicant Details")

with st.form("applicant_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        monthly_income = st.number_input("Monthly Income ($)", min_value=0, value=5000, step=100)
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=0)
        revolving_util = st.number_input(
            "Revolving Utilization of Unsecured Lines (0-1+ ratio)",
            min_value=0.0, value=0.30, step=0.01, format="%.2f"
        )
        debt_ratio = st.number_input(
            "Debt Ratio", min_value=0.0, value=0.30, step=0.01, format="%.2f"
        )
        open_credit_lines = st.number_input(
            "Number of Open Credit Lines and Loans", min_value=0, value=5
        )

    with col2:
        real_estate_loans = st.number_input(
            "Number of Real Estate Loans or Lines", min_value=0, value=1
        )
        past_due_30_59 = st.number_input(
            "Times 30-59 Days Past Due (not worse)", min_value=0, value=0
        )
        past_due_60_89 = st.number_input(
            "Times 60-89 Days Past Due (not worse)", min_value=0, value=0
        )
        past_due_90 = st.number_input(
            "Times 90+ Days Past Due", min_value=0, value=0
        )

    submitted = st.form_submit_button("Assess Risk")

if submitted:
    features = engineer_features(
        age, monthly_income, dependents, revolving_util, debt_ratio,
        open_credit_lines, real_estate_loans,
        past_due_30_59, past_due_60_89, past_due_90
    )

    probability = model.predict_proba(features)[0][1]
    threshold = metadata["cost_based_threshold_analysis"]["primary_recommendation"]["optimal_threshold"]
    decision = "DENY" if probability >= threshold else "APPROVE"

    st.divider()
    st.subheader("Result")
    st.metric("Predicted Risk Probability", f"{probability:.4f}")
    st.write(f"Cost-based threshold: {threshold}")

    if decision == "APPROVE":
        st.success(f"Recommendation: {decision}")
    else:
        st.error(f"Recommendation: {decision}")