"""Risk Alert Classifier - Streamlit app (run: streamlit run app.py)"""

import json
import pickle
import pandas as pd
import streamlit as st
from pathlib import Path

BASE = Path(__file__).parent

with open(BASE / "Models" / "risk_alert_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE / "Models" / "final_metrics.json", "r") as f:
    meta = json.load(f)

THR = meta.get("threshold", 0.5)

FEATURES = [
    "age", "gender", "region", "employment_type",
    "annual_income_inr", "debt_balance_inr", "credit_score",
    "credit_utilization_ratio", "missed_payments_12m",
    "avg_late_payment_days", "cash_advance_count_6m",
    "monthly_transaction_count", "monthly_spend_inr",
    "complaints_last_6m", "failed_login_attempts_3m",
    "account_tenure_months", "days_since_last_txn"
]

CATEGORICAL = ["gender", "region", "employment_type"]

st.set_page_config(page_title="Risk Alert Classifier", layout="wide")
st.title("Risk Alert Classifier")
st.write(f"Prediction threshold: {THR:.2f}")

st.subheader("Enter Customer Details")

with st.form("customer_form"):
    values = {}
    col1, col2, col3 = st.columns(3)

    with col1:
        values["age"] = st.number_input("Age", 18, 90, 35)
        values["gender"] = st.selectbox(
            "Gender", meta.get("cat_values", {}).get("gender", ["Male", "Female"])
        )
        values["region"] = st.selectbox(
            "Region", meta.get("cat_values", {}).get("region", ["Unknown"])
        )
        values["employment_type"] = st.selectbox(
            "Employment Type",
            meta.get("cat_values", {}).get("employment_type", ["Unknown"])
        )
        values["annual_income_inr"] = st.number_input(
            "Annual Income (INR)", min_value=0, value=42000
        )
        values["debt_balance_inr"] = st.number_input(
            "Debt Balance (INR)", min_value=0, value=30000
        )

    with col2:
        values["credit_score"] = st.number_input(
            "Credit Score", 300, 850, 680
        )
        values["credit_utilization_ratio"] = st.slider(
            "Credit Utilization Ratio", 0.0, 1.0, 0.35
        )
        values["missed_payments_12m"] = st.number_input(
            "Missed Payments (12 Months)", 0, 20, 1
        )
        values["avg_late_payment_days"] = st.number_input(
            "Average Late Payment Days", 0.0, 90.0, 4.0
        )
        values["cash_advance_count_6m"] = st.number_input(
            "Cash Advances (6 Months)", 0, 20, 0
        )
        values["monthly_transaction_count"] = st.number_input(
            "Monthly Transactions", 0, 300, 65
        )

    with col3:
        values["monthly_spend_inr"] = st.number_input(
            "Monthly Spend (INR)", min_value=0, value=21000
        )
        values["complaints_last_6m"] = st.number_input(
            "Complaints (6 Months)", 0, 20, 0
        )
        values["failed_login_attempts_3m"] = st.number_input(
            "Failed Login Attempts", 0, 30, 2
        )
        values["account_tenure_months"] = st.number_input(
            "Account Tenure (Months)", 0, 400, 54
        )
        values["days_since_last_txn"] = st.number_input(
            "Days Since Last Transaction", 0, 365, 30
        )

    submit = st.form_submit_button("Predict Risk")

if submit:
    input_df = pd.DataFrame([values])[FEATURES]
    probability = model.predict_proba(input_df)[0, 1]
    prediction = "HIGH RISK" if probability >= THR else "LOW RISK"

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    col1.metric("Predicted Risk", prediction)
    col2.metric("Risk Probability", f"{probability:.2%}")

    if probability >= THR:
        st.error("HIGH RISK - Manual review recommended.")
    else:
        st.success("LOW RISK")

    metrics = meta.get("test_metrics", meta)

    if "Accuracy" in metrics:
        st.metric("Overall Model Accuracy", f"{metrics['Accuracy']:.2%}")
    elif "accuracy" in metrics:
        st.metric("Overall Model Accuracy", f"{metrics['accuracy']:.2%}")
    else:
        st.info("Model accuracy is not available in the JSON file.")
