# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Credit Score Predictor", layout="centered")

# Sidebar with credit score factors
with st.sidebar:
    st.header("📊 Credit Score Factors")
    st.markdown("""
    **Key factors that affect your credit score:**
    
    • **Payment History** - Most important factor
    • **Credit Utilization** - Keep below 30%
    • **Length of Credit History** - Longer is better
    • **Types of Credit** - Mix of credit accounts
    • **New Credit Inquiries** - Limit hard inquiries
    • **Outstanding Debt** - Lower debt improves score
    • **Number of Bank Accounts** - Shows financial stability
    • **Delayed Payments** - Avoid late payments
    • **Annual Income** - Higher income helps
    • **Interest Rates** - Lower rates indicate good credit
    """)

st.title("Credit Score Predictor 🧾🔮")
st.markdown("Provide applicant details and get a predicted Credit Score using an SVM model.")

# --- Load model & preprocessors ---
MODEL_PATH = "svm_best_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "le_frequency.pkl"

@st.cache_resource(show_spinner=False)
def load_objects():
    missing = []
    model = scaler = le = None
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        missing.append(MODEL_PATH)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
    else:
        missing.append(SCALER_PATH)
    if os.path.exists(ENCODER_PATH):
        le = joblib.load(ENCODER_PATH)
    else:
        missing.append(ENCODER_PATH)
    return model, scaler, le, missing

model, scaler, le, missing = load_objects()

if missing:
    st.error("Missing files: " + ", ".join(missing))
    st.info(
        "Make sure the following files are in the same folder as app.py:\n\n"
        "- svm_best_model.pkl  (trained model)\n"
        "- scaler.pkl          (StandardScaler fitted on training features)\n"
        "- le_frequency.pkl    (LabelEncoder fitted on Payment_Frequency)\n\n"
        "If you don't have them, run the training environment snippet to save them."
    )
    st.stop()

# --- Input form ---
with st.form("input_form"):
    st.subheader("Applicant information")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        annual_income = st.number_input("Annual Income (INR)", min_value=0, value=600000, step=1000)
        monthly_salary_input = st.number_input("Monthly Inhand Salary (optional)", min_value=0.0, value=float(annual_income)/12, step=100.0)
    with col2:
        num_bank_accounts = st.number_input("Num Bank Accounts", min_value=0, max_value=20, value=2, step=1)
        num_credit_cards = st.number_input("Num Credit Cards", min_value=0, max_value=10, value=1, step=1)
        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=12.0, step=0.1)
    with col3:
        num_loans = st.number_input("Num of Loans", min_value=0, max_value=20, value=0, step=1)
        delay_from_due_date = st.number_input("Delay from Due Date (days)", min_value=0, max_value=365, value=0, step=1)
        num_delayed_payments = st.number_input("Num of Delayed Payments", min_value=0, max_value=50, value=0, step=1)

    col4, col5 = st.columns(2)
    with col4:
        num_credit_inquiries = st.number_input("Num Credit Inquiries", min_value=0, max_value=50, value=0, step=1)
        total_debt = st.number_input("Total Outstanding Debt (INR)", min_value=0.0, value=100000.0, step=100.0)
    with col5:
        credit_utilization = st.slider("Credit Utilisation Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        payment_done = st.selectbox("Payment Done (Min Amount?)", options=[0, 1], index=1, help="0 = No, 1 = Yes")

    payment_frequency = st.selectbox(
        "Payment Frequency",
        options=["Weekly", "Bi-Weekly", "Monthly", "Quarterly"],
        index=2
    )

    submitted = st.form_submit_button("Predict Credit Score")

if submitted:
    # Keep naming consistent with training features
    monthly_salary = monthly_salary_input if monthly_salary_input > 0 else annual_income / 12.0

    # Build DataFrame in same order as training
    X_new = pd.DataFrame([{
        "Age": age,
        "Annual_Income": annual_income,
        "Monthly_Salary": monthly_salary,
        "Num_Bank_Accounts": num_bank_accounts,
        "Num_Credit_Cards": num_credit_cards,
        "Interest_Rate": float(interest_rate),
        "Num_Loans": num_loans,
        "Delay_from_Due_Date": delay_from_due_date,
        "Num_Delayed_Payments": num_delayed_payments,
        "Num_Credit_Inquiries": num_credit_inquiries,
        "Total_Debt": total_debt,
        "Credit_Utilisation_Ratio": credit_utilization,
        "Payment_Done": payment_done,
        "Payment_Frequency": payment_frequency
    }])

    # Encode categorical
    try:
        X_new["Payment_Frequency"] = le.transform(X_new["Payment_Frequency"])
    except Exception as e:
        st.error(f"Label encoding failed: {e}")
        st.stop()

    # Ensure numeric dtypes
    X_new = X_new.astype(float)

    # Scale features
    try:
        X_scaled = scaler.transform(X_new)
    except Exception as e:
        st.error(f"Scaling failed: {e}")
        st.stop()

    # Predict
    try:
        pred = model.predict(X_scaled)[0]
        st.success(f"Predicted Credit Score: {pred:.0f}")
        st.write("Note: Score range depends on model training (e.g., 300–700).")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px; margin-top: 50px;'>
    <p><strong>Developer:</strong> Harsh Barewar | <strong>Section:</strong> A3 | <strong>Roll No.:</strong> 46</p>
    <p><strong>College:</strong> Shri Ramdeobaba College of Engineering and Management</p>
    <p>© 2025 Credit Score Predictor App</p>
</div>
""", unsafe_allow_html=True)
