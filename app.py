import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="wide")

st.title("🏦 European Bank — Customer Churn Prediction")
st.write("This app predicts whether a bank customer is likely to churn.")

# Load the saved model and helper files (only once, outside the tabs)
model = joblib.load("model/churn_model.pkl")
le_geo = joblib.load("model/le_geo.pkl")
le_gender = joblib.load("model/le_gender.pkl")

with open("model/feature_order.json") as f:
    feature_order = json.load(f)

tab1, tab2, tab3 = st.tabs(["🔮 Predict Churn", "📊 Model Insights", "ℹ️ About"])

with tab1:
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        credit_score = st.slider("Credit Score", 300, 900, 650, key="credit_score")
        geography = st.selectbox("Geography", options=["France", "Germany", "Spain"], key="geography")
        gender = st.selectbox("Gender", options=["Female", "Male"], key="gender")
        age = st.slider("Age", 18, 92, 35, key="age")

    with col2:
        tenure = st.slider("Tenure (years with bank)", 0, 10, 5, key="tenure")
        balance = st.number_input("Account Balance (€)", min_value=0.0, value=50000.0, step=1000.0, key="balance")
        num_products = st.slider("Number of Products", 1, 4, 1, key="num_products")
        estimated_salary = st.number_input("Estimated Salary (€)", min_value=0.0, value=100000.0, step=1000.0, key="salary")

    with col3:
        has_cr_card = st.radio("Has Credit Card?", ["Yes", "No"], horizontal=True, key="cr_card")
        is_active = st.radio("Is Active Member?", ["Yes", "No"], horizontal=True, key="active")

    if st.button("Predict Churn", type="primary"):
        geo_encoded = le_geo.transform([geography])[0]
        gender_encoded = le_gender.transform([gender])[0]
        has_cr_card_val = 1 if has_cr_card == "Yes" else 0
        is_active_val = 1 if is_active == "Yes" else 0
        balance_salary_ratio = balance / (estimated_salary + 1)
        is_zero_balance = 1 if balance == 0 else 0

        row = {
            "CreditScore": credit_score,
            "Geography": geo_encoded,
            "Gender": gender_encoded,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": has_cr_card_val,
            "IsActiveMember": is_active_val,
            "EstimatedSalary": estimated_salary,
            "BalanceSalaryRatio": balance_salary_ratio,
            "IsZeroBalance": is_zero_balance,
        }
        input_df = pd.DataFrame([row])[feature_order]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.divider()
        if prediction == 1:
            st.error(f"⚠️ This customer is likely to churn. (Churn probability: {probability:.1%})")
        else:
            st.success(f"✅ This customer is likely to stay. (Churn probability: {probability:.1%})")

with tab2:
    st.subheader("Exploratory Data Analysis & Model Performance")

    chart_files = [
        "01_churn_distribution.png",
        "02_churn_by_geography.png",
        "03_age_vs_churn.png",
        "04_correlation_heatmap.png",
        "05_confusion_matrix.png",
        "06_feature_importance.png",
    ]

    cols = st.columns(2)
    for i, chart in enumerate(chart_files):
        path = f"outputs/{chart}"
        cols[i % 2].image(path, use_container_width=True)

with tab3:
    st.subheader("About This Project")
    st.markdown("""
    **Project:** Bank Customer Churn Prediction
    **Role:** Finance Analyst Intern
    **Organization:** Unified Mentor Private Limited

    **Objective:** Analyze European bank customer data to identify the key
    drivers of customer churn and build a predictive model that flags
    at-risk customers so the bank can take proactive retention action.

    **Dataset:** 10,000 customer records with demographic, account, and
    activity information.

    **Model:** Random Forest Classifier (compared against Logistic Regression).

    **Tech stack:** Python, pandas, scikit-learn, matplotlib/seaborn, Streamlit.
    """)