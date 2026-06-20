import streamlit as st
import requests

st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

with st.sidebar:
    st.header("About")
    st.write("""
    Loan Default Risk Predictor

    Model: XGBoost
    Backend: FastAPI
    Frontend: Streamlit
    """)

st.title("🏦 Loan Default Risk Predictor")

st.markdown("""
This application predicts the probability of loan default using a trained XGBoost classifier.

The model evaluates:

- Credit Score
- Debt-to-Income Ratio (DTI)
- Loan-to-Value Ratio (LTV)
- Loan Characteristics
- Financial Stress Indicators
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial Information")

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        value=250000.0
    )

    income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=6500.0
    )

    term = st.number_input(
        "Loan Term (Months)",
        min_value=1.0,
        value=360.0
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300.0,
        max_value=900.0,
        value=720.0
    )

with col2:
    st.subheader("Risk Profile")

    ltv = st.number_input(
        "Loan-To-Value Ratio (LTV)",
        min_value=0.0,
        value=75.0
    )

    dti = st.number_input(
        "Debt-To-Income Ratio (DTI)",
        min_value=0.0,
        value=35.0
    )

    neg_am = st.selectbox(
        "Negative Amortization",
        ["No", "Yes"]
    )

    loan_limit = st.selectbox(
        "Loan Limit",
        ["cf", "ncf", "Unknown"]
    )

st.divider()

col3, col4 = st.columns(2)

with col3:
    age = st.selectbox(
        "Age Group",
        ["<25", "25-34", "35-44", "45-54",
         "55-64", "65-74", ">74", "Unknown"]
    )

with col4:
    approv = st.selectbox(
        "Pre Approval Status",
        ["pre", "nopre", "Unknown"]
    )

purpose = st.selectbox(
    "Loan Purpose",
    ["p1", "p2", "p3", "p4"]
)

st.divider()

if st.button("🔍 Assess Risk", use_container_width=True):

    payload = {
        "loan_amount": loan_amount,
        "income": income,
        "term": term,
        "Credit_Score": credit_score,
        "LTV": ltv,
        "dtir1": dti,
        "Neg_ammortization": neg_am,
        "loan_limit": loan_limit,
        "approv_in_adv": approv,
        "age": age,
        "loan_purpose": purpose
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()

            prob = result["risk_probability"]
            decision = result["action_taken"]

            st.success("Prediction Successful")

            st.subheader("Risk Assessment Result")

            colA, colB = st.columns(2)

            with colA:
                st.metric(
                    "Default Probability",
                    f"{prob * 100:.2f}%"
                )

            with colB:
                st.metric(
                    "Decision",
                    "DENY" if prob >= 0.5 else "APPROVE"
                )

            st.write("### Risk Meter")
            st.progress(min(int(prob * 100), 100))

            if prob < 0.30:
                risk_level = "🟢 LOW RISK"
                risk_text = """
                Applicant appears financially stable.
                Historical default probability is low.
                """
            elif prob < 0.60:
                risk_level = "🟡 MODERATE RISK"
                risk_text = """
                Applicant exhibits moderate credit risk.
                Additional review may be required.
                """
            else:
                risk_level = "🔴 HIGH RISK"
                risk_text = """
                Applicant exhibits elevated default risk.
                Manual review strongly recommended.
                """

            st.subheader(risk_level)
            st.info(risk_text)

            st.subheader("Applicant Summary")

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Credit Score",
                    f"{credit_score:.0f}"
                )

            with c2:
                st.metric(
                    "Monthly Income",
                    f"₹{income:,.0f}"
                )

            with c3:
                st.metric(
                    "Loan Amount",
                    f"₹{loan_amount:,.0f}"
                )

            c4, c5 = st.columns(2)

            with c4:
                st.metric(
                    "DTI",
                    f"{dti:.1f}%"
                )

            with c5:
                st.metric(
                    "LTV",
                    f"{ltv:.1f}%"
                )

            st.subheader("Recommendation")

            if prob < 0.30:
                st.success(
                    "Loan may proceed through standard approval workflow."
                )

            elif prob < 0.60:
                st.warning(
                    "Manual review recommended before approval."
                )

            else:
                st.error(
                    "High-risk application. Additional verification recommended."
                )

            with st.expander("How does the model evaluate risk?"):
                st.markdown("""
                The model evaluates:

                - Credit Score
                - Debt-to-Income Ratio (DTI)
                - Loan-to-Value Ratio (LTV)
                - Loan Purpose
                - Negative Amortization Status
                - Income Strength
                - Financial Stress Indicators

                Final prediction is generated using a trained XGBoost classifier.
                """)

            st.divider()

            st.subheader("Model Performance")

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("AUC", "0.88")

            with m2:
                st.metric("Precision", "0.74")

            with m3:
                st.metric("Recall", "0.69")

            st.subheader("Model Decision")
            st.write(decision)

            st.caption(
                f"Predicted probability of default: {prob:.4f}"
            )

        else:
            st.error(
                f"Backend returned status code {response.status_code}"
            )
            st.json(response.json())

    except Exception as e:
        st.error(f"Connection Error: {e}")