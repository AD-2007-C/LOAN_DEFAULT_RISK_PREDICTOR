# Loan Default Risk Predictor

## Overview

This project is an end-to-end Machine Learning application that predicts the probability of loan default using applicant financial and loan-related information.

The system combines a trained XGBoost classifier, a FastAPI backend for model serving, and a Streamlit frontend for interactive risk assessment.

---

## Features

* Real-time loan default prediction
* Automated feature engineering pipeline
* Probability-based risk scoring
* FastAPI REST API backend
* Streamlit user interface
* SHAP-based model explainability
* Production-ready preprocessing pipeline using Scikit-Learn

---

## Tech Stack

### Machine Learning

* Python
* Scikit-Learn
* XGBoost
* NumPy
* Pandas
* SHAP

### Backend

* FastAPI
* Uvicorn

### Frontend

* Streamlit

---

## Input Features

The model evaluates multiple borrower and loan characteristics, including:

* Loan Amount
* Monthly Income
* Credit Score
* Loan Term
* Debt-to-Income Ratio (DTI)
* Loan-to-Value Ratio (LTV)
* Age Group
* Loan Purpose
* Approval Status
* Negative Amortization Status

Additional engineered features are created during preprocessing to improve predictive performance.

---

## Model Pipeline

The preprocessing pipeline performs:

1. Missing value handling
2. Feature engineering
3. Numerical scaling
4. Categorical encoding
5. XGBoost inference

The complete pipeline is serialized and deployed for consistent training and inference behavior.

---

## Project Structure

```text
Loan-Default-Risk-Predictor/

├── backend/
│   ├── main.py
│   ├── loan_default_pipeline.pkl
│
├── frontend/
│   ├── app.py
│
├── notebook/
│   ├── training.ipynb
│
├── README.md
└── .gitignore
```

---

## Running the Project

### Backend

```bash
cd backend
uvicorn main:app --reload
```

Backend runs on:

```text
http://127.0.0.1:8000
```

---

### Frontend

```bash
cd frontend
streamlit run app.py
```

Frontend runs on:

```text
http://localhost:8501
```

---

## API Endpoint

### POST /predict

Accepts applicant and loan information and returns:

* Default probability
* Classification result
* Risk assessment decision

Example response:

```json
{
  "risk_probability": 0.18,
  "prediction_code": 0,
  "action_taken": "APPROVE"
}
```

---

## Future Improvements

* Interactive SHAP visualizations within the frontend
* Cloud deployment
* Batch prediction support
* Model monitoring and analytics dashboard
* CI/CD integration

---

## Live Demo

Frontend (Streamlit):
[https://your-streamlit-link](https://loandefaultriskpredictor-hdl75ofdxggtywdj2ghnxm.streamlit.app)

Backend (FastAPI):
https://loandefaultriskpredictor-production.up.railway.app/docs
## Author

Aditya Nair

B.Tech Student, Delhi Technological University (DTU)
