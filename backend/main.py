import os
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field



class LoanApplication(BaseModel):
    # Continuous Numerical Inputs
    loan_amount: float = Field(..., example=250000.0, description="Total requested loan principal amount ($ or ₹)")
    income: float = Field(..., example=6500.0, description="Gross monthly income of the primary applicant")
    term: float = Field(..., example=360.0, description="Loan repayment duration contract length in months")
    Credit_Score: float = Field(..., example=720.0, description="Applicant credit worthiness score (typically 300-850)")
    
    # Nullable Numerical Inputs (Your engineered missingness paths will catch these if passed as None/null)
    LTV: Optional[float] = Field(None, example=75.2, description="Loan-to-Value Ratio percentage. Pass null to simulate missingness.")
    dtir1: Optional[float] = Field(None, example=38.5, description="Debt-to-Income Ratio percentage. Pass null to simulate missingness.")
    
    # Categorical Inputs (Must strictly pass exact string variants)
    Neg_ammortization: str = Field("No", example="No", description="Options: 'Yes' or 'No'")
    loan_limit: str = Field("cf", example="cf", description="Options: 'cf' (Conforming), 'ncf' (Non-Conforming), or 'Unknown'")
    approv_in_adv: str = Field("nopre", example="nopre", description="Options: 'pre' (Pre-approved), 'nopre' (No pre-approval), or 'Unknown'")
    age: str = Field("45-54", example="45-54", description="Options: '<25', '25-34', '35-44', '45-54', '55-64', '65-74', '>74', or 'Unknown'")
    loan_purpose: str = Field("p1", example="p1", description="Options: 'p1', 'p2', 'p3', 'p4', or 'Unknown'")

  
app = FastAPI(
    title="B.Tech Project: Loan Defaulter Prediction API",
    description="A production-grade REST API serving an engineered XGBoost Binary Classifier.",
    version="1.0.0"
)

# Declare global reference placeholders so our endpoints can access them later
model_pipeline = None


@app.on_event("startup")
def load_ml_artifacts():
    global model_pipeline

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(BASE_DIR, "loan_default_pipeline.pkl")

    print("\n--- Initializing Backend Server & Loading ML Pipeline ---")

    if not os.path.exists(filename):
        error_msg = (
            f"CRITICAL ERROR: Missing pipeline file "
            f"in '{os.getcwd()}'."
        )
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)

    try:
        model_pipeline = joblib.load(filename)

        print(
            "✅ SUCCESS: "
            "'loan_default_pipeline.pkl' loaded smoothly into memory."
        )

    except Exception as e:
        error_msg = (
            "CRITICAL ERROR: Failed to deserialize pipeline. "
            f"Details: {str(e)}"
        )

        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    
    # ==========================================
# COMPONENT 3: THE FEATURE ENGINEERING & ALIGNMENT ENGINE
# ==========================================
def create_features(raw_data):

    df = pd.DataFrame([raw_data])

    df["LTV_is_missing"] = df["LTV"].isnull().astype(float)

    df["dti_missing"] = df["dtir1"].isnull().astype(float)

    df["annual_income"] = df["income"] * 12

    df["loan_income_ratio"] = np.where(
        df["annual_income"] == 0,
        np.nan,
        df["loan_amount"] / df["annual_income"]
    )

    df["principal_to_income_ratio"] = np.where(
        df["income"] == 0,
        np.nan,
        (df["loan_amount"] / df["term"]) / df["income"]
    )

    df["principal_to_income_ratio"] = np.where(
        df["principal_to_income_ratio"] > 1.0,
        1.0,
        df["principal_to_income_ratio"]
    )

    return df[
        [
            'loan_amount',
            'income',
            'annual_income',
            'term',
            'LTV',
            'dtir1',
            'principal_to_income_ratio',
            'loan_income_ratio',
            'Credit_Score',
            'LTV_is_missing',
            'dti_missing',
            'Neg_ammortization',
            'loan_limit',
            'approv_in_adv',
            'age',
            'loan_purpose'
        ]
    ]



@app.get("/health")
def health_check():
    """
    Lightweight health check endpoint. 
    Used to verify that the server is alive and ML assets are successfully loaded.
    """
    if model_pipeline is not None :
        return {"status": "healthy", "model_loaded": True, "message": "Ready for credit scoring."}
    else:
        raise HTTPException(status_code=500, detail="Server running but ML assets are missing.")


@app.post("/predict")
def predict_loan_default(payload: LoanApplication):

    if model_pipeline is None:
        raise HTTPException(
            status_code=500,
            detail="Inference Engine Error: Pipeline is not loaded."
        )

    try:

        raw_input_dict = payload.model_dump()

        processed_df = create_features(raw_input_dict)

        probability_of_default = float(
            model_pipeline.predict_proba(processed_df)[0][1]
        )

        prediction_code = (
            1 if probability_of_default >= 0.5 else 0
        )

        return {
            "risk_probability": round(probability_of_default, 4),
            "prediction_code": prediction_code,
            "action_taken":
                "DENY (High Default Risk Profile)"
                if prediction_code == 1
                else
                "APPROVE (Clean Risk Profile)",
            "classification_threshold": 0.5
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )