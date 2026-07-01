"""
RiskLens AI — FastAPI REST API Service
=======================================
Exposes the scoring engine as a REST API endpoint. Perfect for connecting 
external frontends (like React apps built on Lovable.dev).

To run:
    pip install fastapi uvicorn
    uvicorn app.api:app --reload --port 8000
"""

import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import polars as pl

from src.scoring.predictor import RiskLensScorer
from src.features.engineering import _add_features

app = FastAPI(
    title="RiskLens AI API",
    description="REST API to score retail loan default probabilities using pre-trained LightGBM models.",
    version="1.0.0"
)

# Enable CORS for external frontends (e.g. Lovable web app, local dev servers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize scorer globally
try:
    scorer = RiskLensScorer(use_shap=True)
except Exception as e:
    # Fallback to run without loading models if they don't exist yet
    scorer = None
    print(f"WARNING: Models not loaded. Please run training pipeline first. Error: {e}")

# Cache median template in memory
median_df = None
def get_cached_median_template():
    global median_df
    if median_df is None and scorer is not None:
        # Load 1,000 samples from the test set to build a stable median template
        df_sample = scorer.load_sample_from_test_set(n=1000, issue_year=2018)
        median_series = df_sample.median(numeric_only=True)
        median_df = pd.DataFrame([median_series])
        
        for col in df_sample.columns:
            if col not in median_df.columns:
                mode_vals = df_sample[col].mode()
                median_df[col] = mode_vals.iloc[0] if not mode_vals.empty else df_sample[col].iloc[0]
        median_df = median_df[df_sample.columns]
    return median_df

# Pydantic input schema
class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., example=15000.0, description="The requested loan amount in dollars.")
    term: int = Field(..., example=36, description="Loan term (36 or 60 months).")
    annual_inc: float = Field(..., example=75000.0, description="Borrower's annual income.")
    dti: float = Field(..., example=15.5, description="Debt-to-Income ratio (%).")
    revol_bal: float = Field(..., example=12000.0, description="Outstanding credit card balance ($).")
    revol_util: float = Field(..., example=35.0, description="Revolving utilization rate (%).")
    delinq_2yrs: int = Field(..., example=0, description="Number of delinquencies in past 2 years.")

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "RiskLens AI Prediction API",
        "model_name": scorer.model_name if scorer else "None Loaded",
        "threshold": scorer.threshold if scorer else 0.0
    }

@app.post("/score")
def score_loan(app_data: LoanApplication):
    if scorer is None:
        raise HTTPException(
            status_code=503, 
            detail="Scoring service offline. Models not trained or not found."
        )
        
    try:
        # 1. Fetch the default baseline template
        base_df = get_cached_median_template().copy()
        
        # 2. Overwrite with user input values
        base_df["loan_amnt"] = float(app_data.loan_amnt)
        base_df["term"] = int(app_data.term)
        base_df["annual_inc"] = float(app_data.annual_inc)
        base_df["dti"] = float(app_data.dti)
        base_df["revol_bal"] = float(app_data.revol_bal)
        base_df["revol_util"] = float(app_data.revol_util)
        base_df["delinq_2yrs"] = int(app_data.delinq_2yrs)
        
        # 3. Dynamic Feature Engineering (computes credit_util_x_balance, loan_pct_of_income, etc.)
        df_pl = pl.DataFrame(base_df)
        df_pl = _add_features(df_pl)
        feature_row = df_pl.to_pandas()
        
        # 4. Score the application
        result = scorer.score_application(feature_row, include_shap=True)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
