"""
RiskLens AI — Model Explainability Module
==========================================
Uses SHAP (SHapley Additive exPlanations) to explain LightGBM predictions:
1. Global explainability: Mean absolute SHAP values to rank features globally
2. Local explainability: Explaining a single borrower's prediction by decomposing
   the model output into feature contributions (additive force/waterfall explanation)
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import joblib
import shap

from app_config.logging_config import get_logger

logger = get_logger(__name__)


def get_tree_explainer(model_pipeline_path: str | Path) -> Tuple[Any, Any, List[str]]:
    """
    Load the fitted LightGBM pipeline and initialize a shap.TreeExplainer.
    Returns:
        explainer: The SHAP TreeExplainer instance
        preprocessor: The fitted ColumnTransformer pipeline step
        feature_names: List of preprocessed feature names in order
    """
    model_pipeline_path = Path(model_pipeline_path)
    logger.info(f"Loading model pipeline from {model_pipeline_path}...")
    pipeline = joblib.load(model_pipeline_path)
    
    # Extract pipeline steps (LightGBM pipeline may omit preprocessor)
    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "model"):
        classifier = classifier.model
    
    # Extract feature names after preprocessing
    # Since ColumnTransformer preserves order: numerical cols first, then categorical
    metadata_path = model_pipeline_path.parent / "model_metadata.joblib"
    if metadata_path.exists():
        metadata = joblib.load(metadata_path)
        feature_names = metadata["numerical_features"] + metadata["categorical_features"]
    else:
        # Fallback if metadata not found
        feature_names = [f"feature_{i}" for i in range(classifier.n_features_in_)]
        
    # Initialize TreeExplainer directly on the LightGBM booster/classifier
    # For binary classification, we use the classifier model
    logger.info("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(classifier)
    
    return explainer, preprocessor, feature_names


def explain_prediction(
    model_pipeline_path: str | Path,
    input_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generate local SHAP explanations for a single loan application.
    
    Args:
        model_pipeline_path: Path to champion_lgb.joblib
        input_data: Single row DataFrame containing the raw features of the borrower
        
    Returns:
        Dict containing prediction probability, base value, SHAP contributions, and original values.
    """
    explainer, preprocessor, feature_names = get_tree_explainer(model_pipeline_path)
    
    # Preprocess input
    logger.info("Preprocessing input data for explanation...")
    if preprocessor is not None:
        x_proc = preprocessor.transform(input_data)
    else:
        from src.models.training import _cast_lgb_dtypes
        meta = joblib.load(model_pipeline_path.parent / "model_metadata.joblib")
        x_proc = _cast_lgb_dtypes(
            input_data,
            meta["numerical_features"],
            meta["categorical_features"],
        ).values
    
    # Get probability prediction
    pipeline = joblib.load(model_pipeline_path)
    prob = float(pipeline.predict_proba(input_data)[0, 1])
    
    # Compute SHAP values
    # shap_values is an array of shape (n_samples, n_features, n_classes) or (n_samples, n_features)
    # For TreeExplainer on LightGBM Classifier, it returns shape (n_samples, n_features, 2) or (n_samples, n_features)
    # we want the SHAP values for the positive class (class 1: Default)
    raw_shap = explainer(x_proc)
    
    # Handle multi-class vs single-class shape variations in SHAP output
    if len(raw_shap.shape) == 3:  # (samples, features, classes)
        shap_vals = raw_shap.values[0, :, 1]
        base_value = raw_shap.base_values[0, 1]
    else:  # (samples, features)
        # Note: If TreeExplainer outputs in margin space (log-odds), we might need to convert or keep it as log-odds.
        # But in LightGBM, shap values sum to the model margin (log-odds).
        # Standard approach is to return SHAP values in log-odds space alongside base value.
        shap_vals = raw_shap.values[0]
        base_value = float(raw_shap.base_values[0]) if isinstance(raw_shap.base_values, np.ndarray) else float(raw_shap.base_values)
        
    # Convert log-odds back to probability for explainable numbers if possible:
    # However, SHAP is additively linear only in the log-odds space.
    # Therefore, we present contributions in log-odds contribution (impact on risk)
    # and display the final probability.
    
    contributions = []
    for i, name in enumerate(feature_names):
        # Find original value before preprocessing for user readability
        orig_val = input_data[name].iloc[0] if name in input_data.columns else None
        
        # If it was categorical, return the string category label
        if isinstance(orig_val, (pd.CategoricalDtype, str)) or (
            preprocessor is not None
            and len(preprocessor.transformers_) > 1
            and name in preprocessor.transformers_[1][2]
        ):
            orig_val_str = str(orig_val)
        else:
            orig_val_str = f"{orig_val:.2f}" if isinstance(orig_val, (float, np.float32, np.float64)) else str(orig_val)
            
        contributions.append({
            "feature": name,
            "shap_value": float(shap_vals[i]),
            "feature_value": orig_val_str,
            "impact": "increases_risk" if shap_vals[i] > 0 else ("decreases_risk" if shap_vals[i] < 0 else "neutral")
        })
        
    # Sort contributions by absolute SHAP value (most impactful first)
    contributions = sorted(contributions, key=lambda x: abs(x["shap_value"]), reverse=True)
    
    explanation = {
        "probability_of_default": prob,
        "base_value_log_odds": base_value,
        "prediction_log_odds": float(base_value + sum(shap_vals)),
        "contributions": contributions
    }
    
    return explanation


def generate_global_explanations(
    model_pipeline_path: str | Path,
    test_features_path: str | Path,
    output_csv_path: str | Path,
    sample_size: int = 2000
) -> pd.DataFrame:
    """
    Generate global feature importances by calculating the mean absolute SHAP values
    over a representative sample of test data. Saves findings to csv.
    """
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    explainer, preprocessor, feature_names = get_tree_explainer(model_pipeline_path)
    
    logger.info(f"Loading test dataset from {test_features_path} for global SHAP...")
    df = pd.read_parquet(test_features_path)
    
    # Exclude non-features
    feature_cols = [c for c in df.columns if c not in ["target", "issue_year", "emp_title"]]
    df_feat = df[feature_cols]
    
    # Sample data
    if len(df_feat) > sample_size:
        logger.info(f"Sampling {sample_size} rows for SHAP evaluation...")
        df_sample = df_feat.sample(n=sample_size, random_state=42)
    else:
        df_sample = df_feat
        
    logger.info("Preprocessing test sample...")
    if preprocessor is not None:
        x_proc = preprocessor.transform(df_sample)
    else:
        from src.models.training import _cast_lgb_dtypes
        meta = joblib.load(model_pipeline_path.parent / "model_metadata.joblib")
        x_proc = _cast_lgb_dtypes(
            df_sample,
            meta["numerical_features"],
            meta["categorical_features"],
        ).values
    
    logger.info("Computing SHAP values for test sample...")
    shap_output = explainer(x_proc)
    
    if len(shap_output.shape) == 3:  # (samples, features, classes)
        shap_vals = shap_output.values[:, :, 1]
    else:  # (samples, features)
        shap_vals = shap_output.values
        
    # Compute mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    
    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    
    shap_importance.to_csv(output_csv_path, index=False)
    logger.info(f"Global SHAP feature importance saved to: {output_csv_path}")
    
    return shap_importance
