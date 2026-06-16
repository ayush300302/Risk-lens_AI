"""
RiskLens AI — Model Training Module
====================================
Contains the core machine learning training pipeline:
1. Temporal data splitting (Train: 2007-2016, Val: 2017, Test: 2018)
2. Preprocessing (StandardScaler+OHE for LR, passthrough+Ordinal for LGB)
3. Model training (Logistic Regression baseline & LightGBM champion)
4. Evaluation metrics (ROC-AUC, PR-AUC, KS statistic, Precision/Recall)
5. Optimal threshold calibration using Youden's J statistic
6. Model artifact serialization

Interview tip:
    "I use two models — a simple Logistic Regression as a baseline for
    interpretability, and LightGBM as the champion for accuracy.
    I compare them on an out-of-time test set (2018 data) to simulate
    how the model would perform on truly unseen future applications."
"""

import os
from pathlib import Path
from typing import Dict, Tuple, List, Any
import polars as pl
import pandas as pd
import numpy as np
import joblib

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    classification_report, roc_curve
)
import lightgbm as lgb

from app_config.logging_config import get_logger
from src.utils.memory import MemoryProfiler

logger = get_logger(__name__)


def _cast_lgb_dtypes(
    X: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    """Ensure numeric float + categorical category dtypes for LightGBM."""
    df = X[numerical_cols + categorical_cols].copy()
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    for col in categorical_cols:
        df[col] = df[col].astype(str).astype("category")
    return df


class _LGBMWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper so Pipeline.predict_proba coerces dtypes correctly."""

    def __init__(self, model: lgb.LGBMClassifier, numerical_cols: List[str], categorical_cols: List[str]):
        self.model = model
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_lgb = _cast_lgb_dtypes(X, self.numerical_cols, self.categorical_cols)
        return self.model.predict_proba(X_lgb)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_lgb = _cast_lgb_dtypes(X, self.numerical_cols, self.categorical_cols)
        return self.model.predict(X_lgb)


def load_and_split_data(features_path: str | Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Load feature store parquet and perform a temporal split.
    - Train: 2007 - 2016
    - Val: 2017
    - Test: 2018

    Drops high-cardinality/unusable columns like 'emp_title'.
    """
    features_path = Path(features_path)
    logger.info(f"Loading features from {features_path}...")

    # Load using Polars for memory efficiency
    df = pl.read_parquet(features_path)
    
    # Drop unusable high-cardinality string columns
    drop_cols = ["emp_title"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(col)
            logger.info(f"Dropped column: {col}")

    # Convert to Pandas for compatibility with Scikit-Learn / LightGBM
    logger.info("Converting Polars DataFrame to Pandas...")
    df_pd = df.to_pandas()
    
    # Identify feature types
    target_col = "target"
    split_col = "issue_year"
    
    # Drop rows where target is null
    df_pd = df_pd.dropna(subset=[target_col])
    
    # Define splits
    train_mask = df_pd[split_col] <= 2016
    val_mask = df_pd[split_col] == 2017
    test_mask = df_pd[split_col] == 2018
    
    logger.info(f"Split sizes -> Train: {train_mask.sum():,}, Val: {val_mask.sum():,}, Test: {test_mask.sum():,}")
    
    # Features and labels
    feature_cols = [c for c in df_pd.columns if c not in [target_col, split_col]]
    
    # Identify numeric and categorical columns
    categorical_cols = []
    numerical_cols = []
    
    for col in feature_cols:
        if df_pd[col].dtype.name in ["category", "object"]:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
            
    logger.info(f"Features: {len(feature_cols)} total ({len(numerical_cols)} numerical, {len(categorical_cols)} categorical)")
    logger.info(f"Categorical features: {categorical_cols}")

    X_train = df_pd[train_mask][feature_cols].copy()
    y_train = df_pd[train_mask][target_col].astype(np.int8).copy()
    
    X_val = df_pd[val_mask][feature_cols].copy()
    y_val = df_pd[val_mask][target_col].astype(np.int8).copy()
    
    X_test = df_pd[test_mask][feature_cols].copy()
    y_test = df_pd[test_mask][target_col].astype(np.int8).copy()
    
    return X_train, y_train, X_val, y_val, X_test, y_test, numerical_cols, categorical_cols


def calculate_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic.
    
    KS measures the maximum separation between the cumulative distributions
    of defaults (positives) and non-defaults (negatives).
    
    Financial meaning:
        KS > 40% = excellent, 30-40% = good, 20-30% = acceptable.
    """
    from scipy.stats import ks_2samp
    pos_probs = y_prob[y_true == 1]
    neg_probs = y_prob[y_true == 0]
    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return 0.0
    ks_stat, _ = ks_2samp(pos_probs, neg_probs)
    return float(ks_stat * 100.0)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Find the optimal classification threshold using Youden's J statistic.
    J = Sensitivity + Specificity - 1 = TPR - FPR
    
    The threshold that maximizes J gives the best balance between
    catching defaults (recall) and not over-flagging good borrowers.
    
    Interview tip:
        "I don't use 0.5 as the threshold because with 20% default rate,
        0.5 is too conservative — the model never predicts default.
        Instead, I use Youden's J statistic to find the threshold that
        maximizes the separation between true positives and false positives."
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[best_idx])
    logger.info(f"Optimal threshold: {optimal_threshold:.4f} (Youden's J = {j_scores[best_idx]:.4f})")
    return optimal_threshold


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate model and return dictionary of metrics.
    Uses a calibrated threshold instead of the default 0.5.
    """
    logger.info(f"Evaluating model: {model_name} (threshold={threshold:.4f})...")
    
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
        
    # ROC-AUC
    roc_auc = roc_auc_score(y, y_prob)
    
    # PR-AUC
    prec_curve, rec_curve, _ = precision_recall_curve(y, y_prob)
    pr_auc = auc(rec_curve, prec_curve)
    
    # KS Statistic
    ks_stat = calculate_ks_statistic(y.values, y_prob)
    
    # Gini Coefficient = 2 * ROC-AUC - 1
    gini = 2.0 * roc_auc - 1.0
    
    # Standard classification report
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    accuracy = report["accuracy"]
    default_precision = report["1"]["precision"]
    default_recall = report["1"]["recall"]
    default_f1 = report["1"]["f1-score"]
    
    metrics = {
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4),
        "Gini": round(gini, 4),
        "KS-Stat (%)": round(ks_stat, 2),
        "Accuracy": round(accuracy, 4),
        "Default-Precision": round(default_precision, 4),
        "Default-Recall": round(default_recall, 4),
        "Default-F1": round(default_f1, 4),
        "Threshold": round(threshold, 4),
    }
    
    logger.info(f"Results for {model_name}: {metrics}")
    return metrics


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numerical_cols: List[str],
    categorical_cols: List[str],
    random_seed: int = 42
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Train a Logistic Regression baseline model.
    Returns: (fitted pipeline, validation metrics, optimal threshold)
    """
    logger.info("Setting up Logistic Regression pipeline...")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ]
    )
    
    lr_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=random_seed,
                    max_iter=1000,
                ),
            ),
        ]
    )
    
    logger.info("Training Logistic Regression...")
    lr_pipeline.fit(X_train, y_train)
    
    # Find optimal threshold on validation set
    y_val_prob = lr_pipeline.predict_proba(X_val)[:, 1]
    optimal_threshold = find_optimal_threshold(y_val.values, y_val_prob)
    
    # Evaluate on validation set with optimal threshold
    val_metrics = evaluate_model(lr_pipeline, X_val, y_val, "LR Baseline (Val)", threshold=optimal_threshold)
    
    return lr_pipeline, val_metrics, optimal_threshold


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numerical_cols: List[str],
    categorical_cols: List[str],
    random_seed: int = 42
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Train a LightGBM champion classifier.
    Returns: (fitted pipeline, validation metrics, optimal threshold)
    """
    logger.info("Setting up LightGBM pipeline...")

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    logger.info(f"Class ratio (neg/pos): {scale_pos_weight:.2f}")

    X_train_df = _cast_lgb_dtypes(X_train, numerical_cols, categorical_cols)
    X_val_df = _cast_lgb_dtypes(X_val, numerical_cols, categorical_cols)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_seed,
        n_jobs=-1,
        verbosity=-1,
    )

    logger.info("Training LightGBM Champion model...")
    lgb_model.fit(
        X_train_df,
        y_train.values,
        eval_set=[(X_val_df, y_val.values)],
        eval_metric="auc",
        categorical_feature=categorical_cols,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    logger.info(f"LightGBM trained with {lgb_model.n_estimators} trees")

    lgb_pipeline = Pipeline(
        steps=[
            ("classifier", _LGBMWrapper(lgb_model, numerical_cols, categorical_cols)),
        ]
    )
    lgb_pipeline.fit(X_train, y_train)
    
    # Find optimal threshold on validation set
    y_val_prob = lgb_pipeline.predict_proba(X_val)[:, 1]
    optimal_threshold = find_optimal_threshold(y_val.values, y_val_prob)
    
    # Evaluate on validation set with optimal threshold
    val_metrics = evaluate_model(lgb_pipeline, X_val, y_val, "LightGBM Champion (Val)", threshold=optimal_threshold)
    
    return lgb_pipeline, val_metrics, optimal_threshold
