"""
RiskLens AI — End-to-End Scoring Service
=========================================
Loads the champion model and returns a full credit decision:
  PD (probability of default) → risk bucket → policy decision → optional SHAP.

This wires together modules that were previously standalone:
  - src.models.training (model artifacts)
  - src.risk.bucketing (risk tiers)
  - src.risk.decision (knockout policy rules)
  - src.explainability.shap_explainer (local explanations)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import polars as pl

from app_config.settings import get_settings
from app_config.logging_config import get_logger
from src.risk.bucketing import assign_risk_bucket
from src.risk.decision import evaluate_policy_rules
from src.explainability.shap_explainer import explain_prediction

logger = get_logger(__name__)


class RiskLensScorer:
    """
    Production-style scorer: one borrower in, full decision out.

    Usage:
        scorer = RiskLensScorer()
        result = scorer.score_application(feature_row)
    """

    def __init__(self, model_dir: Path | None = None, use_shap: bool = True):
        settings = get_settings()
        self.model_dir = Path(model_dir or settings.model_store_path)
        self.use_shap = use_shap

        metadata_path = self.model_dir / "model_metadata.joblib"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Model metadata not found at {metadata_path}. Run pipelines/train.py first."
            )

        self.metadata = joblib.load(metadata_path)
        self.feature_names: List[str] = self.metadata["all_features"]
        self.lgb_threshold: float = float(self.metadata["lgb_threshold"])
        self.lr_threshold: float = float(self.metadata["lr_threshold"])

        lgb_path = self.model_dir / "champion_lgb.joblib"
        lr_path = self.model_dir / "baseline_lr.joblib"
        lgb_metrics = self.metadata.get("lgb_test_metrics", {})
        lr_metrics = self.metadata.get("lr_test_metrics", {})

        # Pick champion by out-of-time test ROC-AUC stored at training time.
        if lgb_metrics.get("ROC-AUC", 0) >= lr_metrics.get("ROC-AUC", 0):
            self.model = joblib.load(lgb_path)
            self.model_name = "LightGBM"
            self.threshold = self.lgb_threshold
            self.model_path = lgb_path
        else:
            self.model = joblib.load(lr_path)
            self.model_name = "Logistic Regression"
            self.threshold = self.lr_threshold
            self.model_path = lr_path

        logger.info(f"Scorer ready: {self.model_name} (threshold={self.threshold:.4f})")

    def _prepare_features(self, application: pd.DataFrame) -> pd.DataFrame:
        """Align input to the exact feature schema the model expects."""
        if len(application) != 1:
            raise ValueError("score_application expects exactly one row")

        row = application.copy()
        missing = [c for c in self.feature_names if c not in row.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing[:10]}{'...' if len(missing) > 10 else ''}")

        return row[self.feature_names]

    def score_application(
        self,
        application: pd.DataFrame,
        include_shap: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Score a single loan application.

        Args:
            application: One-row DataFrame with all model features.
            include_shap: Override instance SHAP setting (slower).

        Returns:
            Dict with PD, risk bucket, policy decision, and optional SHAP.
        """
        X = self._prepare_features(application)
        pd_prob = float(self.model.predict_proba(X)[0, 1])
        predicted_default = pd_prob >= self.threshold

        bucket = assign_risk_bucket(pd_prob)

        row = application.iloc[0]

        def _val(col: str, default: float = 0.0) -> float:
            if col in application.columns:
                v = row[col]
                return float(v) if pd.notna(v) else default
            if col in X.columns:
                v = X[col].iloc[0]
                return float(v) if pd.notna(v) else default
            return default

        policy = evaluate_policy_rules(
            default_prob=pd_prob,
            dti=_val("dti"),
            inq_last_6mths=_val("inq_last_6mths"),
            delinq_2yrs=_val("delinq_2yrs"),
            annual_inc=_val("annual_inc", 1.0),
            loan_amnt=_val("loan_amnt", 1.0),
        )

        result: Dict[str, Any] = {
            "model": self.model_name,
            "threshold": round(self.threshold, 4),
            "probability_of_default": round(pd_prob, 4),
            "predicted_default": bool(predicted_default),
            "risk_bucket": bucket,
            "policy_decision": policy,
        }

        run_shap = self.use_shap if include_shap is None else include_shap
        if run_shap and self.model_name == "LightGBM":
            try:
                shap_result = explain_prediction(self.model_path, X)
                result["shap_top_factors"] = shap_result["contributions"][:5]
            except Exception as exc:
                logger.warning(f"SHAP explanation skipped: {exc}")
                result["shap_top_factors"] = []

        return result

    def score_batch(self, applications: pd.DataFrame) -> pd.DataFrame:
        """Score many applications; returns PD and bucket per row (no per-row SHAP)."""
        X = applications[self.feature_names]
        probs = self.model.predict_proba(X)[:, 1]

        rows = []
        for i, prob in enumerate(probs):
            bucket = assign_risk_bucket(float(prob))
            row = applications.iloc[i]
            policy = evaluate_policy_rules(
                default_prob=float(prob),
                dti=float(row.get("dti", 0)),
                inq_last_6mths=float(row.get("inq_last_6mths", 0)),
                delinq_2yrs=float(row.get("delinq_2yrs", 0)),
                annual_inc=float(row.get("annual_inc", 1)),
                loan_amnt=float(row.get("loan_amnt", 1)),
            )
            rows.append(
                {
                    "probability_of_default": round(float(prob), 4),
                    "risk_tier": bucket["risk_tier"],
                    "credit_grade": bucket["credit_grade"],
                    "recommended_action": bucket["recommended_action"],
                    "policy_decision": policy["decision"],
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def load_sample_from_test_set(cls, n: int = 1, issue_year: int = 2018) -> pd.DataFrame:
        """Load real feature rows from the held-out test period for demos."""
        settings = get_settings()
        features_path = settings.feature_store_path / "features_v1.parquet"
        df = pl.read_parquet(features_path)
        sample = (
            df.filter(pl.col("issue_year") == issue_year)
            .drop(["target", "issue_year", "emp_title"], strict=False)
            .sample(n=n, seed=42)
        )
        return sample.to_pandas()
