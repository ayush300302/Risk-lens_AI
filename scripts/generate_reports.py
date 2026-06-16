"""
RiskLens AI — Step 8: Generate Reports & Visualizations
========================================================
Creates charts for hackathon submission and portfolio review.

Outputs (in reports/):
  - roc_curve.png
  - pr_curve.png
  - risk_bucket_default_rates.png
  - shap_importance.png
  - portfolio_risk_distribution.png
  - model_comparison.json

Usage:
    python scripts/generate_reports.py
"""

import sys
import io
import json
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from app_config.settings import get_settings
from app_config.logging_config import setup_logging, get_logger
from src.models.training import load_and_split_data
from src.scoring.predictor import RiskLensScorer

setup_logging()
logger = get_logger(__name__)


def plot_roc(y_true, y_prob, out_path: Path, title: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_pr(y_true, y_prob, out_path: Path, title: str):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    plt.figure(figsize=(7, 5))
    plt.plot(rec, prec, label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_shap_importance(shap_csv: Path, out_path: Path, top_n: int = 15):
    df = pd.read_csv(shap_csv).head(top_n)
    plt.figure(figsize=(8, 6))
    plt.barh(df["feature"][::-1], df["mean_abs_shap"][::-1], color="#2563eb")
    plt.xlabel("Mean |SHAP|")
    plt.title(f"Top {top_n} Global Feature Importances")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_bucket_default_rates(test_df: pd.DataFrame, out_path: Path):
    """Validate that default rate rises monotonically across risk tiers."""
    scorer = RiskLensScorer(use_shap=False)
    feature_df = test_df.drop(columns=["target", "issue_year", "emp_title"], errors="ignore")
    scored = scorer.score_batch(feature_df)
    combined = pd.concat([test_df.reset_index(drop=True), scored], axis=1)

    summary = (
        combined.groupby("risk_tier", observed=True)
        .agg(
            count=("probability_of_default", "count"),
            avg_pd=("probability_of_default", "mean"),
            actual_default_rate=("target", "mean"),
        )
        .reset_index()
    )

    tier_order = ["Very Low", "Low", "Medium", "High", "Very High"]
    summary["risk_tier"] = pd.Categorical(summary["risk_tier"], categories=tier_order, ordered=True)
    summary = summary.sort_values("risk_tier")

    x = np.arange(len(summary))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, summary["avg_pd"], width, label="Avg Predicted PD")
    plt.bar(x + width / 2, summary["actual_default_rate"], width, label="Actual Default Rate")
    plt.xticks(x, summary["risk_tier"], rotation=15)
    plt.ylabel("Rate")
    plt.title("Risk Bucket Validation (2018 Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()

    summary.to_csv(out_path.with_suffix(".csv"), index=False)
    return summary


def plot_portfolio_distribution(scored: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(8, 5))
    plt.hist(scored["probability_of_default"], bins=40, color="#0d9488", edgecolor="white")
    plt.xlabel("Probability of Default")
    plt.ylabel("Loan Count")
    plt.title("Portfolio PD Distribution (2018 Test Set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    print("\n" + "=" * 60)
    print("  RiskLens AI — Step 8: Reports & Visualizations")
    print("=" * 60 + "\n")

    settings = get_settings()
    reports_dir = settings.reports_path
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_dir = settings.model_store_path

    metadata = joblib.load(model_dir / "model_metadata.joblib")
    lr_model = joblib.load(model_dir / "baseline_lr.joblib")
    lgb_model = joblib.load(model_dir / "champion_lgb.joblib")

    _, _, _, _, X_test, y_test, _, _ = load_and_split_data(
        settings.feature_store_path / "features_v1.parquet"
    )

    # ROC / PR for champion (higher test AUC)
    if metadata["lgb_test_metrics"]["ROC-AUC"] >= metadata["lr_test_metrics"]["ROC-AUC"]:
        champion, champ_name = lgb_model, "LightGBM"
        threshold = metadata["lgb_threshold"]
    else:
        champion, champ_name = lr_model, "Logistic Regression"
        threshold = metadata["lr_threshold"]

    y_prob = champion.predict_proba(X_test)[:, 1]
    plot_roc(y_test, y_prob, reports_dir / "roc_curve.png", f"ROC Curve — {champ_name} (2018 Test)")
    plot_pr(y_test, y_prob, reports_dir / "pr_curve.png", f"PR Curve — {champ_name} (2018 Test)")
    print(f"  Saved: roc_curve.png, pr_curve.png")

    shap_csv = model_dir / "shap_importance.csv"
    if shap_csv.exists():
        plot_shap_importance(shap_csv, reports_dir / "shap_importance.png")
        print("  Saved: shap_importance.png")

    test_full = pd.read_parquet(settings.feature_store_path / "features_v1.parquet")
    test_full = test_full[test_full["issue_year"] == 2018].copy()
    feature_df = test_full.drop(columns=["target", "issue_year", "emp_title"], errors="ignore")
    bucket_summary = plot_bucket_default_rates(test_full, reports_dir / "risk_bucket_default_rates.png")
    print("  Saved: risk_bucket_default_rates.png + .csv")

    scorer = RiskLensScorer(use_shap=False)
    feature_df = test_full.drop(columns=["target", "issue_year", "emp_title"], errors="ignore")
    scored = scorer.score_batch(feature_df)
    plot_portfolio_distribution(scored, reports_dir / "portfolio_risk_distribution.png")
    print("  Saved: portfolio_risk_distribution.png")

    comparison = {
        "logistic_regression": metadata["lr_test_metrics"],
        "lightgbm": metadata["lgb_test_metrics"],
        "champion": champ_name,
        "champion_threshold": threshold,
        "bucket_summary": bucket_summary.to_dict(orient="records"),
    }
    (reports_dir / "model_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print("  Saved: model_comparison.json")

    print("\n[OK] Step 8 complete. Charts are in reports/\n")


if __name__ == "__main__":
    main()
