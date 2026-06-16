"""
RiskLens AI — Step 9: Portfolio Analysis (Bonus)
=================================================
Computes portfolio-level risk metrics on the 2018 test set:
  - Exposure by risk tier
  - Simplified Expected Loss (EL = PD × LGD × EAD)
  - Policy decision breakdown

Usage:
    python scripts/portfolio_analysis.py
"""

import sys
import io
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from app_config.settings import get_settings
from app_config.logging_config import setup_logging, get_logger
from src.scoring.predictor import RiskLensScorer

setup_logging()
logger = get_logger(__name__)

# Industry-standard simplification for retail unsecured: LGD ~ 60%
DEFAULT_LGD = 0.60


def run_portfolio_analysis(test_df: pd.DataFrame, lgd: float = DEFAULT_LGD) -> pd.DataFrame:
    """
    Portfolio view: group loans by risk tier and compute exposure + expected loss.

    EL formula (simplified):
        Expected Loss = PD × LGD × EAD
        EAD (Exposure at Default) ≈ loan_amnt
    """
    scorer = RiskLensScorer(use_shap=False)
    feature_cols = [c for c in test_df.columns if c not in ("target", "issue_year", "emp_title")]
    scored = scorer.score_batch(test_df[feature_cols])
    portfolio = pd.concat([test_df.reset_index(drop=True), scored], axis=1)

    portfolio["ead"] = portfolio["loan_amnt"].fillna(0)
    portfolio["expected_loss"] = portfolio["probability_of_default"] * lgd * portfolio["ead"]

    summary = (
        portfolio.groupby("risk_tier", observed=True)
        .agg(
            loan_count=("loan_amnt", "count"),
            total_exposure=("ead", "sum"),
            avg_pd=("probability_of_default", "mean"),
            actual_default_rate=("target", "mean"),
            total_expected_loss=("expected_loss", "sum"),
            approve_count=("policy_decision", lambda s: (s == "Approve").sum()),
            refer_count=("policy_decision", lambda s: (s == "Refer").sum()),
            deny_count=("policy_decision", lambda s: (s == "Deny").sum()),
        )
        .reset_index()
    )

    summary["pct_of_portfolio"] = (summary["loan_count"] / summary["loan_count"].sum() * 100).round(1)
    summary["pct_of_exposure"] = (summary["total_exposure"] / summary["total_exposure"].sum() * 100).round(1)
    return summary


def main():
    print("\n" + "=" * 60)
    print("  RiskLens AI — Step 9: Portfolio Analysis")
    print("=" * 60 + "\n")

    settings = get_settings()
    reports_dir = settings.reports_path
    reports_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_parquet(settings.feature_store_path / "features_v1.parquet")
    test_df = test_df[test_df["issue_year"] == 2018].copy()
    logger.info(f"Analyzing {len(test_df):,} loans from 2018 test set")

    summary = run_portfolio_analysis(test_df)
    out_csv = reports_dir / "portfolio_summary.csv"
    summary.to_csv(out_csv, index=False)

    total_el = summary["total_expected_loss"].sum()
    total_exposure = summary["total_exposure"].sum()

    print(f"  Total exposure:      ${total_exposure:,.0f}")
    print(f"  Total expected loss: ${total_el:,.0f}  (LGD={DEFAULT_LGD:.0%})")
    print(f"  Portfolio EL rate:   {total_el / total_exposure * 100:.2f}%")
    print()
    print(summary.to_string(index=False))
    print(f"\n  Saved: {out_csv}")
    print("\n[OK] Step 9 complete.\n")


if __name__ == "__main__":
    main()
