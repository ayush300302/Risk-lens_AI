"""
RiskLens AI — Step 7: Score a Loan Application (Demo CLI)
==========================================================
Demonstrates the full decision path:
  features → PD → risk bucket → policy → SHAP

Usage:
    python scripts/score_demo.py
    python scripts/score_demo.py --no-shap
"""

import sys
import io
import json
import argparse
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app_config.logging_config import setup_logging
from src.scoring.predictor import RiskLensScorer

setup_logging()


def main():
    parser = argparse.ArgumentParser(description="Score a sample 2018 loan application")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP (faster)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  RiskLens AI — Loan Scoring Demo")
    print("=" * 60 + "\n")

    scorer = RiskLensScorer(use_shap=not args.no_shap)
    sample = RiskLensScorer.load_sample_from_test_set(n=1, issue_year=2018)

    print(f"  Model:     {scorer.model_name}")
    print(f"  Threshold: {scorer.threshold:.4f}")
    print("  Sample:    1 loan from 2018 test set\n")

    result = scorer.score_application(sample, include_shap=not args.no_shap)

    print("-" * 60)
    print(f"  Probability of Default: {result['probability_of_default']:.2%}")
    print(f"  Predicted Default:      {result['predicted_default']}")
    print(f"  Risk Tier:              {result['risk_bucket']['risk_tier']} ({result['risk_bucket']['credit_grade']})")
    print(f"  Recommended Rate:       {result['risk_bucket']['recommended_interest_rate']}")
    print(f"  Bucket Action:          {result['risk_bucket']['recommended_action']}")
    print(f"  Policy Decision:        {result['policy_decision']['decision']}")
    print(f"  Comments:               {result['policy_decision']['comments']}")

    if result["policy_decision"]["triggered_rules"]:
        print("\n  Triggered Policy Rules:")
        for rule in result["policy_decision"]["triggered_rules"]:
            print(f"    - {rule}")

    if result.get("shap_top_factors"):
        print("\n  Top SHAP Risk Factors:")
        for factor in result["shap_top_factors"]:
            print(f"    - {factor['feature']}: {factor['impact']} (SHAP={factor['shap_value']:.4f})")

    print("\n" + "-" * 60)
    print("  Full JSON output:")
    print(json.dumps(result, indent=2, default=str))
    print("\n[OK] Step 7 demo complete.\n")


if __name__ == "__main__":
    main()
