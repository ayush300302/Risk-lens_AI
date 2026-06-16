"""
RiskLens AI — Full Pipeline Orchestrator
=========================================
Runs every step in order. Skips steps when outputs already exist.

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --force-train
    python scripts/run_all.py --skip-setup
"""

import sys
import io
import argparse
import subprocess
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

project_root = Path(__file__).parent.parent
PYTHON = sys.executable


def run_step(name: str, script: str, skip: bool = False) -> None:
    if skip:
        print(f"[SKIP] {name}")
        return
    print(f"\n{'=' * 60}\n  Running: {name}\n{'=' * 60}")
    result = subprocess.run([PYTHON, str(project_root / script)], cwd=project_root)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-setup", action="store_true", help="Skip CSV→Parquet if parquet exists")
    parser.add_argument("--force-train", action="store_true", help="Re-run model training")
    parser.add_argument("--skip-reports", action="store_true", help="Skip chart generation")
    args = parser.parse_args()

    raw_parquet = project_root / "data" / "processed" / "loan_raw.parquet"
    features = project_root / "data" / "features" / "features_v1.parquet"
    models = project_root / "data" / "models" / "model_metadata.joblib"

    print("\n" + "=" * 60)
    print("  RiskLens AI — Full Pipeline")
    print("=" * 60)

    run_step("Step 0: CSV → Parquet", "scripts/setup_db.py", skip=args.skip_setup and raw_parquet.exists())
    run_step("Step 1-2: Clean + Features", "scripts/run_pipeline.py")
    run_step("Step 3-6: Train Models", "pipelines/train.py", skip=models.exists() and not args.force_train)
    run_step("Step 7: Score Demo", "scripts/score_demo.py", skip=False)
    run_step("Step 8: Reports", "scripts/generate_reports.py", skip=args.skip_reports)
    run_step("Step 9: Portfolio Analysis", "scripts/portfolio_analysis.py", skip=args.skip_reports)

    print("\n" + "=" * 60)
    print("  [OK] ALL STEPS COMPLETE")
    print("  Demo UI:  streamlit run app/streamlit_app.py")
    print("  Reports:  reports/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
