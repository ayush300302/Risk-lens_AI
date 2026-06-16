"""
RiskLens AI — Full Data Pipeline Orchestrator
===============================================
Runs: Raw Parquet -> Clean -> Leakage Removal -> Feature Engineering

Usage:
    python scripts/run_pipeline.py
"""

import sys
import io
from pathlib import Path

# Fix Windows encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app_config.settings import get_settings
from app_config.logging_config import setup_logging, get_logger
from src.utils.memory import MemoryProfiler, get_system_memory_info
from src.data.cleaning import run_cleaning_pipeline
from src.data.leakage import print_leakage_report
from src.features.engineering import engineer_features

setup_logging()
logger = get_logger(__name__)


def main():
    print("\n" + "=" * 60)
    print("  RiskLens AI -- Full Data Pipeline")
    print("  Raw -> Clean -> Features")
    print("=" * 60 + "\n")

    settings = get_settings()
    settings.ensure_directories()
    profiler = MemoryProfiler()

    # Paths
    raw_parquet = settings.processed_data_path / "loan_raw.parquet"
    cleaned_parquet = settings.processed_data_path / "loan_cleaned.parquet"
    features_parquet = settings.feature_store_path / "features_v1.parquet"

    # Verify raw parquet exists
    if not raw_parquet.exists():
        print(f"[ERROR] Raw parquet not found: {raw_parquet}")
        print("Run 'python scripts/setup_db.py' first to create it.")
        sys.exit(1)

    sys_info = get_system_memory_info()
    print(f"System RAM: {sys_info['total_gb']:.1f} GB total, {sys_info['available_gb']:.1f} GB available\n")

    # ---- Step 1: Leakage Report ----
    print_leakage_report()

    # ---- Step 2: Data Cleaning (Phase 3 + 4) ----
    print("-" * 60)
    print("STEP 1: Data Cleaning + Leakage Removal")
    print("-" * 60)

    clean_stats = run_cleaning_pipeline(
        input_path=raw_parquet,
        output_path=cleaned_parquet,
        profiler=profiler,
    )

    print(f"  Raw:     {clean_stats['raw_rows']:>12,} rows x {clean_stats['raw_columns']} cols")
    print(f"  Cleaned: {clean_stats['cleaned_rows']:>12,} rows x {clean_stats['cleaned_columns']} cols")
    print(f"  Removed: {clean_stats['rows_removed']:>12,} rows ({clean_stats['rows_removed_pct']}%)")
    print(f"  Defaults:    {clean_stats['default_count']:>8,} ({clean_stats['default_rate_pct']}%)")
    print(f"  Non-defaults:{clean_stats['non_default_count']:>8,}")
    print(f"  File size:   {clean_stats['file_size_mb']} MB")

    # ---- Step 3: Feature Engineering (Phase 6) ----
    print("\n" + "-" * 60)
    print("STEP 2: Feature Engineering")
    print("-" * 60)

    feat_stats = engineer_features(
        input_path=cleaned_parquet,
        output_path=features_parquet,
        profiler=profiler,
    )

    print(f"  Features:    {feat_stats['total_features']} columns")
    print(f"  Rows:        {feat_stats['rows']:,}")
    print(f"  File size:   {feat_stats['file_size_mb']} MB")

    # ---- Memory Report ----
    profiler.print_report()

    # ---- Summary ----
    print("=" * 60)
    print("  [OK] PIPELINE COMPLETE")
    print(f"  Features file: {features_parquet}")
    print(f"  Ready for model training (Phase 7)")
    print("=" * 60 + "\n")

    return {"cleaning": clean_stats, "features": feat_stats}


if __name__ == "__main__":
    main()
