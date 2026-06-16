"""
RiskLens AI — Step 0: Raw Data Ingest
======================================
Converts loan.csv → loan_raw.parquet for the cleaning pipeline.

Usage:
    python scripts/setup_db.py

Why this step exists:
    Parquet is columnar, compressed, and typed — cleaning 2M+ rows from CSV
    repeatedly would be slow. Run once, reuse loan_raw.parquet everywhere.
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

import polars as pl

from app_config.settings import get_settings
from app_config.logging_config import setup_logging, get_logger
from src.utils.memory import MemoryProfiler

setup_logging()
logger = get_logger(__name__)


def export_csv_to_parquet(csv_path: Path, output_path: Path) -> dict:
    """Stream CSV to Parquet with Polars (memory-efficient on 1GB+ files)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    logger.info(f"Reading CSV: {csv_path}")
    row_count = (
        pl.scan_csv(csv_path, infer_schema_length=10_000, ignore_errors=True)
        .select(pl.len())
        .collect()
        .item()
    )

    logger.info(f"Exporting {row_count:,} rows to Parquet...")
    (
        pl.scan_csv(csv_path, infer_schema_length=10_000, ignore_errors=True)
        .sink_parquet(output_path)
    )

    verify = pl.read_parquet(output_path)
    file_mb = output_path.stat().st_size / (1024 ** 2)
    stats = {
        "rows": len(verify),
        "columns": verify.width,
        "file_size_mb": round(file_mb, 1),
        "output": str(output_path),
    }
    logger.info(f"Parquet export complete: {stats}")
    return stats


def main():
    print("\n" + "=" * 60)
    print("  RiskLens AI — Step 0: CSV → Parquet")
    print("=" * 60 + "\n")

    settings = get_settings()
    settings.ensure_directories()
    profiler = MemoryProfiler()

    csv_path = settings.project_root / "loan.csv"
    output_path = settings.processed_data_path / "loan_raw.parquet"

    if output_path.exists():
        print(f"[SKIP] Raw parquet already exists: {output_path}")
        print("       Delete it to re-export from CSV.\n")
        return

    with profiler.track("csv_to_parquet"):
        stats = export_csv_to_parquet(csv_path, output_path)

    print(f"  Rows:      {stats['rows']:>12,}")
    print(f"  Columns:   {stats['columns']:>12}")
    print(f"  Size:      {stats['file_size_mb']:>11} MB")
    print(f"  Output:    {stats['output']}")
    profiler.print_report()
    print("\n[OK] Step 0 complete. Next: python scripts/run_pipeline.py\n")


if __name__ == "__main__":
    main()
