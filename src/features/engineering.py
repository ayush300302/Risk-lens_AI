"""
RiskLens AI — Feature Engineering (Simple & Explainable)
=========================================================
Creates features from cleaned LendingClub data.

Design: Every feature is a SIMPLE RATIO or COMBINATION.
You should be able to explain each one in ONE sentence.

Interview tip:
    "I created ratio features because they normalize raw numbers.
    For example, loan_amnt alone doesn't tell you much — but
    loan_amnt / annual_inc tells you how big the loan is
    relative to what the person earns."
"""

import polars as pl
from pathlib import Path
from typing import Optional

from app_config.logging_config import get_logger
from src.utils.memory import MemoryProfiler

logger = get_logger(__name__)


# ============================================================
# FEATURE DEFINITIONS — Keep it simple!
# ============================================================
# Each feature:
#   - Name: descriptive
#   - Formula: one-line math
#   - Why: one sentence
# ============================================================

FEATURE_DOCS = {
    "income_to_loan": "annual_inc / loan_amnt → higher = safer borrower",
    "loan_pct_of_income": "loan_amnt / annual_inc * 100 → what % of income is the loan",
    "monthly_debt_dollar": "dti/100 * annual_inc / 12 → actual monthly debt in dollars",
    "credit_util_x_balance": "revol_util * revol_bal / 10000 → combines usage rate with amount",
    "delinq_score": "delinq_2yrs*2 + pub_rec*3 → past bad behavior, weighted",
    "inquiry_score": "inq_last_6mths * 2 → recent credit-seeking activity",
    "open_acct_ratio": "open_acc / total_acc → fraction of accounts still active",
    "balance_vs_income": "tot_cur_bal / annual_inc → total debt relative to income",
    "credit_age_group": "credit_history_months bucketed into 4 groups",
    "revol_bal_vs_limit": "revol_bal / total_rev_hi_lim → another utilization measure",
}


def engineer_features(
    input_path: str | Path,
    output_path: str | Path,
    profiler: Optional[MemoryProfiler] = None,
) -> dict:
    """
    Create features from cleaned data. Writes features.parquet.

    Simple approach:
        1. Load cleaned parquet
        2. Add 10 ratio/combination features
        3. Save
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting feature engineering: {input_path}")

    track = profiler.track if profiler else _noop_context

    with track("load_cleaned_data"):
        df = pl.read_parquet(input_path)
        logger.info(f"Loaded: {len(df):,} rows x {df.width} columns")

    with track("create_features"):
        df = _add_features(df)
        logger.info(f"After features: {len(df):,} rows x {df.width} columns")

    with track("drop_temp_columns"):
        cols_to_drop = [c for c in df.columns if c.startswith("_")]
        if cols_to_drop:
            df = df.drop(cols_to_drop)

    with track("write_features"):
        df.write_parquet(output_path)
        file_size = output_path.stat().st_size / (1024 ** 2)
        logger.info(f"Wrote features: {output_path} ({file_size:.1f} MB)")

    new_features = [c for c in df.columns if c in FEATURE_DOCS or c.endswith("_ratio") or c.endswith("_score")]
    stats = {
        "rows": len(df),
        "total_features": df.width - 1,  # minus target
        "new_features": len(new_features),
        "file_size_mb": round(file_size, 1),
        "columns": df.columns,
    }

    return stats


def _add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add all engineered features. Each one is simple math."""

    # 1. Income to Loan ratio
    #    = annual_inc / loan_amnt
    #    "How many times your income covers the loan"
    df = df.with_columns(
        (pl.col("annual_inc") / pl.col("loan_amnt").clip(1))
        .cast(pl.Float32)
        .alias("income_to_loan")
    )

    # 2. Loan as % of income
    #    = loan_amnt / annual_inc * 100
    #    "What fraction of your yearly income is this loan"
    df = df.with_columns(
        (pl.col("loan_amnt") / pl.col("annual_inc").clip(1) * 100.0)
        .cast(pl.Float32)
        .alias("loan_pct_of_income")
    )

    # 3. Monthly debt in dollars
    #    = (dti / 100) * annual_inc / 12
    #    "How much you actually pay per month for all debts"
    df = df.with_columns(
        ((pl.col("dti") / 100.0) * pl.col("annual_inc") / 12.0)
        .cast(pl.Float32)
        .alias("monthly_debt_dollar")
    )

    # 4. Credit utilization × balance
    #    = revol_util * revol_bal / 10000
    #    "High usage + high balance = more risk"
    df = df.with_columns(
        (pl.col("revol_util") * pl.col("revol_bal") / 10000.0)
        .cast(pl.Float32)
        .alias("credit_util_x_balance")
    )

    # 5. Delinquency score (past bad behavior)
    #    = delinq_2yrs * 2 + pub_rec * 3
    #    "More past problems = higher score = more risky"
    delinq_expr = pl.lit(0.0)
    if "delinq_2yrs" in df.columns:
        delinq_expr = delinq_expr + pl.col("delinq_2yrs").fill_null(0) * 2.0
    if "pub_rec" in df.columns:
        delinq_expr = delinq_expr + pl.col("pub_rec").fill_null(0) * 3.0
    if "num_accts_ever_120_pd" in df.columns:
        delinq_expr = delinq_expr + pl.col("num_accts_ever_120_pd").fill_null(0)
    df = df.with_columns(delinq_expr.cast(pl.Float32).alias("delinq_score"))

    # 6. Inquiry score (credit-seeking behavior)
    #    = inq_last_6mths * 2
    #    "Lots of recent inquiries = desperately seeking credit"
    if "inq_last_6mths" in df.columns:
        expr = pl.col("inq_last_6mths").fill_null(0) * 2.0
        if "inq_last_12m" in df.columns:
            expr = expr + pl.col("inq_last_12m").fill_null(0)
        df = df.with_columns(expr.cast(pl.Float32).alias("inquiry_score"))

    # 7. Open account ratio
    #    = open_acc / total_acc
    #    "What fraction of all your accounts are still open"
    if "open_acc" in df.columns and "total_acc" in df.columns:
        df = df.with_columns(
            (pl.col("open_acc") / pl.col("total_acc").clip(1))
            .cast(pl.Float32)
            .alias("open_acct_ratio")
        )

    # 8. Balance vs income
    #    = tot_cur_bal / annual_inc
    #    "Total debt compared to what you earn"
    if "tot_cur_bal" in df.columns:
        df = df.with_columns(
            (pl.col("tot_cur_bal") / pl.col("annual_inc").clip(1))
            .cast(pl.Float32)
            .alias("balance_vs_income")
        )

    # 9. Credit age group (bucketed)
    #    Thin file (<5 yr) / Established (5-15 yr) / Seasoned (>15 yr)
    if "credit_history_months" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("credit_history_months") < 60)
            .then(pl.lit(0))       # thin file
            .when(pl.col("credit_history_months") < 120)
            .then(pl.lit(1))       # building
            .when(pl.col("credit_history_months") < 180)
            .then(pl.lit(2))       # established
            .otherwise(pl.lit(3))  # seasoned
            .cast(pl.Int8)
            .alias("credit_age_group")
        )

    # 10. Revolving balance to limit
    #     = revol_bal / total_rev_hi_lim
    #     "How much of your credit limit are you using"
    if "total_rev_hi_lim" in df.columns:
        df = df.with_columns(
            (pl.col("revol_bal") / pl.col("total_rev_hi_lim").clip(1))
            .cast(pl.Float32)
            .alias("revol_bal_vs_limit")
        )

    return df


def print_feature_docs():
    """Print what each feature means (for interview prep)."""
    print("\n" + "=" * 60)
    print("  ENGINEERED FEATURES — Quick Reference")
    print("=" * 60)
    for name, desc in FEATURE_DOCS.items():
        print(f"  {name:<25} {desc}")
    print("=" * 60 + "\n")


from contextlib import contextmanager

@contextmanager
def _noop_context(name: str = ""):
    """No-op context manager when profiler is not provided."""
    yield
