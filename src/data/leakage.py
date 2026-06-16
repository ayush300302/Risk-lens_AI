"""
RiskLens AI — Data Leakage Detection Framework
================================================
Identifies and removes columns that leak future information.

What is Data Leakage?
    Data leakage happens when your model uses information that would NOT
    be available at the time of prediction. In credit risk:

    At prediction time (loan application):
        - You KNOW: income, employment, credit history, loan amount requested
        - You DON'T KNOW: whether they'll pay, how much they'll recover,
          if they'll enter hardship, what their last payment will be

    If the model trains on "total_pymnt" (total payments made), it learns
    that low payments = default. This is CHEATING — the model is seeing
    the answer during training.

Why Leakage is Dangerous:
    1. Model shows 99% AUC in development -> 55% AUC in production
    2. You deploy a model that seems perfect but is useless
    3. In interviews, leakage detection is a TOP question for ML roles

Categories of Leakage in LendingClub:
    1. POST-ORIGINATION: Columns that only exist after the loan is issued
       (total_pymnt, recoveries, last_pymnt_d, out_prncp)

    2. LENDER-ASSIGNED: Columns set by the lender based on risk assessment
       (grade, sub_grade, int_rate, installment)
       Using these is circular — the lender already used a risk model!

    3. HARDSHIP PROGRAM: Columns from post-default hardship programs
       (hardship_flag, hardship_amount, settlement_*)

    4. SETTLEMENT: Columns from debt settlement after default
       (settlement_status, settlement_amount, settlement_date)

Interview Question:
    "How do you detect data leakage in a credit risk model?"

Strong Answer:
    "I use a time-based test: For each feature, I ask 'Would this
    information be available at the moment a new loan application
    arrives?' If no, it's leakage. I categorize leakage into four
    types: post-origination metrics, lender-assigned grades,
    hardship program data, and settlement data. I also check for
    subtle leakage like funded_amnt (determined after approval) vs
    loan_amnt (requested by borrower)."
"""

from app_config.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================
# LEAKAGE COLUMN DEFINITIONS
# ============================================================
# Each category explains WHY these columns leak future information.
# This serves as both code AND documentation for interviews.
# ============================================================

LEAKAGE_CATEGORIES = {
    "post_origination": {
        "description": "Only known AFTER the loan is issued and payments begin",
        "risk": "Model learns outcome directly from payment behavior",
        "columns": [
            "out_prncp",                # Outstanding principal (changes over loan life)
            "out_prncp_inv",            # Outstanding principal for investors
            "total_pymnt",              # Total payments received (directly indicates default)
            "total_pymnt_inv",          # Total payments to investors
            "total_rec_prncp",          # Total principal received
            "total_rec_int",            # Total interest received
            "total_rec_late_fee",       # Late fees (only exist if late!)
            "last_pymnt_d",             # Last payment date
            "last_pymnt_amnt",          # Last payment amount
            "next_pymnt_d",             # Next payment date
        ],
    },

    "post_default": {
        "description": "Only exist AFTER a borrower defaults",
        "risk": "Directly reveals the target variable",
        "columns": [
            "recoveries",               # Money recovered after charge-off
            "collection_recovery_fee",   # Fees from recovery process
        ],
    },

    "lender_assigned": {
        "description": "Set by the lender BASED ON their own risk model",
        "risk": "Circular reasoning — using the lender's risk score to predict risk",
        "columns": [
            "grade",                     # LC's risk grade (A-G)
            "sub_grade",                 # LC's sub-grade (A1-G5)
            "int_rate",                  # Interest rate (set based on risk!)
            "installment",              # Monthly payment (calculated from rate)
            "funded_amnt",              # Amount actually funded (post-approval)
            "funded_amnt_inv",          # Amount funded by investors
        ],
    },

    "hardship_program": {
        "description": "From hardship programs entered AFTER loan origination",
        "risk": "Hardship enrollment happens only when borrower is struggling",
        "columns": [
            "hardship_flag",
            "hardship_type",
            "hardship_reason",
            "hardship_status",
            "deferral_term",
            "hardship_amount",
            "hardship_start_date",
            "hardship_end_date",
            "payment_plan_start_date",
            "hardship_length",
            "hardship_dpd",
            "hardship_loan_status",
            "orig_projected_additional_accrued_interest",
            "hardship_payoff_balance_amount",
            "hardship_last_payment_amount",
        ],
    },

    "settlement": {
        "description": "From debt settlement AFTER default",
        "risk": "Settlement only happens after default — directly reveals target",
        "columns": [
            "debt_settlement_flag",
            "debt_settlement_flag_date",
            "settlement_status",
            "settlement_date",
            "settlement_amount",
            "settlement_percentage",
            "settlement_term",
        ],
    },
}


def get_leakage_columns() -> list[str]:
    """
    Get flat list of all leakage columns.

    Returns:
        List of column names that must be removed before modeling.
    """
    all_leakage = []
    for category in LEAKAGE_CATEGORIES.values():
        all_leakage.extend(category["columns"])
    return all_leakage


def get_leakage_report() -> list[dict]:
    """
    Generate a detailed leakage report for documentation and interviews.

    Returns:
        List of dicts with column, category, description, and risk.
    """
    report = []
    for cat_name, cat_info in LEAKAGE_CATEGORIES.items():
        for col in cat_info["columns"]:
            report.append({
                "column": col,
                "category": cat_name,
                "description": cat_info["description"],
                "risk": cat_info["risk"],
            })
    return report


def print_leakage_report() -> None:
    """Print a formatted leakage report to console."""
    print("\n" + "=" * 70)
    print("  DATA LEAKAGE REPORT")
    print("=" * 70)

    total = 0
    for cat_name, cat_info in LEAKAGE_CATEGORIES.items():
        count = len(cat_info["columns"])
        total += count
        print(f"\n  [{cat_name.upper()}] ({count} columns)")
        print(f"  Why: {cat_info['description']}")
        print(f"  Risk: {cat_info['risk']}")
        print(f"  Columns: {', '.join(cat_info['columns'][:5])}")
        if count > 5:
            print(f"           + {count - 5} more...")

    print(f"\n  TOTAL LEAKAGE COLUMNS: {total}")
    print("=" * 70 + "\n")


def detect_leakage_in_dataframe(columns: list[str]) -> dict:
    """
    Check a DataFrame's columns for leakage.

    Args:
        columns: List of column names from a DataFrame.

    Returns:
        Dict with found leakage columns grouped by category.

    Usage:
        import polars as pl
        df = pl.read_parquet("data/processed/loan_raw.parquet")
        result = detect_leakage_in_dataframe(df.columns)
        print(f"Found {result['total_leakage']} leakage columns!")
    """
    leakage_found = {}
    total = 0

    for cat_name, cat_info in LEAKAGE_CATEGORIES.items():
        found = [c for c in cat_info["columns"] if c in columns]
        if found:
            leakage_found[cat_name] = found
            total += len(found)
            logger.warning(
                f"Leakage detected [{cat_name}]: {len(found)} columns",
                columns=found,
            )

    if total == 0:
        logger.info("No leakage columns detected - dataset is clean!")

    return {
        "total_leakage": total,
        "categories": leakage_found,
        "clean": total == 0,
    }
