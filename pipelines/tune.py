"""
RiskLens AI — Hyperparameter Tuning Pipeline
=============================================
Orchestrates the full hyperparameter search for both models and saves
the best parameters to disk so ``pipelines/train.py`` can re-use them.

Usage:
    # Quick smoke test (5 LGB trials, 10 LR combinations)
    python pipelines/tune.py --n-trials 5 --lr-n-iter 10

    # Full production search (recommended: 50 LGB trials, 30 LR combinations)
    python pipelines/tune.py --n-trials 50 --lr-n-iter 30

    # Tune LightGBM only (skip LR tuning)
    python pipelines/tune.py --skip-lr

    # Tune Logistic Regression only (skip LGB tuning)
    python pipelines/tune.py --skip-lgb

Output:
    data/model_store/best_params.joblib   — loaded automatically by train.py
    data/model_store/optuna_study.joblib  — Optuna study object for analysis

Runtime estimate:
    LR (30 iter, 5-fold):  ~2-5 min
    LGB (50 trials):       ~15-45 min depending on hardware
"""

import sys
import io
import argparse
import json
import time
from pathlib import Path

# Fix Windows encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app_config.settings import get_settings
from app_config.logging_config import setup_logging, get_logger
from src.models.training import load_and_split_data
from src.models.tuning import tune_logistic_regression, tune_lightgbm
import joblib

setup_logging()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RiskLens AI — Hyperparameter Tuning Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n-trials", type=int, default=50,
        help="Number of Optuna trials for LightGBM (default: 50)",
    )
    parser.add_argument(
        "--lr-n-iter", type=int, default=30,
        help="Number of RandomizedSearchCV iterations for LR (default: 30)",
    )
    parser.add_argument(
        "--lr-cv", type=int, default=5,
        help="Number of CV folds for LR tuning (default: 5)",
    )
    parser.add_argument(
        "--skip-lr", action="store_true",
        help="Skip Logistic Regression tuning",
    )
    parser.add_argument(
        "--skip-lgb", action="store_true",
        help="Skip LightGBM tuning",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def _section(text: str) -> None:
    print("\n" + "-" * 70)
    print(f"  {text}")
    print("-" * 70)


def _fmt_params(params: dict) -> str:
    """Pretty-print a params dict as a code snippet."""
    lines = ["{"]
    for k, v in params.items():
        if isinstance(v, float):
            lines.append(f'    "{k}": {v:.6g},')
        else:
            lines.append(f'    "{k}": {v!r},')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    _banner("RiskLens AI -- Hyperparameter Tuning Pipeline")
    print(f"  LGB Optuna trials  : {args.n_trials}")
    print(f"  LR RandomSearch    : {args.lr_n_iter} iter, {args.lr_cv}-fold CV")
    print(f"  Random seed        : {args.seed}")
    print(f"  Skip LR tuning     : {args.skip_lr}")
    print(f"  Skip LGB tuning    : {args.skip_lgb}")

    settings = get_settings()
    settings.ensure_directories()

    features_parquet = settings.feature_store_path / "features_v1.parquet"
    model_dir = settings.model_store_path
    best_params_path = model_dir / "best_params.joblib"
    study_path = model_dir / "optuna_study.joblib"

    if not features_parquet.exists():
        print(f"\n[ERROR] Features file not found: {features_parquet}")
        print("  Run the feature engineering pipeline first:")
        print("  python pipelines/run_pipeline.py")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1 — Load and split data
    # ------------------------------------------------------------------
    _section("STEP 1: Loading and Splitting Data (Temporal Split)")
    t0 = time.time()
    (
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        numerical_cols, categorical_cols,
    ) = load_and_split_data(features_parquet)
    elapsed = time.time() - t0
    print(f"  Train : {len(X_train):>10,} rows (2007-2016)")
    print(f"  Val   : {len(X_val):>10,} rows (2017)   <- tuning target")
    print(f"  Test  : {len(X_test):>10,} rows (2018)   <- held out, never touched here")
    print(f"  Features: {len(numerical_cols)} numerical + {len(categorical_cols)} categorical")
    print(f"  Default rate (train): {y_train.mean() * 100:.1f}%")
    print(f"  Data loaded in {elapsed:.1f}s")

    # Container for all best params
    best_params_all: dict = {}

    # ------------------------------------------------------------------
    # Step 2 — Tune Logistic Regression
    # ------------------------------------------------------------------
    if not args.skip_lr:
        _section(f"STEP 2: Tuning Logistic Regression  (RandomizedSearchCV, {args.lr_n_iter} iter, {args.lr_cv}-fold)")
        print("  Searching over: C (log-uniform), solver, penalty, max_iter")
        print("  Scoring metric: ROC-AUC\n")
        t1 = time.time()

        lr_best_params = tune_logistic_regression(
            X_train, y_train,
            X_val,   y_val,
            numerical_cols, categorical_cols,
            n_iter=args.lr_n_iter,
            cv=args.lr_cv,
            random_seed=args.seed,
        )
        lr_elapsed = time.time() - t1

        best_params_all["lr"] = lr_best_params
        print(f"\n  [OK] LR tuning complete in {lr_elapsed:.1f}s")
        print(f"  Best LR params:")
        for k, v in lr_best_params.items():
            print(f"    {k:20s} = {v!r}")
    else:
        print("\n  [SKIP] LR tuning skipped (--skip-lr)")
        lr_best_params = None

    # ------------------------------------------------------------------
    # Step 3 — Tune LightGBM (Optuna)
    # ------------------------------------------------------------------
    if not args.skip_lgb:
        _section(f"STEP 3: Tuning LightGBM  (Optuna TPE, {args.n_trials} trials)")
        print("  Search space: n_estimators, learning_rate, num_leaves,")
        print("                max_depth, min_child_samples, subsample,")
        print("                colsample_bytree, reg_alpha, reg_lambda")
        print("  Objective:    Maximize ROC-AUC on 2017 validation set")
        print(f"  Trials:       {args.n_trials}  (use --n-trials N to change)\n")

        t2 = time.time()
        lgb_best_params, lgb_best_auc, study = tune_lightgbm(
            X_train, y_train,
            X_val,   y_val,
            numerical_cols, categorical_cols,
            n_trials=args.n_trials,
            random_seed=args.seed,
            show_progress_bar=True,
        )
        lgb_elapsed = time.time() - t2

        best_params_all["lgb"] = lgb_best_params

        print(f"\n  [OK] LGB tuning complete in {lgb_elapsed:.1f}s")
        print(f"  Best LGB Val ROC-AUC : {lgb_best_auc:.4f}")
        print(f"  Best LGB params:")
        for k, v in lgb_best_params.items():
            if isinstance(v, float):
                print(f"    {k:25s} = {v:.6g}")
            else:
                print(f"    {k:25s} = {v!r}")

        # Save Optuna study for further analysis (e.g., importance plots)
        joblib.dump(study, study_path)
        print(f"\n  Optuna study saved to: {study_path}")
    else:
        print("\n  [SKIP] LGB tuning skipped (--skip-lgb)")
        lgb_best_params = None

    # ------------------------------------------------------------------
    # Step 4 — Save best params
    # ------------------------------------------------------------------
    _section("STEP 4: Saving Best Hyperparameters")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_params_all, best_params_path)
    print(f"  Saved best params to: {best_params_path}")

    # ------------------------------------------------------------------
    # Summary banner
    # ------------------------------------------------------------------
    _banner("TUNING COMPLETE — Summary")

    if "lr" in best_params_all:
        print("\n  Logistic Regression best params:")
        print("  " + _fmt_params(best_params_all["lr"]).replace("\n", "\n  "))

    if "lgb" in best_params_all:
        print(f"\n  LightGBM best params (val ROC-AUC={lgb_best_auc:.4f}):")
        print("  " + _fmt_params(best_params_all["lgb"]).replace("\n", "\n  "))

    print(f"""
  Next steps:
    1. Run the training pipeline to use these params:
       python pipelines/train.py

    2. train.py will automatically detect and load:
       {best_params_path}
""")


if __name__ == "__main__":
    main()
