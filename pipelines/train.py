"""
RiskLens AI - Model Training Pipeline Orchestrator
===================================================
Runs: Features Parquet -> Load/Split -> Train Baseline -> Train Champion -> Compare -> Save

Usage:
    C:\\Users\\siddp\\AppData\\Local\\Programs\\Python\\Python314\\python.exe pipelines\\train.py
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
from src.utils.memory import MemoryProfiler
from src.models.training import (
    load_and_split_data, train_logistic_regression,
    train_lightgbm, evaluate_model
)
import joblib

setup_logging()
logger = get_logger(__name__)


def main():
    print("\n" + "=" * 70)
    print("  RiskLens AI -- Model Training Pipeline")
    print("  Baseline (Logistic Regression) vs Champion (LightGBM)")
    print("=" * 70 + "\n")

    settings = get_settings()
    settings.ensure_directories()
    profiler = MemoryProfiler()

    # Paths
    features_parquet = settings.feature_store_path / "features_v1.parquet"
    model_dir = settings.model_store_path
    
    if not features_parquet.exists():
        print(f"[ERROR] Features file not found: {features_parquet}")
        sys.exit(1)

    # 1. Load and Split Data
    print("-" * 70)
    print("STEP 1: Loading and Splitting Data (Temporal Split)")
    print("-" * 70)
    with profiler.track("load_and_split"):
        X_train, y_train, X_val, y_val, X_test, y_test, numerical_cols, categorical_cols = load_and_split_data(
            features_parquet
        )
    print(f"  Train: {len(X_train):>10,} rows (2007-2016)")
    print(f"  Val:   {len(X_val):>10,} rows (2017)")
    print(f"  Test:  {len(X_test):>10,} rows (2018)")
    print(f"  Default rate (train): {y_train.mean()*100:.1f}%")
    print(f"  Features: {len(numerical_cols)} numerical + {len(categorical_cols)} categorical = {len(numerical_cols)+len(categorical_cols)} total\n")

    # 2. Train Logistic Regression Baseline
    print("-" * 70)
    print("STEP 2: Training Logistic Regression Baseline")
    print("-" * 70)
    with profiler.track("train_logistic_regression"):
        lr_model, lr_val_metrics, lr_threshold = train_logistic_regression(
            X_train, y_train, X_val, y_val, numerical_cols, categorical_cols, settings.random_seed
        )
    print(f"  Optimal threshold: {lr_threshold:.4f}")
    print(f"  Val ROC-AUC: {lr_val_metrics['ROC-AUC']:.4f}\n")

    # 3. Train LightGBM Champion
    print("-" * 70)
    print("STEP 3: Training LightGBM Champion")
    print("-" * 70)
    with profiler.track("train_lightgbm"):
        lgb_model, lgb_val_metrics, lgb_threshold = train_lightgbm(
            X_train, y_train, X_val, y_val, numerical_cols, categorical_cols, settings.random_seed
        )
    print(f"  Optimal threshold: {lgb_threshold:.4f}")
    print(f"  Val ROC-AUC: {lgb_val_metrics['ROC-AUC']:.4f}\n")

    # 4. Evaluate both on Test Set (2018 Data)
    print("-" * 70)
    print("STEP 4: Evaluating on OUT-OF-TIME TEST SET (2018)")
    print("-" * 70)
    
    lr_test_metrics = evaluate_model(lr_model, X_test, y_test, "LR Baseline (Test)", threshold=lr_threshold)
    lgb_test_metrics = evaluate_model(lgb_model, X_test, y_test, "LightGBM Champion (Test)", threshold=lgb_threshold)

    # 5. Print Comparison Table
    print("\n" + "=" * 70)
    print("         MODEL PERFORMANCE COMPARISON (TEST SET - 2018)")
    print("=" * 70)
    print(f"  {'Metric':<25} | {'Baseline (LR)':>15} | {'Champion (LGB)':>15} | {'Delta':>10}")
    print("-" * 70)
    
    for metric_name in lr_test_metrics.keys():
        lr_val = lr_test_metrics[metric_name]
        lgb_val = lgb_test_metrics[metric_name]
        delta = round(lgb_val - lr_val, 4)
        delta_str = f"+{delta}" if delta > 0 else f"{delta}"
        
        if metric_name == "KS-Stat (%)":
            print(f"  {metric_name:<25} | {lr_val:>14.2f}% | {lgb_val:>14.2f}% | {delta_str:>10}")
        elif metric_name == "Threshold":
            print(f"  {metric_name:<25} | {lr_val:>15.4f} | {lgb_val:>15.4f} | {'':>10}")
        else:
            print(f"  {metric_name:<25} | {lr_val:>15.4f} | {lgb_val:>15.4f} | {delta_str:>10}")
            
    print("=" * 70 + "\n")

    # 6. Save Model Artifacts
    print("-" * 70)
    print("STEP 5: Saving Model Artifacts")
    print("-" * 70)
    with profiler.track("save_artifacts"):
        model_dir.mkdir(parents=True, exist_ok=True)
        
        lr_path = model_dir / "baseline_lr.joblib"
        lgb_path = model_dir / "champion_lgb.joblib"
        
        joblib.dump(lr_model, lr_path)
        joblib.dump(lgb_model, lgb_path)
        
        metadata = {
            "numerical_features": numerical_cols,
            "categorical_features": categorical_cols,
            "all_features": list(X_train.columns),
            "lr_test_metrics": lr_test_metrics,
            "lgb_test_metrics": lgb_test_metrics,
            "lr_threshold": lr_threshold,
            "lgb_threshold": lgb_threshold,
        }
        joblib.dump(metadata, model_dir / "model_metadata.joblib")
        
        print(f"  Saved LR Baseline:    {lr_path}")
        print(f"  Saved LGB Champion:   {lgb_path}")
        print(f"  Saved Metadata:       {model_dir / 'model_metadata.joblib'}\n")
        
    # 7. Generate SHAP explanations
    print("-" * 70)
    print("STEP 6: Generating Global SHAP Explanations")
    print("-" * 70)
    with profiler.track("generate_shap_explanations"):
        from src.explainability.shap_explainer import generate_global_explanations
        shap_csv = model_dir / "shap_importance.csv"
        shap_df = generate_global_explanations(
            model_pipeline_path=lgb_path,
            test_features_path=features_parquet,
            output_csv_path=shap_csv,
            sample_size=2000
        )
        print(f"\n  Top 10 Most Important Features (by SHAP):")
        for i, row in shap_df.head(10).iterrows():
            print(f"    {i+1:>2}. {row['feature']:<30} SHAP={row['mean_abs_shap']:.4f}")
        print()

    # Summary
    profiler.print_report()
    print("=" * 70)
    print("  [OK] TRAINING PIPELINE COMPLETE")
    print(f"  Champion model: LightGBM (ROC-AUC={lgb_test_metrics['ROC-AUC']:.4f})")
    print(f"  Models saved to: {model_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
