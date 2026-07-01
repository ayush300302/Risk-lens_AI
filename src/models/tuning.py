"""
RiskLens AI — Hyperparameter Tuning Module
===========================================
Provides two tuning strategies:

1. Logistic Regression — RandomizedSearchCV (scikit-learn)
   - Fast and sufficient for a simple linear model
   - Searches over C, solver, and max_iter
   - Uses 5-fold stratified CV on the training set

2. LightGBM — Optuna (Bayesian / TPE optimization)
   - Smarter than grid/random for large search spaces
   - Evaluates directly on the held-out 2017 validation set (no CV overhead)
   - Searches over n_estimators, learning_rate, num_leaves, depth, regularisation, etc.

Usage:
    from src.models.tuning import tune_logistic_regression, tune_lightgbm

    lr_best  = tune_logistic_regression(X_train, y_train, X_val, y_val, num_cols, cat_cols)
    lgb_best = tune_lightgbm(X_train, y_train, X_val, y_val, num_cols, cat_cols, n_trials=50)

Interview tip:
    "For Logistic Regression I use RandomizedSearchCV because the model is cheap
    to train and CV variance is low. For LightGBM I use Optuna's TPE sampler —
    it builds a probabilistic model of which hyperparameters are likely to be good
    and focuses trials there, so it finds better params in far fewer evaluations
    than random search."
"""

import warnings
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import loguniform

from app_config.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cast_lgb_dtypes(
    X: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    """Coerce dtypes so LightGBM handles categoricals natively."""
    df = X[numerical_cols + categorical_cols].copy()
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float32)
    for col in categorical_cols:
        df[col] = df[col].astype(str).astype("category")
    return df


def _build_lr_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
    random_seed: int,
    sparse_ohe: bool = False,
) -> Pipeline:
    """Build a fresh LR preprocessing + classifier pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            # Keep sparse=True during CV to avoid OOM on large datasets.
            # LogisticRegression handles sparse matrices natively.
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_ohe), categorical_cols),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=random_seed,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Logistic Regression — RandomizedSearchCV
# ---------------------------------------------------------------------------

def tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numerical_cols: List[str],
    categorical_cols: List[str],
    n_iter: int = 30,
    cv: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Tune Logistic Regression using RandomizedSearchCV.

    Combines train + val sets for cross-validation so that every data point
    gets used in the search without leaking the final test set.

    Parameters
    ----------
    n_iter : int
        Number of random parameter combinations to try. Default 30.
    cv : int
        Number of stratified CV folds. Default 5.
    random_seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict
        Best hyperparameters for the classifier step, ready to be passed
        to `train_logistic_regression(best_params=...)`.

    Example best_params output:
        {"C": 0.42, "solver": "lbfgs", "penalty": "l2", "max_iter": 1000}
    """
    logger.info(f"Starting LR RandomizedSearchCV (n_iter={n_iter}, cv={cv})...")

    # -----------------------------------------------------------------------
    # Memory-safe strategy for large datasets:
    # OneHotEncoder on 1M+ rows blows up RAM when densified inside CV workers.
    # Solution: use a stratified 10% sample for the CV search, which still
    # gives reliable ROC-AUC estimates while fitting in memory.
    # The final model in train.py is always trained on the full dataset.
    # -----------------------------------------------------------------------
    MAX_CV_ROWS = 150_000  # ~150K rows keeps OHE dense matrix < 100 MB
    X_cv_all = pd.concat([X_train, X_val], ignore_index=True)
    y_cv_all = pd.concat([y_train, y_val], ignore_index=True)

    if len(X_cv_all) > MAX_CV_ROWS:
        frac = MAX_CV_ROWS / len(X_cv_all)
        logger.info(
            f"Dataset too large for dense CV ({len(X_cv_all):,} rows). "
            f"Using stratified {frac*100:.0f}% sample ({MAX_CV_ROWS:,} rows) for LR search."
        )
        from sklearn.model_selection import train_test_split
        X_cv, _, y_cv, _ = train_test_split(
            X_cv_all, y_cv_all,
            train_size=MAX_CV_ROWS,
            stratify=y_cv_all,
            random_state=random_seed,
        )
    else:
        X_cv, y_cv = X_cv_all, y_cv_all

    logger.info(f"LR CV dataset size: {len(X_cv):,} rows")

    # Use sparse OHE during CV to avoid dense matrix OOM
    pipeline = _build_lr_pipeline(numerical_cols, categorical_cols, random_seed, sparse_ohe=True)

    # Search space — lbfgs only (saga is very slow on large datasets with many features)
    param_dist = {
        "classifier__C":        loguniform(0.001, 10),
        "classifier__solver":   ["lbfgs"],
        "classifier__penalty":  ["l2"],
        "classifier__max_iter": [500, 1000],
    }

    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_seed)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv_splitter,
        n_jobs=1,    # Use n_jobs=1 to avoid multiprocess RAM duplication on large OHE matrices
        random_state=random_seed,
        verbose=1,
        refit=False,
        error_score="raise",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_cv, y_cv)

    best_params_raw = search.best_params_
    logger.info(f"LR best CV ROC-AUC: {search.best_score_:.4f}")
    logger.info(f"LR best params (raw): {best_params_raw}")

    # Strip the "classifier__" prefix so params can be applied directly
    best_params = {
        k.replace("classifier__", ""): v
        for k, v in best_params_raw.items()
    }

    logger.info(f"LR best params (clean): {best_params}")
    return best_params


# ---------------------------------------------------------------------------
# LightGBM — Optuna Bayesian Optimization
# ---------------------------------------------------------------------------

def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    numerical_cols: List[str],
    categorical_cols: List[str],
    n_trials: int = 50,
    random_seed: int = 42,
    show_progress_bar: bool = True,
) -> Dict[str, Any]:
    """
    Tune LightGBM using Optuna's TPE (Tree-structured Parzen Estimator) sampler.

    Each Optuna trial trains a LightGBM model on X_train and evaluates it on X_val
    using ROC-AUC. This avoids expensive cross-validation for LGB while still
    producing a reliable estimate because we always evaluate on the unseen 2017 data.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials. 50 is a good default; use 20 for a quick test.
    random_seed : int
        Seed for both Optuna sampler and LightGBM.
    show_progress_bar : bool
        Show tqdm progress bar during the search.

    Returns
    -------
    dict
        Best hyperparameters ready to be passed to `train_lightgbm(best_params=...)`.

    Example best_params output:
        {
            "n_estimators": 450, "learning_rate": 0.03, "num_leaves": 63,
            "max_depth": 7, "min_child_samples": 80, "subsample": 0.75,
            "colsample_bytree": 0.8, "reg_alpha": 0.05, "reg_lambda": 2.0
        }
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError(
            "Optuna is required for LightGBM tuning. "
            "Install it with: pip install optuna>=3.0.0"
        )

    logger.info(f"Starting LightGBM Optuna tuning (n_trials={n_trials})...")

    # Precompute the class imbalance ratio (used in every trial)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = float(neg_count / pos_count)
    logger.info(f"Class imbalance ratio (neg/pos): {scale_pos_weight:.2f}")

    # Pre-cast dtypes once (expensive for large datasets)
    X_train_lgb = _cast_lgb_dtypes(X_train, numerical_cols, categorical_cols)
    X_val_lgb   = _cast_lgb_dtypes(X_val,   numerical_cols, categorical_cols)

    def objective(trial: "optuna.Trial") -> float:
        """Optuna objective: maximize validation ROC-AUC."""
        params = {
            "n_estimators":      trial.suggest_int("n_estimators",     100, 1000),
            "learning_rate":     trial.suggest_float("learning_rate",  0.01, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves",       20, 150),
            "max_depth":         trial.suggest_int("max_depth",        3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample":         trial.suggest_float("subsample",      0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha",      1e-8, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda",     1e-8, 10.0, log=True),
            # Fixed params
            "scale_pos_weight":  scale_pos_weight,
            "random_state":      random_seed,
            "n_jobs":            -1,
            "verbosity":         -1,
        }

        model = lgb.LGBMClassifier(**params)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(
                X_train_lgb,
                y_train.values,
                eval_set=[(X_val_lgb, y_val.values)],
                eval_metric="auc",
                categorical_feature=categorical_cols,
                callbacks=[lgb.log_evaluation(period=0)],
            )

        y_val_prob = model.predict_proba(X_val_lgb)[:, 1]
        roc_auc = roc_auc_score(y_val.values, y_val_prob)
        return roc_auc

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
        n_jobs=1,   # LGB itself uses all CPUs; outer loop is serial
    )

    best_trial = study.best_trial
    logger.info(f"LGB Optuna best val ROC-AUC: {best_trial.value:.4f} (trial #{best_trial.number})")
    logger.info(f"LGB best params: {best_trial.params}")

    # Return only the model hyperparameters (strip fixed params from params dict)
    best_params = {k: v for k, v in best_trial.params.items()}

    return best_params, best_trial.value, study
