# RiskLens AI

End-to-end **credit default prediction** on LendingClub loan data — from raw CSV to risk buckets, policy decisions, SHAP explanations, and portfolio analytics.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place loan.csv in project root (already included)

# 3. Run full pipeline (skips CSV export if parquet exists)
python scripts/run_all.py

# 4. Launch interactive demo
streamlit run app/streamlit_app.py
```

## Pipeline steps

| Step | Script | Output |
|------|--------|--------|
| 0 | `scripts/setup_db.py` | `data/processed/loan_raw.parquet` |
| 1–2 | `scripts/run_pipeline.py` | `loan_cleaned.parquet`, `features_v1.parquet` |
| 3–6 | `pipelines/train.py` | `data/models/*.joblib`, SHAP CSV |
| 7 | `scripts/score_demo.py` | CLI scoring demo |
| 8 | `scripts/generate_reports.py` | `reports/*.png`, `model_comparison.json` |
| 9 | `scripts/portfolio_analysis.py` | `reports/portfolio_summary.csv` |

See **[PROJECT_COMPLETION_GUIDE.md](PROJECT_COMPLETION_GUIDE.md)** for full step-by-step documentation.

## Project structure

```
RiskLens AI/
├── app/streamlit_app.py      # Interactive demo UI
├── app_config/               # Settings & logging
├── data/
│   ├── processed/            # Raw & cleaned parquet
│   ├── features/             # Feature store
│   └── models/               # Trained models & SHAP
├── pipelines/train.py        # Model training orchestrator
├── reports/                  # Charts & portfolio outputs
├── scripts/                  # Runnable pipeline steps
├── src/
│   ├── data/                 # Cleaning & leakage removal
│   ├── features/             # Feature engineering
│   ├── models/               # Training & evaluation
│   ├── explainability/       # SHAP
│   ├── risk/                 # Bucketing & policy
│   └── scoring/              # End-to-end scorer
├── INTERVIEW_GUIDE.md        # How to present in interviews
└── PROJECT_COMPLETION_GUIDE.md
```

## Model validation

**Out-of-time temporal split** (simulates production):

- **Train:** 2007–2016
- **Validate:** 2017 (threshold tuning)
- **Test:** 2018 (final metrics)

Leakage columns (`grade`, `int_rate`, payment history, etc.) are removed before training.

## Hyperparameter Tuning (Optuna)

We use **Optuna** for tuning the LightGBM champion model's hyperparameters (optimizing validation ROC-AUC). We chose Optuna over Grid or Random Search because:
1. **Bayesian Optimization (TPE):** Instead of checking parameters blindly, Optuna builds a probability model of the search space using previous trials to guide subsequent trials toward the most promising parameter regions.
2. **Compute Efficiency:** LightGBM has a large search space (9+ hyperparameters). Optuna finds high-performing parameters in far fewer trials (e.g., 50 trials) compared to Grid Search, saving hours of training time.
3. **Out-of-Time Validation Integration:** Optuna is set up to evaluate trials directly on the 2017 out-of-time validation set. This avoids the high overhead of multi-fold cross-validation on the 1.1M row training set, while ensuring the model tunes for temporal generalization.

## Key results (2018 test set)

Re-run `pipelines/train.py` after code updates. Typical metrics:

| Model | ROC-AUC | KS |
|-------|---------|-----|
| Logistic Regression | ~0.67 | ~25% |
| LightGBM | **~0.71** | **~30%** |

## Docs

- [PROJECT_COMPLETION_GUIDE.md](PROJECT_COMPLETION_GUIDE.md) — build steps, what's done, how to run each phase
- [INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md) — layered talking points for interviews

## Hackathon deliverables checklist

- [x] GitHub-ready code structure
- [x] End-to-end runnable pipeline
- [x] Risk bucketing + policy engine
- [x] SHAP explainability
- [x] Visualizations (`reports/`)
- [x] Portfolio analysis (bonus)
- [x] Streamlit demo (local)
- [ ] Jupyter notebook (optional — use Streamlit + reports instead)
- [ ] PDF report (export from notebook or write separately)
- [ ] Cloud hosting (optional — Streamlit Community Cloud)

## License

Educational / hackathon project.
