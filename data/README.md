# Data directory

Large files are **not** committed to Git (see root `.gitignore`).

## Setup

1. Place `loan.csv` in the project root (LendingClub dataset).
2. Run the pipeline:

```bash
python scripts/setup_db.py          # CSV → data/processed/loan_raw.parquet
python scripts/run_pipeline.py      # clean + features
python pipelines/train.py           # train models → data/models/
```

Generated folders:

| Path | Contents |
|------|----------|
| `data/processed/` | Raw & cleaned parquet |
| `data/features/` | Feature store |
| `data/models/` | Trained models (regenerate locally) |

`data/models/shap_importance.csv` is committed as a small reference output.
