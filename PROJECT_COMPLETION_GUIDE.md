# RiskLens AI — Project Completion Guide

Step-by-step guide to the **full project**. Each step has a script, what it does, expected outputs, and how to verify it worked.

Use this to finish the hackathon, debug issues, or explain the build in interviews.

---

## Progress overview

```text
[✅] Step 0  CSV → Parquet ingest
[✅] Step 1  Data cleaning + leakage removal
[✅] Step 2  Feature engineering
[✅] Step 3  Model training (LR + LightGBM)
[✅] Step 4  Temporal evaluation (2018 OOT test)
[✅] Step 5  SHAP global explanations
[✅] Step 6  Risk bucketing (wired)
[✅] Step 7  Policy engine + scoring service (wired)
[✅] Step 8  Reports & visualizations
[✅] Step 9  Portfolio analysis (bonus)
[✅] Step 10 Streamlit demo (local)
[✅] Step 11 README + requirements
[ ] Step 12 Jupyter notebook (optional)
[ ] Step 13 PDF report (you write / export)
[ ] Step 14 Cloud hosting (optional)
```

**Run everything at once:**
```bash
pip install -r requirements.txt
python scripts/run_all.py
```

---

## Step 0 — Raw data ingest

| | |
|---|---|
| **Script** | `python scripts/setup_db.py` |
| **Input** | `loan.csv` (~2.26M rows) |
| **Output** | `data/processed/loan_raw.parquet` |
| **Skip if** | Parquet already exists |

**What it does:** Converts the 1GB CSV to compressed Parquet so downstream steps are fast and typed.

**Verify:**
```bash
# Should print row count ~2,260,668
python -c "import polars as pl; print(pl.read_parquet('data/processed/loan_raw.parquet').shape)"
```

---

## Step 1 — Cleaning + leakage removal

| | |
|---|---|
| **Script** | `python scripts/run_pipeline.py` (Step 1 section) |
| **Module** | `src/data/cleaning.py`, `src/data/leakage.py` |
| **Input** | `loan_raw.parquet` |
| **Output** | `data/processed/loan_cleaned.parquet` |

**What it does:**
1. Maps `loan_status` → binary `target` (default / non-default)
2. **Excludes** `Current` and unresolved loans (stronger than hackathon naive mapping)
3. Removes **leakage columns** (payments, grade, int_rate, hardship, etc.)
4. Imputes missing values, caps outliers, encodes categories
5. Extracts `issue_year`, `issue_month`, `credit_history_months`

**Verify:** ~1.3M rows, ~20% default rate in cleaned file.

---

## Step 2 — Feature engineering

| | |
|---|---|
| **Script** | `python scripts/run_pipeline.py` (Step 2 section) |
| **Module** | `src/features/engineering.py` |
| **Input** | `loan_cleaned.parquet` |
| **Output** | `data/features/features_v1.parquet` |

**What it does:** Adds 10 interpretable ratio features (`income_to_loan`, `loan_pct_of_income`, `delinq_score`, etc.) on top of ~80 cleaned columns → **~90 features + target + issue_year**.

**Verify:**
```bash
python -c "import polars as pl; df=pl.read_parquet('data/features/features_v1.parquet'); print(df.shape, df.columns[-5:])"
```

---

## Step 3 — Model training

| | |
|---|---|
| **Script** | `python pipelines/train.py` |
| **Module** | `src/models/training.py` |
| **Input** | `features_v1.parquet` |
| **Output** | `data/models/baseline_lr.joblib`, `champion_lgb.joblib`, `model_metadata.joblib` |

**What it does:**
1. **Temporal split:** train ≤2016, val 2017, test 2018
2. Trains **Logistic Regression** (scaled + one-hot) as baseline
3. Trains **LightGBM** (native categoricals + early stopping on AUC)
4. Tunes threshold with **Youden's J** on validation set
5. Evaluates both on **2018 out-of-time test**

**Fix applied (v2):** LightGBM now uses native `category` dtypes and `eval_metric='auc'` with `stopping_rounds=100` — previously stopped at only 5 trees.

**Re-train after fix:**
```bash
python pipelines/train.py
```

**Verify:** Check `data/models/model_metadata.joblib` for test ROC-AUC.

---

## Step 4 — Evaluation metrics

| | |
|---|---|
| **Where** | Printed by `pipelines/train.py`, saved in `model_metadata.joblib` |
| **Primary metric** | ROC-AUC on **2018 test only** |

**Splits:**

| Split | Years | Purpose |
|-------|-------|---------|
| Train | 2007–2016 | Learn patterns |
| Val | 2017 | Threshold + early stopping |
| Test | 2018 | Final honest score |

**Note:** `issue_year` is used for splitting but **not** fed to the model.

---

## Step 5 — SHAP explainability

| | |
|---|---|
| **Script** | Part of `pipelines/train.py` (Step 6) |
| **Module** | `src/explainability/shap_explainer.py` |
| **Output** | `data/models/shap_importance.csv` |

**What it does:** Global feature importance on a 2,000-row test sample.

**Top drivers (typical):** `term`, `acc_open_past_24mths`, `loan_pct_of_income`, `dti`.

**Local SHAP** (single borrower): used in scoring service when model is LightGBM.

---

## Step 6 — Risk bucketing

| | |
|---|---|
| **Module** | `src/risk/bucketing.py` |
| **Wired in** | `src/scoring/predictor.py` |

**What it does:** Maps PD → 5 tiers with grade and suggested rate band.

| PD | Tier | Grade |
|----|------|-------|
| < 5% | Very Low | A |
| 5–12% | Low | B |
| 12–22% | Medium | C |
| 22–35% | High | D |
| ≥ 35% | Very High | E/F |

---

## Step 7 — Policy engine + scoring service

| | |
|---|---|
| **Modules** | `src/risk/decision.py`, `src/scoring/predictor.py` |
| **Demo** | `python scripts/score_demo.py` |

**What it does (full path per loan):**
```text
features → model PD → risk bucket → knockout policy → optional SHAP
```

**Knockout rules:**
- PD ≥ 35%
- DTI > 50%
- Inquiries (6m) > 5
- Delinquencies (2y) > 3
- Loan > 45% of annual income

**Outputs:** Approve / Refer / Deny + adverse action reasons.

**Try it:**
```bash
python scripts/score_demo.py
python scripts/score_demo.py --no-shap   # faster
```

---

## Step 8 — Reports & visualizations

| | |
|---|---|
| **Script** | `python scripts/generate_reports.py` |
| **Output folder** | `reports/` |

**Files created:**

| File | Description |
|------|-------------|
| `roc_curve.png` | ROC on 2018 test |
| `pr_curve.png` | Precision-recall curve |
| `shap_importance.png` | Top 15 SHAP features |
| `risk_bucket_default_rates.png` | Validates buckets vs actual defaults |
| `risk_bucket_default_rates.csv` | Bucket stats table |
| `portfolio_risk_distribution.png` | PD histogram |
| `model_comparison.json` | LR vs LGB metrics |

**Use these in your hackathon PDF / slides.**

---

## Step 9 — Portfolio analysis (bonus)

| | |
|---|---|
| **Script** | `python scripts/portfolio_analysis.py` |
| **Output** | `reports/portfolio_summary.csv` |

**What it does:**
- Groups 2018 test loans by risk tier
- Computes exposure, avg PD, actual default rate
- **Expected Loss** = PD × LGD (60%) × loan amount
- Policy decision counts (approve / refer / deny)

---

## Step 10 — Streamlit demo (local “hosting”)

| | |
|---|---|
| **Script** | `streamlit run app/streamlit_app.py` |
| **URL** | http://localhost:8501 |

**What it does:**
- Load a random 2018 test loan
- Score it with full decision output
- Show SHAP top factors
- Display report charts from `reports/`

**This is your demo for judges / interviews** until you deploy to Streamlit Cloud.

---

## Step 11 — Repo polish

| File | Purpose |
|------|---------|
| `README.md` | Quick start for judges |
| `requirements.txt` | `pip install -r requirements.txt` |
| `INTERVIEW_GUIDE.md` | How to talk about the project |
| `PROJECT_COMPLETION_GUIDE.md` | This file |

---

## Step 12 — Jupyter notebook (optional)

**Not created yet.** Hackathon asks for a notebook — you can either:

1. **Option A:** Create `notebooks/RiskLens_EDA.ipynb` with EDA + model summary (copy charts from `reports/`)
2. **Option B:** Point judges to Streamlit + `reports/` + this guide

Minimal notebook sections:
- Data overview & target distribution
- Leakage columns removed (table)
- Temporal split diagram
- Model metrics table
- Risk bucket validation chart
- Sample scoring output

---

## Step 13 — PDF report (you write)

Export from notebook or write in Word/Google Docs. Suggested sections:

1. **Problem & objective**
2. **Data & target definition** (why Current excluded)
3. **Leakage strategy**
4. **Features** (table of top 10 + engineered ratios)
5. **Model & validation** (temporal split, metrics table)
6. **Risk buckets & policy**
7. **Portfolio insights** (from `portfolio_summary.csv`)
8. **Conclusion & next steps**

Paste charts from `reports/`.

---

## Step 14 — Cloud hosting (optional)

| Platform | What to deploy |
|----------|----------------|
| **Streamlit Community Cloud** | `app/streamlit_app.py` (free, easiest) |
| **Hugging Face Spaces** | Streamlit app + upload model artifacts |
| **Render / Railway** | FastAPI wrapper (not built yet) |

**Note:** Models are ~60KB; feature parquet is large — demo app uses pre-trained models + samples, not full 1.3M row upload.

---

## One-command run order

```bash
# Full pipeline (recommended first time)
python scripts/run_all.py

# Force re-train after code changes
python scripts/run_all.py --force-train

# Demo UI
streamlit run app/streamlit_app.py
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `loan_raw.parquet not found` | Run `python scripts/setup_db.py` |
| `Features file not found` | Run `python scripts/run_pipeline.py` |
| `Model metadata not found` | Run `python pipelines/train.py` |
| LightGBM worse than LR | Re-run training after v2 fix |
| Charts empty | Run `python scripts/generate_reports.py` after training |
| Streamlit import error | `pip install streamlit` |
| Out of memory on CSV | Parquet already exists — use `--skip-setup` |

---

## Interview one-liner per step

| Step | Say this |
|------|----------|
| 0 | "I materialized 2M rows to Parquet for efficient replayable pipelines." |
| 1 | "I removed leakage and unresolved loans before any modeling." |
| 2 | "I built interpretable ratio features, not black-box embeddings." |
| 3 | "LR baseline for interpretability, LightGBM for accuracy." |
| 4 | "Out-of-time 2018 test — train on past, score the future." |
| 5 | "SHAP for global importance and per-borrower explanations." |
| 6–7 | "PD feeds risk tiers and a policy engine with adverse action reasons." |
| 8–9 | "I validated buckets against actual default rates and computed portfolio EL." |
| 10 | "Streamlit demo for end-to-end scoring in front of stakeholders." |

See **[INTERVIEW_GUIDE.md](INTERVIEW_GUIDE.md)** for deeper Q&A.

---

## What changed in this completion pass

| Added / fixed | File |
|---------------|------|
| LightGBM training fix | `src/models/training.py` |
| Missing ingest script | `scripts/setup_db.py` |
| End-to-end scorer | `src/scoring/predictor.py` |
| CLI scoring demo | `scripts/score_demo.py` |
| Report charts | `scripts/generate_reports.py` |
| Portfolio EL analysis | `scripts/portfolio_analysis.py` |
| Streamlit UI | `app/streamlit_app.py` |
| Full orchestrator | `scripts/run_all.py` |
| Dependencies | `requirements.txt` |
| Project README | `README.md` |
| This guide | `PROJECT_COMPLETION_GUIDE.md` |

---

*Update the metrics table in Step 4 after re-running `pipelines/train.py`.*
