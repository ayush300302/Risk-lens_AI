# RiskLens AI — Project Report

**Author:** Ayush Patil  
**Dataset:** LendingClub historical loans (~1.3M resolved outcomes)  
**Task:** Predict probability of default (PD) and assign risk tiers

---

## 1. Objective

Build an end-to-end credit risk system that:

1. Predicts **probability of default** for retail loan borrowers
2. Maps PD to **interpretable risk buckets** (Very Low → Very High)
3. Applies a **credit policy engine** (approve / refer / deny)
4. Validates on **out-of-time** 2018 data (simulates production)

---

## 2. Methodology

### 2.1 Data preprocessing

- Target: `Charged Off` / `Default` = 1, `Fully Paid` = 0
- Excluded `Current` and unresolved statuses (avoids survivorship bias)
- Removed **leakage columns**: `grade`, `int_rate`, `installment`, post-origination payment fields
- Imputed missing values, capped income/DTI outliers
- Engineered 10 ratio features (e.g. `loan_pct_of_income`, `delinq_score`)

### 2.2 Validation strategy

| Split | Period | Rows | Default rate |
|-------|--------|------|--------------|
| Train | 2007–2016 | 1,097,537 | 19.9% |
| Val | 2017 | 158,910 | 22.9% |
| Test | **2018** | 47,191 | 14.7% |

`issue_year` is used for splitting only — **not** as a model feature.

### 2.3 Models

- **Baseline:** Logistic Regression (scaled numerics + one-hot categoricals)
- **Champion:** LightGBM (native categorical features, tuned via **Optuna TPE Bayesian search** on the 2017 validation set). Optuna was chosen over Grid/Random search because it builds a probability model of the search space, focusing trials on promising regions to optimize the 9+ hyperparameters efficiently and maximize ROC-AUC.
- Threshold tuned with **Youden's J** on 2017 validation set

---

## 3. Results (2018 Out-of-Time Test)

| Metric | Logistic Regression | LightGBM (Champion) |
|--------|--------------------:|--------------------:|
| **ROC-AUC** | 0.667 | **0.707** |
| PR-AUC | 0.243 | 0.290 |
| Gini | 0.334 | 0.414 |
| **KS (%)** | 24.8 | **30.2** |
| Default Recall | 62.5% | 65.1% |
| Default Precision | 22.0% | 24.2% |

**Champion:** LightGBM (KS ~30% = acceptable/good for retail credit)

### Top SHAP drivers

1. `term` (loan length)  
2. `acc_open_past_24mths`  
3. `dti`  
4. `loan_pct_of_income`  
5. `addr_state`

---

## 4. Risk Bucketing

| PD range | Tier | Action |
|----------|------|--------|
| < 5% | Very Low | Approve |
| 5–12% | Low | Approve |
| 12–22% | Medium | Approve (review) |
| 22–35% | High | Conditional |
| ≥ 35% | Very High | Deny |

**Bucket validation (2018):** Actual default rate increases monotonically from Very Low (1.6%) → Very High (18.8%), confirming ranking power.

---

## 5. Portfolio Insights (2018 test, simplified EL)

- **Total exposure:** $705M  
- **Expected loss (PD × 60% LGD):** $216M  
- **70%** of loans classified Very High PD — reflects 2018 distribution shift (lower actual default rate than train)

Policy breakdown: majority Very High tier referred to deny under knockout rules; Low/Very Low tiers mostly approved.

---

## 6. Business Recommendations

1. Deploy **LightGBM** as champion scorer; keep LR for regulatory explainability baseline
2. **Recalibrate** PD on recent vintages before production (2018 shift visible)
3. Use **SHAP** for adverse action explanations on denials
4. Combine model PD with **hard policy rules** (DTI, inquiries, delinquencies)

---

## 7. Reproducibility

```bash
pip install -r requirements.txt
python scripts/run_all.py --force-train
streamlit run app/streamlit_app.py
```

**Repository:** https://github.com/ayush300302/Risk-lens_AI

---

*Export this file to PDF: open in VS Code / browser → Print → Save as PDF*
