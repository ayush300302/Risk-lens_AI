# RiskLens AI — Interview Guide

> **Build & run guide:** see [PROJECT_COMPLETION_GUIDE.md](PROJECT_COMPLETION_GUIDE.md) for pipeline steps, scripts, and what's done vs optional.

Use this as a **layered script**. Start short. Only go deeper when the interviewer asks follow-ups.

---

## How to use this guide

| Signal from interviewer | How deep to go |
|------------------------|----------------|
| "Tell me about a project" / "Walk me through your resume" | **Level 1** (30 sec) → stop, let them ask |
| "How did you build it?" / "What was your approach?" | **Level 2** (2 min) |
| "Explain your validation" / "How did you handle leakage?" | **Level 3** (that topic only) |
| "Why is AUC only 0.67?" / "What would you improve?" | **Level 4** (honest tradeoffs + next steps) |

**Rule:** Answer the question asked, then pause. Don't dump the whole pipeline unless they want it.

---

## Level 1 — 30-second pitch (memorize this)

> "I built **RiskLens AI**, an end-to-end **credit default prediction** system on LendingClub loan data — about **2.2M applications**, roughly **1.3M** after cleaning for resolved outcomes.
>
> The pipeline goes from raw data → cleaning and **leakage removal** → feature engineering → **Logistic Regression baseline** plus **LightGBM** champion → **SHAP explanations** → **risk buckets** and a simple **credit policy** layer.
>
> I validated with an **out-of-time split** — train through 2016, validate on 2017, test on 2018 — so metrics reflect how the model would perform on **future** loans, not a random mix of years.
>
> On the 2018 holdout, logistic regression reached about **0.67 ROC-AUC** with **~25% KS**, which is reasonable for a model that deliberately excludes lender-assigned fields like grade and interest rate."

**Stop here.** Wait for a follow-up.

---

## Level 2 — 2-minute walkthrough (if they say "go on" or "how does it work?")

### Problem
- Predict **probability of default (PD)** for a retail loan borrower.
- Turn PD into **actionable risk tiers** (approve / review / deny) with business meaning.

### Data
- LendingClub historical loans, **145+ raw columns**.
- Target: `loan_status` → binary default (1 = Charged Off / Default, 0 = Fully Paid).
- Excluded **Current** and ambiguous statuses — outcome is unknown for active loans.

### Pipeline (4 stages)
1. **Ingest** — CSV → Parquet (DuckDB / Polars for scale).
2. **Clean + de-leak** — impute missing values, cap outliers, encode categories, remove post-origination and lender-assigned columns.
3. **Features** — ~90 features including ratios like income-to-loan, DTI burden, delinquency score.
4. **Model + decision** — LR baseline + LightGBM; SHAP for explainability; risk buckets + policy rules on top.

### Validation
- **Train:** 2007–2016 (~1.1M rows)
- **Val:** 2017 (~159k) — threshold tuning, model selection
- **Test:** 2018 (~47k) — final out-of-time evaluation

### Results (2018 test, headline numbers)
| Model | ROC-AUC | KS | Notes |
|-------|---------|-----|-------|
| Logistic Regression | **0.667** | **24.8%** | Interpretable baseline |
| LightGBM | 0.655 | 22.1% | Champion candidate; tuning in progress |

### Business layer
- PD → 5 risk tiers (Very Low → Very High) with suggested rate bands.
- Policy engine adds knockout rules (DTI, inquiries, delinquencies) for approve/deny/refer.

**Stop here** unless they drill into a specific area.

---

## Level 3 — Topic cards (go deep only when asked)

### A) Temporal split — "Why train ≤2016, val 2017, test 2018?"

**Short answer (15 sec):**
> "Credit models are deployed on **future** loans. I split by `issue_year` so the test set simulates scoring 2018 applications using only past data."

**If they push deeper (30–45 sec):**
> "`issue_year` is used for splitting only — it's **not** a model feature, so the model can't cheat by learning '2018 was safer.'
>
> Default rate shifts over time — train ~20%, 2017 ~23%, 2018 ~15% — so out-of-time AUC is lower than a random split, but it's **more honest** and closer to production.
>
> I wouldn't use random split as the primary metric in credit risk; I'd use it only as a comparison baseline."

**One-liner for interviews:**
> "Out-of-time validation = train on the past, test on the future."

---

### B) Data leakage — "How did you handle leakage?"

**Short answer:**
> "I removed anything not known at application time — payment history after origination, recoveries, hardship fields, and **lender-assigned pricing** like grade and interest rate."

**If they push deeper:**
> "Four leakage types I checked:
> 1. **Post-origination** — `total_pymnt`, `out_prncp`, `last_pymnt_d`
> 2. **Post-default** — `recoveries`
> 3. **Lender-assigned** — `grade`, `int_rate`, `installment` (circular — LC already priced risk)
> 4. **Hardship / settlement** — only exist after stress
>
> Keeping `grade`/`int_rate` would inflate AUC to ~0.75+, but that's not a deployable underwriting model — it's re-learning LC's existing score."

---

### C) Target definition — "How did you define default?"

**Short answer:**
> "Default = Charged Off or Default. Non-default = Fully Paid. I excluded Current and late/grace statuses because the outcome isn't resolved yet."

**If they push deeper:**
> "Including Current loans creates **survivorship bias** — most haven't had time to default. The hackathon brief maps Current to non-default, but excluding it is the stronger modeling choice and I can defend that in a risk team setting."

---

### D) Features — "What features mattered?"

**Short answer:**
> "Behavioral and capacity features — loan term, recent account openings, utilization, income-to-loan ratio, DTI. I also engineered simple ratios so each feature is explainable in one sentence."

**Top drivers (from SHAP):**
1. `term` (loan length)
2. `acc_open_past_24mths`
3. `bc_open_to_buy`
4. `loan_pct_of_income`
5. `dti`

**If they ask about categoricals:**
> "`purpose` and `addr_state` are in the model but need better encoding — target encoding with time-safe CV is a planned improvement."

---

### E) Models — "Why LR and LightGBM?"

**Short answer:**
> "Logistic regression is the **interpretable baseline** banks still use. LightGBM is the **performance champion** for tabular credit data. I compare both on the same out-of-time test."

**If they ask why LGB < LR:**
> "LightGBM underperformed because early stopping kicked in too aggressively — only ~5 trees. That's a training-config issue, not a reason to drop the approach. Fix: native categorical handling, relaxed early stopping, and hyperparameter tuning on temporal CV."

---

### F) Threshold — "Why not 0.5?"

**Short answer:**
> "Default rate is ~15–20%, so 0.5 is too high — the model rarely flags defaults. I used **Youden's J** on the 2017 validation set to balance sensitivity and specificity."

---

### G) Risk buckets + policy — "How does ML connect to business?"

**Short answer:**
> "ML outputs PD. I map PD to five tiers with suggested rate bands. A separate **policy engine** applies knockout rules — high DTI, too many inquiries, excessive loan-to-income — and returns adverse action reasons for denials."

**Example tiers:**
| PD range | Tier | Action |
|----------|------|--------|
| < 5% | Very Low | Approve |
| 5–12% | Low | Approve |
| 12–22% | Medium | Approve (review) |
| 22–35% | High | Conditional |
| ≥ 35% | Very High | Deny |

---

### H) Explainability — "How do you explain a prediction?"

**Short answer:**
> "SHAP on the LightGBM model — global feature importance for the portfolio, local contributions for individual borrowers. Each factor shows whether it increases or decreases default risk."

---

### I) Tech stack — "What tools did you use?"

**One sentence:**
> "Polars for scalable cleaning, scikit-learn pipelines for preprocessing, LightGBM for modeling, SHAP for explainability, Pydantic for config, and a modular pipeline layout so cleaning, features, and training are independently rerunnable."

---

## Level 4 — Hard questions (honest answers)

### "Your AUC is only 0.67 — is that good enough?"

> "For an **out-of-time** test without grade or interest rate, **0.65–0.72 is realistic**. KS around 25% is in the acceptable range for retail credit. The goal isn't leaderboard AUC — it's a **deployable, explainable** score that ranks risk correctly and supports policy decisions.
>
> I'd improve it by fixing LightGBM training, better categorical encoding, and probability calibration — not by re-adding leakage features."

---

### "Random split would give higher AUC — why not use that?"

> "Random split mixes years and **overstates** performance. I'd report random split only as a baseline comparison. Primary evaluation should always be **out-of-time** in credit risk."

---

### "What would you do next?"

Pick 2–3 (don't list all ten):

1. Fix LightGBM early stopping and use native categorical features.
2. Target encoding for `purpose` / `addr_state` with temporal cross-validation.
3. Calibrate probabilities (Platt / isotonic) for 2018 distribution shift.
4. Validate risk buckets — default rate should increase monotonically across tiers.
5. Portfolio view — expected loss and concentration by bucket (hackathon bonus).

---

## Numbers cheat sheet (glance before interview)

| Item | Value |
|------|-------|
| Raw loans | ~2.26M |
| After cleaning | ~1.3M (resolved outcomes only) |
| Features | ~90 |
| Train / Val / Test | 1.10M / 159k / 47k |
| Default rate (train / val / test) | ~20% / ~23% / ~15% |
| LR test ROC-AUC | **0.667** |
| LGB test ROC-AUC | 0.655 |
| LR test KS | **24.8%** |
| LR optimal threshold | ~0.49 |
| Leakage columns removed | grade, int_rate, installment, payment fields, etc. |

---

## What NOT to say

| Avoid | Say instead |
|-------|-------------|
| "I used all 145 columns" | "I used application-time features only after leakage review" |
| "LightGBM is the champion" (without caveat) | "LightGBM is the target champion; LR currently wins on OOT test while LGB tuning is in progress" |
| "Current = non-default" (as your only definition) | "I excluded Current because outcome is unknown — stronger than naive mapping" |
| "0.67 is bad" | "0.67 OOT without pricing fields is reasonable; KS ~25% supports ranking power" |
| Long monologue on Polars internals | "Polars for memory-efficient cleaning at 2M+ row scale" |

---

## Closing line (when they ask "anything else?")

> "What I'm most proud of is treating this like a **production credit risk** problem — leakage control, temporal validation, interpretability, and a path from PD to policy — not just maximizing AUC on a random split."

---

## Quick map: interviewer question → section

| They ask… | Jump to |
|-----------|---------|
| "Walk me through the project" | Level 1 → Level 2 if nodding |
| "How did you validate?" | **A) Temporal split** |
| "Leakage?" | **B) Data leakage** |
| "Target variable?" | **C) Target definition** |
| "Important features?" | **D) Features** |
| "Why two models?" | **E) Models** |
| "Threshold?" | **F) Threshold** |
| "Business impact?" | **G) Risk buckets** |
| "Explain a denial" | **H) Explainability** + **G) Policy** |
| "Low AUC?" | **Level 4** |
| "Next steps?" | **Level 4 — What would you do next** |

---

*Last updated from saved model artifacts and pipeline code. Re-run `pipelines/train.py` and update the numbers cheat sheet if you retrain.*
