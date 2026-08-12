# Credit Risk Prediction — Give Me Some Credit

Predicting the probability that a borrower will experience serious delinquency (90+ days past due) within two years, using the "Give Me Some Credit" dataset (150,000 borrowers, 12 raw features).

This project covers the full pipeline: environment setup, EDA, data cleaning, feature engineering, model training/selection, interpretability, probability calibration, and cross-validated evaluation.

---

## 1. Dataset & Problem

- **Source data:** `data/cs-training.csv`, 150,000 rows × 12 columns
- **Target:** binary — did the borrower experience serious delinquency (`SeriousDlqin2yrs`)
- **Class imbalance:** ~6.7% positive (default) rate — a realistic, imbalanced credit risk problem where accuracy alone is a misleading metric

---

## 2. EDA — Key Findings

- **Missing data:** `MonthlyIncome` (~20% missing), `NumberOfDependents` (~2.6% missing)
- **Duplicates:** 767 duplicate rows
- **Data error:** 1 row with `age = 0`
- **Placeholder codes:** the three "days past due" columns contained 96/98 placeholder values. Rows carrying these codes had a **54.6% default rate vs. a 6.7% baseline** — a massive, non-random signal that needed careful handling rather than blind removal
- **Extreme outliers:** `DebtRatio` and `RevolvingUtilizationOfUnsecuredLines` both had implausible extreme values requiring capping

---

## 3. Data Cleaning

- Imputed missing values, with an explicit `MonthlyIncome_missing` flag preserved as a feature (missingness itself can be informative)
- Dropped the index column, 767 duplicate rows, and the 1 bad `age=0` row
- Replaced 96/98 placeholder codes with realistic per-column maxes, while preserving a `WasFlagged_9698` indicator so the information wasn't silently destroyed
- Capped `DebtRatio` and `RevolvingUtilizationOfUnsecuredLines` at their 99th percentiles

**Result:** 149,232 rows × 13 columns, zero missing values → `data/cleaned_data.csv`

---

## 4. Feature Engineering

- **Delinquency aggregates:** `TotalPastDue`, `EverDelinquent`, `WorstDelinquency`
- **Income/debt ratios:** `IncomePerDependent`, `DebtToIncomeProxy`, `IsZeroIncome`
- **Log transforms:** `DebtRatio_log`, `DebtToIncomeProxy_log`, `MonthlyIncome_log`, `RevolvingUtilizationOfUnsecuredLines_log`
- **Age bins:** one-hot encoded (`<25` as reference category)
- Fixed a `DebtRatio` capping bug from the cleaning phase by recapping using only clean, non-imputed-income rows

**Result:** 149,232 rows × 29 columns → `data/featured_data.csv`

---

## 5. Train/Test Split & Preprocessing

- **80/20 stratified split** — both sets hold the 6.7% default rate
- Two parallel feature sets built:
  - **Raw/unscaled** (27 features) for tree-based models — `X_train`, `X_test`
  - **Scaled + decorrelated** (19 features) for Logistic Regression — `X_train_lr`, `X_test_lr`, after removing 8 features with `|r| > 0.8` correlation with other features (kept the more information-dense version in each redundant pair, e.g. log-transformed over raw, aggregated delinquency counts over individual raw columns)
- Scaler fit **only on training data** to avoid leakage

---

## 6. Model Training & Selection

Three models trained and compared on held-out test data, evaluated with **ROC-AUC and PR-AUC** (not accuracy, given the 6.7% base rate):

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.8584 | 0.3745 |
| Random Forest | 0.8637 | 0.3996 |
| XGBoost (default hyperparameters) | **0.8657** | **0.4098** |
| XGBoost (tuned via RandomizedSearchCV) | 0.8656 | 0.4072 |

**Final model: XGBoost with hand-set, untuned hyperparameters** (`n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=13.92`).

A 40-combination RandomizedSearchCV (5-fold, optimizing PR-AUC) did **not** beat the untuned defaults — a useful negative result indicating the initial hyperparameters were already near-optimal for this dataset.

### A data-quality catch worth highlighting: `WasFlagged_9698`

During Logistic Regression training, this feature's coefficient blew up to **-8.27** — roughly 6x larger than any other coefficient, and the wrong sign given its known 54.6% default rate. Diagnosis:

- Only 170 of 119,385 training rows (0.14%) had this flag set
- Among those 170 rows, `WorstDelinquency` had **zero variance** — every flagged row collapsed to an identical extreme scaled value, because the delinquency aggregates were built *after* 96/98 codes were replaced with per-column maxes
- This created **quasi-complete separation**, destabilizing the Logistic Regression coefficient estimate

**Fix:** dropped `WasFlagged_9698` from the Logistic Regression feature set only. Tree-based models were unaffected (the feature's gain-based importance was ~0.00002, i.e. irrelevant to XGBoost/Random Forest) — a good illustration of how the same redundant feature can break one model class while being harmless to another.

---

## 7. Cross-Validation — Confirming the Model Choice Is Robust

To confirm XGBoost's edge over Random Forest wasn't an artifact of one particular train/test split, both models were evaluated via 5-fold stratified cross-validation on the full dataset:

| Model | ROC-AUC (mean ± std) | PR-AUC (mean ± std) |
|---|---|---|
| XGBoost | 0.8647 ± 0.0034 | 0.4010 ± 0.0088 |
| Random Forest | 0.8633 ± 0.0035 | 0.3983 ± 0.0059 |

XGBoost won 4 of 5 folds. **Honest interpretation:** the gap between the two models (~0.0014 mean ROC-AUC) is smaller than either model's fold-to-fold standard deviation — this is a real, consistent, but modest edge, not a dramatic one. XGBoost was selected as the final model on the strength of this consistent advantage plus its native handling of class imbalance via `scale_pos_weight`; Random Forest would also have been a defensible choice.

---

## 8. Interpretability — SHAP

Used `shap.TreeExplainer` on the final XGBoost model to move beyond gain-based feature importance to actual per-prediction attribution.

**Global feature importance:** `TotalPastDue`, `RevolvingUtilizationOfUnsecuredLines`, and `age` were the top SHAP-ranked features — notably different from the gain-based XGBoost ranking (which put `TotalPastDue`, `EverDelinquent`, `WorstDelinquency` on top). This divergence is expected and worth understanding: gain-based importance measures how useful a feature was *to the trees during training*; SHAP measures how much a feature actually moved *real predictions* on the test set. SHAP is the more relevant lens for explaining decisions to a business or regulatory audience.

**Data-quality note found via SHAP:** `NumberRealEstateLoansOrLines` showed a noisy high-SHAP tail for large values. Checking the underlying default rates confirmed this reflects **small-sample noise** (e.g. 4 borrowers with 12 real estate loans, 75% default rate) rather than a genuine, generalizable risk relationship — a caution worth keeping in mind for any individual explanation that leans heavily on this feature.

**Individual explanation example** (`shap_waterfall_example.png`): for a test-set borrower correctly predicted at 98.17% default probability, the largest contributors were `TotalPastDue=7` (+1.6 in log-odds), `RevolvingUtilizationOfUnsecuredLines=1.094` — i.e. maxed out and over-limit (+0.88), prior delinquency history, and elevated open credit lines. Every contributing factor is intuitive and defensible — exactly the kind of "why was this applicant flagged" explanation a credit risk model needs to produce.

*Embed `shap_summary.png`, `shap_importance_bar.png`, and `shap_waterfall_example.png` here.*

---

## 9. Probability Calibration

**Finding:** the raw XGBoost model's predicted probabilities were **severely overconfident**, despite strong ROC-AUC/PR-AUC. This is a known side effect of `scale_pos_weight`-based imbalance handling — it improves *ranking* but distorts the actual probability *values*.

| Predicted probability (raw model) | Actual default rate |
|---|---|
| 0.034 | 0.006 |
| 0.339 | 0.042 |
| 0.629 | 0.126 |
| 0.861 | 0.367 |

An applicant scored at "86% default risk" by the raw model was, in reality, closer to 37% risk.

**Fix:** applied isotonic regression calibration (`CalibratedClassifierCV`, 5-fold). Post-calibration, predicted and actual rates matched closely (e.g. 0.365 predicted → 0.368 actual), while ranking performance was essentially unchanged (ROC-AUC 0.8657 → 0.8653, PR-AUC 0.4098 → 0.4055).

*Embed `calibration_comparison.png` here.*

**Practical implication:** two model artifacts are saved — the raw model for ranking/classification tasks, and the calibrated model for any use case that needs trustworthy probability values (e.g. expected-loss calculations, risk-based pricing).

---

## 10. Decision Threshold

Rather than defaulting to 0.5, the threshold was chosen deliberately, with the trade-off made explicit:

- **F1-optimal threshold (statistically balanced):** 0.786 on raw probabilities → Precision 0.409, Recall 0.492
- **Business-oriented threshold (recall-prioritized):** chosen instead, reflecting that in credit risk, missing an actual defaulter is typically more costly than declining a borderline-good applicant

| Model version | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| Raw XGBoost | 0.30 | 0.154 | 0.885 | 0.262 |
| Calibrated XGBoost | 0.05 | 0.184 | 0.830 | 0.301 |

Both threshold philosophies are documented in `models/model_metadata.json` so the reasoning is auditable, not just the final number.

---

## 11. Final Artifacts

```
models/
├── xgb_final_model.pkl          # Raw XGBoost — use for ranking/classification
├── xgb_calibrated_model.pkl     # Isotonic-calibrated — use for real probability values
└── model_metadata.json          # Hyperparameters, thresholds, rationale, test metrics
```

---

## 12. Limitations & Honest Caveats

- **Single dataset era/source:** the "Give Me Some Credit" dataset reflects historical US consumer credit patterns and would need revalidation before any real-world deployment.
- **RandomizedSearchCV coverage:** only 40 of a very large combinatorial hyperparameter space were tried; a more exhaustive or Bayesian search might find further (likely small) gains.
- **XGBoost vs. Random Forest gap is modest:** cross-validation confirmed it's real but small — both are reasonable choices.
- **`NumberRealEstateLoansOrLines` SHAP signal is partly noise-driven** at high loan counts (small subgroup sizes) and shouldn't be over-interpreted for individual high-count cases.
- **Threshold choice is a business judgment**, not a purely statistical one — the values used here should be revisited with real cost data (cost of a missed default vs. cost of a declined good applicant) before production use.

---

## 13. Tech Stack

Python · pandas · scikit-learn · XGBoost · SHAP · matplotlib
