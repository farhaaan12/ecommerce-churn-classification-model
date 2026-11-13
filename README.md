# E-Commerce Customer Churn Prediction with Machine Learning

A binary classification model that predicts which e-commerce customers are at risk of **churning** (`Churn = 1`), so the Customer Retention team can target interventions (vouchers, promos, priority service) precisely instead of offering them to everyone.

All work lives in a single notebook: `E-Commerce_Customer_Churn_Prediction.ipynb`.

> Note: the notebook itself is written in Indonesian.

---

## Background & Goals

Acquiring new customers costs far more than retaining existing ones. Without a predictive model, the business has two equally bad options: do nothing (customers leave) or give promos to everyone (budget is wasted).

Project goals:

- **Early detection** of customers at risk of churning (high Recall).
- **Budget efficiency**, keeping Precision reasonable so promos are not wasted on loyal customers.
- **Understanding churn drivers** through feature importance and SHAP.
- **Measuring financial impact** through a Cost-Benefit Analysis.

### Success Criteria (defined before looking at test set results)

| # | Criterion | Target |
| :-: | :--- | :--- |
| 1 | Churn-class Recall (test set) | ≥ 0.70 |
| 2 | Churn-class Precision (test set) | ≥ 0.50 *(assumption; adjust to the retention budget)* |
| 3 | Cost with the model | Lower than both "no campaign" and "campaign to all customers" |

Each criterion is checked on the point estimate and on the lower bound of a 95% bootstrap confidence interval.

---

## Dataset

Required file: **`data_ecommerce_customer_churn.csv`** (not included in this repository; place it in the notebook's folder, in `data/`, or in `/mnt/user-data/uploads/`).

- 3,941 rows x 11 columns (raw); 3,270 rows after cleaning.
- One row = one customer. There is no customer ID or date column.
- Churn rate is about 16% (imbalanced classes).

| Column | Type | Description |
| :--- | :--- | :--- |
| `Churn` | Binary | **Target.** 1 = stopped shopping, 0 = retained |
| `Tenure` | Numeric | How long the customer has been registered (months) |
| `WarehouseToHome` | Numeric | Distance from warehouse to the customer's home |
| `NumberOfDeviceRegistered` | Discrete numeric | Number of devices registered on the account |
| `PreferedOrderCat` | Categorical | Product category ordered most often |
| `SatisfactionScore` | Ordinal (1-5) | Customer satisfaction score |
| `MaritalStatus` | Categorical | Single / Married / Divorced |
| `NumberOfAddress` | Discrete numeric | Number of saved shipping addresses |
| `Complain` | Binary | Whether the customer has filed a complaint |
| `DaySinceLastOrder` | Numeric | Days since the last order |
| `CashbackAmount` | Numeric | Average cashback received |

---

## Notebook Workflow

1. **Business Understanding**: background, problem statement, metrics, success criteria, and notes on temporal leakage risk.
2. **Feature Availability Registry**: every feature must be registered with its definition, recording time, and leakage risk level. `check_feature_availability()` stops the notebook if an unregistered column appears.
3. **Data Cleaning**
   - Missing values in `Tenure`, `WarehouseToHome`, and `DaySinceLastOrder` are not dropped; they are handled inside the pipeline (median imputation plus a missing-indicator column).
   - 671 duplicate rows (about 17%) are removed to prevent leakage between train and test.
   - The label `Mobile` is merged into `Mobile Phone`.
   - Outliers are kept and handled with `RobustScaler`.
   - A data quality report is saved as `data_quality_report.csv`; the cleaned data as `ecommerce_churn_clean.csv`.
4. **Train-Test Split**: 80:20, stratified, `random_state=42`.
5. **EDA** (train data only): univariate and bivariate analysis against churn.
6. **Feature Engineering**: `ColumnTransformer` (One-Hot, `RobustScaler`, `StandardScaler`, passthrough).
7. **Resampling**: SMOTE inside the pipeline, so it is applied only to the training portion of each fold.
8. **Model Benchmarking**: 6 algorithms with Stratified 5-Fold CV: Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, LightGBM.
9. **Hyperparameter Tuning**: `GridSearchCV` optimizing churn-class F1.
10. **Overfitting Reduction & Final Model Selection**: regularized XGBoost and early stopping; the model is chosen with a one-standard-deviation rule (the most stable among equivalent candidates).
11. **Imbalance Strategy Comparison**: SMOTE, SMOTENC, RandomOverSampler, class weights.
12. **Model Interpretation**: feature importance and SHAP.
13. **Cost-Benefit Analysis**: the threshold is chosen from total cost on out-of-fold train predictions, then compared against two baseline scenarios on the test set.
14. **Advanced Evaluation**: bootstrap CI, success criteria check, full metric report, probability calibration (Platt scaling).
15. **Sensitivity Tests**: with vs without duplicates, and ablation of risky features (`Complain`, `CashbackAmount`, `DaySinceLastOrder`).
16. **Risk Segmentation**: Low / Medium / High / Very High, with cost-benefit simulation and break-even success rate per segment.
17. **Deployment**: the model is saved as a single `joblib` bundle with a version registry and rollback function.
18. **Monitoring**: data drift (PSI), performance checks, retraining schedule, rollback procedure.
19. **A/B Test Plan**, **Conclusion & Recommendations**, and **Limitations**.

---

## Cost Assumptions (Illustrative)

The values below are assumptions and **must be replaced with real figures from Finance/Marketing**:

| Parameter | Value |
| :--- | :--- |
| `LOST_VALUE` (loss per churner not saved) | IDR 500,000 |
| `CAMPAIGN_COST` (campaign cost per targeted customer) | IDR 50,000 |
| `SUCCESS_RATE` (chance a churner is saved when targeted) | 30% |

---

## Key Findings (EDA)

- **Short tenure** is the strongest signal: churn rate is 55.7% for customers with 0 months of tenure and 0% beyond 24 months (train data).
- Customers who **have filed a complaint** churn far more often (31.4% vs 10.6%).
- Higher `SatisfactionScore` is associated with higher churn (counter-intuitive; check survey quality and timing).
- Final model results (Recall, Precision, cost savings, success criteria status) are computed automatically when the full notebook is run, in the *Conclusion & Recommendation* and *Executive Summary* sections.

---

## Installation

Python 3.10+ and a virtual environment are recommended.

```bash
pip install pandas numpy scipy matplotlib seaborn \
            scikit-learn imbalanced-learn xgboost lightgbm shap \
            joblib jupyter
```

## How to Run

1. Place `data_ecommerce_customer_churn.csv` in the notebook folder (or in `data/`).
2. Open the notebook:
   ```bash
   jupyter notebook E-Commerce_Customer_Churn_Prediction_with_Machine_Learning_revised_v2.ipynb
   ```
3. Run all cells in order (*Run All*). The `GridSearchCV` tuning stage can take several minutes.

### Generated Outputs

| File / Folder | Contents |
| :--- | :--- |
| `ecommerce_churn_clean.csv` | Data after cleaning |
| `data_quality_report.csv` | Data quality report |
| `model_artifacts/` | Model bundle (`joblib`) and `registry.json` (version history) |

---

## Using the Model on New Data

The bundle contains the full pipeline (preprocessing, SMOTE, model), the final threshold, a calibrator, expected columns, training ranges, and library-version metadata.

```python
bundle = load_model()                             # 'current' version from model_artifacts/registry.json
result = predict_churn(new_data, bundle=bundle)   # churn_score, calibrated probability, churn_flag, risk_tier

# if the new version has problems:
rollback_model()                                  # revert to the previous version without retraining
```

Notes:

- New data must contain **exactly the same columns** as the training data (names and types).
- Use the same `scikit-learn`, `xgboost`, and `imbalanced-learn` versions recorded in `metadata['library_versions']`.
- `load_model`, `rollback_model`, and `predict_churn` are defined inside the notebook; copy them into a service script if used outside the notebook.
- Treat the raw score as a **risk ranking**; use `churn_flag` for decisions.

---

## Monitoring

| PSI | Status | Action |
| :--- | :--- | :--- |
| < 0.10 | Stable | None |
| 0.10 - 0.25 | Warning | Investigate the cause, increase monitoring frequency |
| > 0.25 | Drift | Re-evaluate performance, consider retraining |

Performance (Recall and Precision) is checked monthly once churn labels are available; retrain if it stays below the minimum for 2 consecutive periods.

---

## Limitations

- **The churn definition and feature recording times are unconfirmed.** `Complain`, `CashbackAmount`, and `DaySinceLastOrder` carry temporal leakage risk; every feature in the registry is still `verified=False`. Confirm with the data team before deployment.
- The data is a snapshot with no dates or customer IDs, so time-based validation cannot be used yet and duplicates cannot be confirmed as true duplicates.
- The dataset is small (3,270 rows; test set of 654 rows with 107 churners), so performance estimates carry considerable uncertainty.
- Tree-based models tend to overfit; look at the train-CV F1 gap, not the train score.
- Rupiah figures depend on the cost assumptions; validate with an A/B test.
- Correlation is not causation. Consider ethical implications before using `MaritalStatus` to treat customers differently.
- Predictions outside the training value ranges (e.g. `Tenure` > 61, `CashbackAmount` > ~325) are less reliable.

---

## Business Recommendations (Summary)

1. Onboarding program for new customers (0-3 months).
2. Fast complaint handling (SLA and proactive follow-up).
3. Target campaigns by model score, not to all customers.
4. Pay attention to the *Mobile Phone* category and *Single* customers.
5. Investigate the `SatisfactionScore` finding.
6. Run an A/B test before full rollout.
7. Confirm the churn definition and feature timing with the data team.
8. Monitor drift and performance regularly.

---

## Tech Stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, LightGBM, SHAP, seaborn, matplotlib, joblib.


## 6. Contact
- Name: Emily Vitasya Haloho, Farhan
- Email: farhanfarhan54202@gmail.cpm
- Linkedin: https://www.linkedin.com/in/farhan-farhan-350a48253/
