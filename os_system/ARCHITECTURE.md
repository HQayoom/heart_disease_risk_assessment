# Explainable Osteoarthritis Risk Prediction and Healthcare Analytics System

## Pipeline

```
User (clinician / administrator / researcher)
  → Streamlit web dashboard (12 pages)
  → Patient management · OA prediction · Data analytics
  → Preprocessing (IterativeImputer + BMI features)
  → ML engine (Logistic Regression · Random Forest · SVM · XGBoost)
  → Best calibrated model (test ROC-AUC selection)
  → SHAP explainer + fairness engine
  → Clinical insights + PDF reports + SQLite audit trail
```

## Backend modules

| Module | File | Role |
|--------|------|------|
| Authentication | `auth.py` | bcrypt login, roles, audit on login |
| Patients | `services/patients.py` | CRUD, prediction history |
| Preprocessing | `services/features.py` | Imputation, BMI engineering |
| Training | `services/train.py` | Four-model comparison, calibration, artifacts |
| Prediction + SHAP | `services/explain.py` | Local/global explanations, risk bands |
| Fairness | `services/fairness.py` | Sex, age, race subgroup metrics |
| Reports | `services/reports.py` | PDF export |
| UI | `ui.py`, `assets/style.css` | Shared styling |

## Persistence

- **SQLite** (`oa_clinic.db`): users, patients, predictions, audit_log  
- **Artifacts** (`artifacts/`): `bundle.joblib`, `comparison.json`, `curves.json`, `thresholds.json`, `fairness.json`, `global_shap.json`

## Production form constraint

Only the NHANES-trained columns in `config.FEATURE_COLUMNS` are collected (demographics, comorbidities, activity, mobility, and self-rated health). Derived BMI/age interaction terms are computed internally.

Email and cloud storage are specified for a production deployment and are not wired in this prototype.
