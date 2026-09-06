# Explainable Osteoarthritis Risk Prediction and Healthcare Analytics System

A Streamlit clinical decision-support prototype that takes twelve routine patient fields and returns a calibrated osteoarthritis (OA) risk probability, a Low / Moderate / High band, a per-patient SHAP explanation, a subgroup fairness audit, and a printable PDF report.

Trained on **public NHANES 2011–2018** data from the CDC. No data licence or NDA is required.

> **Research prototype.** This is not a medical device and must not replace professional clinical diagnosis, imaging, or treatment decisions.

---

## Contents of this project folder

| Folder | What it is | Status |
|--------|------------|--------|
| `oa_system/` | **The main application** — OA risk prediction system (12 pages, SQLite, SHAP, fairness, PDF) | Active |
| `Project_Documentation/` | Dissertation Chapters 5 and 6 plus all 11 generated figures | Active |
| `App/` | Legacy heart-disease risk app from an earlier project | Archived |
| `Notebooks/` | Legacy heart-disease notebooks (wrangling, EDA, modelling) | Archived |
| `data_directory/`, `images/` | Reference material for the legacy heart-disease work | Archived |

The two applications have **incompatible dependencies** and must not share one virtual environment. See [Running the legacy heart-disease app](#running-the-legacy-heart-disease-app).

---

## Requirements

| Item | Version |
|------|---------|
| Python | 3.12 (developed on 3.12.3) |
| Disk space | ~1 GB including dependencies |
| OS | Windows 10/11, macOS, or Linux |

---

## Quick start — Windows

Open **PowerShell** or **Command Prompt** in the project folder. If you are not already there:

```powershell
cd "C:\path\to\AI_powered_heart_disease_risk_assessment_app"
```

### Step 1 — Create a virtual environment

```powershell
py -3.12 -m venv venv
```

If the `py` launcher is unavailable, use `python -m venv venv` instead.

### Step 2 — Activate it

**PowerShell:**

```powershell
venv\Scripts\Activate.ps1
```

If PowerShell blocks the script with an execution-policy error, allow local scripts for your user once, then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Command Prompt (cmd.exe):**

```cmd
venv\Scripts\activate.bat
```

Your prompt should now be prefixed with `(venv)`.

### Step 3 — Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r oa_system\requirements.txt
```

### Step 4 — Launch the application

```powershell
cd oa_system
streamlit run app.py --server.port 8502
```

Then open <http://localhost:8502> in your browser. Streamlit usually opens it for you.

To stop the server press `Ctrl + C`. To leave the environment, run `deactivate`.

---

## Quick start — macOS and Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r oa_system/requirements.txt
cd oa_system
streamlit run app.py --server.port 8502
```

---

## What happens on first launch

The application bootstraps itself, so no manual setup step is needed:

1. Creates the SQLite database `oa_system/oa_clinic.db` (users, patients, predictions, audit log, settings).
2. Seeds the four demo accounts and three demo patients.
3. If `oa_system/artifacts/` is missing or was built by an older pipeline, trains all four models and writes the artefacts.

The first launch therefore takes a few minutes while training and SHAP run. Later launches start immediately. Every metric shown in the interface is read from `oa_system/artifacts/` — nothing is hard-coded in the interface.

---

## Demo accounts

| Role | Username | Password | Access |
|------|----------|----------|--------|
| Clinician / Doctor | `clinician` | `Clinic#2026` | All pages except Administration |
| Healthcare professional | `healthcare` | `Health#2026` | All pages except Administration |
| Administrator | `admin` | `Admin#2026` | All pages, including user management and retraining |
| Researcher | `researcher` | `Research#2026` | Read-only analytics; cannot create predictions |

Passwords are bcrypt-hashed in the database. They are seeded in source code for demonstration and would be replaced in any real deployment.

---

## Application pages

1. Dashboard
2. Patient management — add, search, edit, archive, history
3. OA risk prediction
4. Prediction explanation (SHAP)
5. Risk factor analytics
6. Model comparison
7. Fairness dashboard
8. Data analytics
9. Prediction history
10. Reports (PDF)
11. Architecture
12. Administration

---

## The clinical form

Only these twelve fields are collected. Interaction and squared terms are computed internally and never entered by hand.

| Field | Meaning | NHANES source |
|-------|---------|---------------|
| AGE | Age in years, 40–79 | `RIDAGEYR` |
| SEX | 1 male, 2 female | `RIAGENDR` |
| BMI | Body mass index | `BMXBMI` |
| RAC | 0 Other, 1 Caucasian, 2 African American, 3 Asian | `RIDRETH3` / `RIDRETH1` |
| SMK | Ever smoked at least 100 cigarettes | `SMQ020` |
| OSP | Osteoporosis | `OSQ060` |
| DIA | Diabetes (borderline counted as no) | `DIQ010` |
| HTN | Hypertension | `BPQ020` |
| PA | Regular physical activity | `PAQ650/665/605/620` |
| KPN | Difficulty kneeling or stooping | `PFQ061C` |
| MOB | Walking or stair difficulty | `PFQ061A/B/C` |
| GH | Self-rated general health, 1 excellent to 5 poor | `HSD010` |

Target is `KOA`, self-reported osteoarthritis. Cases are adults reporting arthritis of the osteoarthritis type; controls report no arthritis. Other arthritis types are excluded rather than treated as controls. The training table holds **1,756 adults sampled 1:1** (878 cases, 878 controls).

---

## Results on the held-out test set

Test fold `n = 352`, threshold `t = 0.503` locked on the validation fold. Source: `oa_system/artifacts/comparison.json`.

| Model | Accuracy | F1 | ROC-AUC |
|-------|---------:|---:|--------:|
| **Logistic Regression** (production) | **0.739** | 0.746 | 0.789 |
| Linear SVM | 0.739 | 0.760 | 0.789 |
| Random Forest | 0.733 | 0.746 | 0.792 |
| XGBoost | 0.722 | 0.734 | 0.763 |

The production model is chosen by accuracy, then ROC-AUC. Logistic Regression reaches sensitivity 0.767 and specificity 0.710, with a Wilson 95% confidence interval on accuracy of 0.69 to 0.78. Results are for the recorded split seed (`split_random_state = 5`); a different stratified split can land slightly lower.

Model version string: `oa-v4-nhanes-logisticregression-f4788d8f2bb5`.

### Risk bands

Bands are derived from the validation fold, not from arbitrary round numbers:

| Band | Probability | Rule used to set the cut |
|------|-------------|--------------------------|
| Low | ≤ 0.430 | Highest threshold still giving sensitivity ≥ 0.80 |
| Moderate | 0.430 – 0.617 | The interval in between |
| High | ≥ 0.617 | Lowest threshold giving specificity ≥ 0.80 |

---

## Retraining and rebuilding the dataset

Retraining is available in the interface under **Administration → Retrain all models**, or from the command line.

**Windows:**

```powershell
cd oa_system
python -c "from services.train import clear_model_caches, train_and_persist; clear_model_caches(); train_and_persist()"
```

**macOS / Linux:**

```bash
cd oa_system
python -c "from services.train import clear_model_caches, train_and_persist; clear_model_caches(); train_and_persist()"
```

To rebuild the training CSV from the raw NHANES `.xpt` files in `oa_system/data/nhanes/`:

```powershell
python oa_system\scripts\build_nhanes_oa.py
```

Random seeds are pinned throughout (cohort sampling, split, cross-validation, imputer, SHAP background), so a rebuild reproduces the same numbers.

---

## Documentation

| File | Contents |
|------|----------|
| `Project_Documentation/Chapter_5_Implementation_and_Experimental_Setup.md` | Architecture, ETL, imputation, feature engineering, training protocol, thresholds |
| `Project_Documentation/Chapter_6_Results_Evaluation_and_Discussion.md` | Held-out results, calibration, SHAP, fairness, discussion, limitations |
| `Project_Documentation/figures/` | Figures 5.1–5.3 and 6.1–6.8 as 300 dpi PNG and vector PDF |
| `Project_Documentation/figures/generate_chapter5_figures.py` | Regenerates the Chapter 5 figures |
| `Project_Documentation/figures/generate_chapter6_figures.py` | Regenerates the Chapter 6 figures from live artefacts |
| `oa_system/ARCHITECTURE.md` | Module and persistence overview |

---

## Running the legacy heart-disease app

`App/heart_app.py` belongs to an earlier project and is kept for reference only. Its dependencies are pinned to old releases (`pandas==1.5.3`, `numpy==1.24.3`, `shap==0.40.0`) which conflict with the OA system's requirements, so it needs **its own separate virtual environment**. Those pinned versions predate Python 3.12 and may require Python 3.10 or 3.11 to install successfully.

**Windows:**

```powershell
py -3.11 -m venv venv_heart
venv_heart\Scripts\Activate.ps1
pip install -r App\requirements.txt
cd App
streamlit run heart_app.py
```

**macOS / Linux:**

```bash
python3.11 -m venv venv_heart
source venv_heart/bin/activate
pip install -r App/requirements.txt
cd App
streamlit run heart_app.py
```

It serves on port 8501 by default, so both applications can run at the same time.

---

## Troubleshooting

**`streamlit` is not recognised as a command (Windows).** The virtual environment is not active. Re-run `venv\Scripts\Activate.ps1` and confirm your prompt shows `(venv)`. As a fallback, use `python -m streamlit run app.py --server.port 8502`.

**PowerShell refuses to run the activation script.** Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, answer yes, then activate again. This affects only your user account.

**Port 8502 is already in use.** Pick another port, for example `--server.port 8503`.

**XGBoost fails to load on Windows.** Install the Microsoft Visual C++ Redistributable (x64), then restart the terminal.

**Blank dashboard or stale metrics.** Delete the `oa_system/artifacts/` folder and restart the app to force a clean retrain, or use **Administration → Retrain all models**.

**Want to reset all patients, predictions and users.** Stop the app and delete `oa_system/oa_clinic.db`. It is recreated and reseeded on the next launch.

**Training seems slow on first launch.** This is expected. Four models with randomised search, calibration, and SHAP take a few minutes. Subsequent launches load the cached artefacts.

---

## Notes

- The label is NHANES **self-reported** osteoarthritis, not a radiographic Kellgren–Lawrence grade. External validity to imaging-confirmed OA is untested.
- Controls were down-sampled to a 1:1 ratio, so accuracy and net-benefit figures are not population-prevalence estimates.
- Osteoporosis (`OSP`) is missing for about 54% of rows because two NHANES cycles omit the questionnaire file; values are imputed.
- Fairness gaps are largest across age groups, which is expected when age is the strongest predictor and a single global threshold is applied. The fairness dashboard reports the gaps rather than hiding them behind one score.
- No licence file is included with this project folder. NHANES source data is public domain, published by the CDC.
