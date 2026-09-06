# Explainable Osteoarthritis Risk Prediction and Healthcare Analytics System

Streamlit prototype implementing the full MSc pipeline:

**patient data → prediction → SHAP explanation → fairness audit → PDF report**

## Dataset (production form)

Public **NHANES 2011–2018** adults aged 40–79 (CDC). Osteoarthritis cases vs no-arthritis controls, sampled 1:1.

| Field | Description |
|-------|-------------|
| AGE | Age (years) |
| SEX | Sex |
| BMI | Body mass index |
| RAC | Race |
| SMK | Smoking status |
| OSP | Osteoporosis |
| DIA | Diabetes |
| HTN | Hypertension |
| PA | Regular physical activity |
| KPN | Difficulty kneeling / stooping |
| MOB | Walking or stair difficulty |
| GH | Self-rated general health (1=excellent … 5=poor) |

Target: **KOA** (self-reported osteoarthritis). Rebuild with `python scripts/build_nhanes_oa.py` if the XPT files are present. The original 6-column MatchThem extract is `data/osteoarthritis_matchthem.csv`.

## Run

Requires Python 3.12. Run these from the **project root** (the folder containing `oa_system/`).

**Windows (PowerShell):**

```powershell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
pip install -r oa_system\requirements.txt
cd oa_system
streamlit run app.py --server.port 8502
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once, then activate again. In `cmd.exe` use `venv\Scripts\activate.bat` instead.

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r oa_system/requirements.txt
cd oa_system
streamlit run app.py --server.port 8502
```

Then open <http://localhost:8502>.

On first launch the system seeds users, demo patients and trains four models (Logistic Regression, Random Forest, SVM, XGBoost), so it takes a few minutes. All dashboard metrics are read from `artifacts/` — not hard-coded.

To force a clean retrain, delete `artifacts/` and restart, or use **Administration → Retrain all models**. To reset all stored data, delete `oa_clinic.db`.

## Pages (12)

1. Dashboard  
2. Patient management (add / search / edit / archive / history)  
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

Login is a separate gate before the page set. Access is filtered by role in `ROLE_PAGES`.

## Demo accounts

| Username | Password | Role | Access |
|----------|----------|------|--------|
| `clinician` | `Clinic#2026` | clinician | All pages except Administration |
| `healthcare` | `Health#2026` | clinician | All pages except Administration |
| `admin` | `Admin#2026` | administrator | All 12 pages |
| `researcher` | `Research#2026` | researcher | Read-only; no OA risk prediction, no Administration |

## Risk bands

Low / Moderate / High thresholds are **validation-derived** in `services/train.py` (sensitivity ≥ 80%, specificity ≥ 80%, Youden's J for binary decisions) — not arbitrary 0–39 / 40–69 / 70–100 cut-offs.
