from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_CSV = ROOT / "data" / "osteoarthritis.csv"
ARTIFACTS = ROOT / "artifacts"
DB_PATH = ROOT / "oa_clinic.db"
SECRET_PEPPER = "oa-explainable-prototype-pepper"

SYSTEM_NAME = "Explainable Osteoarthritis Risk Prediction and Healthcare Analytics System"
SYSTEM_SHORT = "OA Healthcare Analytics"

FEATURE_COLUMNS = ["AGE", "SEX", "BMI", "RAC", "SMK", "OSP", "DIA", "HTN", "PA", "KPN", "MOB", "GH"]
TARGET_COLUMN = "KOA"

FEATURE_LABELS = {
    "AGE": "Age (years)",
    "SEX": "Sex",
    "BMI": "Body mass index",
    "RAC": "Race",
    "SMK": "Smoking status",
    "OSP": "Osteoporosis",
    "DIA": "Diabetes",
    "HTN": "Hypertension",
    "PA": "Regular physical activity",
    "KPN": "Difficulty kneeling / stooping",
    "MOB": "Walking or stair difficulty",
    "GH": "Self-rated general health (1=excellent … 5=poor)",
}

SEX_MAP = {1: "Male", 2: "Female"}
RACE_MAP = {0: "Other", 1: "Caucasian", 2: "African American", 3: "Asian"}
SMOKE_MAP = {0: "Non-smoker", 1: "Smoker"}
OSP_MAP = {0: "Negative", 1: "Positive"}
YES_NO_MAP = {0: "No", 1: "Yes"}
GH_MAP = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}

RISK_LABELS = ("Low", "Moderate", "High")

PATIENT_FIELDS = FEATURE_COLUMNS

DISCLAIMER = (
    "This system is a decision-support and research prototype. "
    "It is not a medical device and must not replace professional clinical diagnosis, "
    "imaging, or treatment decisions."
)

DATASET_CITATION = (
    "Training data are public NHANES 2011–2018 questionnaire and examination files from CDC "
    "(DEMO, BMX, MCQ, SMQ, DIQ, BPQ, PAQ, OSQ, PFQ). Adults aged 40–79 with self-reported "
    "osteoarthritis (MCQ160A + MCQ195=osteoarthritis) are cases; adults with no arthritis "
    "are controls, sampled 1:1. Features: age, sex, BMI, race, smoking, osteoporosis, "
    "diabetes, hypertension, recreational/work physical activity, kneeling/stooping "
    "difficulty (PFQ061C), walking/stair difficulty (PFQ061A/B), and self-rated general "
    "health (HSD010). The original 6-column MatchThem/OAI extract is kept as osteoarthritis_matchthem.csv."
)
