from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
import pandas as pd

from config import FEATURE_COLUMNS, TARGET_COLUMN

BINARY_COLUMNS = ["SEX", "RAC", "SMK", "OSP", "DIA", "HTN", "PA", "KPN", "MOB"]
ENGINEERED_COLUMNS = FEATURE_COLUMNS + [
    "BMI_SQ",
    "BMI_AGE",
    "BMI_OSP",
    "AGE_SQ",
    "BMI_DIA",
    "BMI_KPN",
    "AGE_KPN",
    "AGE_HTN",
    "BMI_MOB",
    "GH_AGE",
]


def impute_features(df: pd.DataFrame, imputer: IterativeImputer | None = None):
    work = df.copy()
    for col in FEATURE_COLUMNS:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    fit_imputer = imputer
    if fit_imputer is None:
        try:
            fit_imputer = IterativeImputer(max_iter=15, random_state=42, skip_complete=True)
        except TypeError:
            fit_imputer = IterativeImputer(max_iter=15, random_state=42)
        matrix = fit_imputer.fit_transform(work[FEATURE_COLUMNS])
    else:
        matrix = imputer.transform(work[FEATURE_COLUMNS])
    imputed = pd.DataFrame(matrix, columns=FEATURE_COLUMNS, index=work.index)
    imputed["SEX"] = imputed["SEX"].round().clip(1, 2)
    imputed["RAC"] = imputed["RAC"].round().clip(0, 3)
    for col in ("SMK", "OSP", "DIA", "HTN", "PA", "KPN", "MOB"):
        imputed[col] = imputed[col].round().clip(0, 1)
    if "GH" in imputed.columns:
        imputed["GH"] = imputed["GH"].clip(1, 5)
    return imputed, fit_imputer


def engineer(X: pd.DataFrame) -> pd.DataFrame:
    out = X[FEATURE_COLUMNS].copy()
    age = out["AGE"].astype(float)
    bmi = out["BMI"].astype(float)
    osp = out["OSP"].astype(float)
    dia = out["DIA"].astype(float)
    kpn = out["KPN"].astype(float)
    htn = out["HTN"].astype(float)
    mob = out["MOB"].astype(float)
    gh = out["GH"].astype(float)
    out["BMI_SQ"] = bmi ** 2
    out["BMI_AGE"] = bmi * age
    out["BMI_OSP"] = bmi * osp
    out["AGE_SQ"] = age ** 2
    out["BMI_DIA"] = bmi * dia
    out["BMI_KPN"] = bmi * kpn
    out["AGE_KPN"] = age * kpn
    out["AGE_HTN"] = age * htn
    out["BMI_MOB"] = bmi * mob
    out["GH_AGE"] = gh * age
    return out[ENGINEERED_COLUMNS]


def prepare_supervised(df: pd.DataFrame, imputer=None):
    work = df.copy()
    work[TARGET_COLUMN] = pd.to_numeric(work[TARGET_COLUMN], errors="coerce")
    labelled = work.dropna(subset=[TARGET_COLUMN]).copy()
    labelled[TARGET_COLUMN] = labelled[TARGET_COLUMN].astype(int)
    X, used_imputer = impute_features(labelled, imputer)
    X[TARGET_COLUMN] = labelled[TARGET_COLUMN].values
    return X, used_imputer
