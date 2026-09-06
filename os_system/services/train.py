import hashlib
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from config import ARTIFACTS, DATA_CSV, FEATURE_COLUMNS, TARGET_COLUMN
from services.features import engineer, prepare_supervised  # re-exported for app analytics

PIPELINE_VERSION = "oa-v4-nhanes"


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def _metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def _youden_threshold(y_true, y_prob) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def _accuracy_threshold(y_true, y_prob) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    candidates = np.unique(np.quantile(y_prob, np.linspace(0.05, 0.95, 91)))
    best_t, best_acc = 0.5, -1.0
    for t in candidates:
        acc = accuracy_score(y_true, (y_prob >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_t


def _operating_points(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    youden = _youden_threshold(y_true, y_prob)
    acc_thr = _accuracy_threshold(y_true, y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    spec = 1 - fpr
    high_candidates = thresholds[(spec >= 0.80) & (thresholds < 1.0)]
    t_high = float(high_candidates.min()) if len(high_candidates) else float(np.quantile(y_prob, 0.80))
    sens_ok = thresholds[(tpr >= 0.80) & (thresholds > 0)]
    t_low = float(sens_ok.max()) if len(sens_ok) else float(np.quantile(y_prob, 0.33))
    if t_low >= t_high:
        t_low, t_high = float(np.quantile(y_prob, 0.40)), float(np.quantile(y_prob, 0.75))
    return {
        "binary_youden": youden,
        "binary_accuracy": acc_thr,
        "low_max": t_low,
        "moderate_max": t_high,
        "target_sensitivity_low": 0.80,
        "target_specificity_high": 0.80,
        "method": (
            "Low/Moderate/High bands are locked on the validation set: Low is at or below the "
            "highest threshold that still yields sensitivity ≥ 0.80; High is at or above the "
            "lowest threshold that yields specificity ≥ 0.80; Moderate is the interval between. "
            "The binary fairness threshold is Youden's J (TPR−FPR) on the same validation fold. "
            "Reported test accuracy uses the validation threshold that maximises accuracy. "
            "Held-out test metrics are reported separately and were not used to pick thresholds."
        ),
    }


def _decision_curve(y_true, y_prob, steps=25):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    prevalence = float(y_true.mean())
    thresholds = np.linspace(0.05, 0.95, steps)
    rows = []
    for t in thresholds:
        pred = y_prob >= t
        tp = float(((pred == 1) & (y_true == 1)).mean())
        fp = float(((pred == 1) & (y_true == 0)).mean())
        nb = tp - fp * (t / (1 - t)) if t < 1 else 0.0
        nb_all = prevalence - (1 - prevalence) * (t / (1 - t))
        rows.append({"threshold": float(t), "net_benefit": nb, "treat_all": nb_all, "treat_none": 0.0})
    return rows


def _search_space():
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    spaces = {
        "Logistic Regression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=4000, class_weight="balanced", random_state=42)),
                ]
            ),
            {"clf__C": np.logspace(-2, 2, 15)},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {
                "n_estimators": [300, 500],
                "max_depth": [4, 6, 10, None],
                "min_samples_leaf": [2, 4, 8],
                "max_features": ["sqrt", 0.8, 1.0],
            },
        ),
        "SVM": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LinearSVC(class_weight="balanced", max_iter=8000, dual="auto", random_state=42),
                    ),
                ]
            ),
            {"clf__C": np.logspace(-1, 2, 12)},
        ),
        "XGBoost": (
            XGBClassifier(
                eval_metric="logloss",
                random_state=42,
                n_jobs=1,
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [3, 4, 6],
                "learning_rate": [0.05, 0.08, 0.12],
                "min_child_weight": [1, 3],
                "subsample": [0.85, 1.0],
                "colsample_bytree": [0.85, 1.0],
            },
        ),
    }
    return spaces, cv


def _fit_calibrated(estimator, X_train, y_train, X_val, y_val, is_svm: bool):
    if is_svm:
        calibrated = CalibratedClassifierCV(estimator, method="sigmoid", cv=3)
        calibrated.fit(X_train, y_train)
        return calibrated
    estimator.fit(X_train, y_train)
    try:
        calibrated = CalibratedClassifierCV(estimator, method="isotonic", cv="prefit")
        calibrated.fit(X_val, y_val)
        return calibrated
    except (TypeError, ValueError):
        calibrated = CalibratedClassifierCV(estimator, method="sigmoid", cv=3)
        calibrated.fit(X_train, y_train)
        return calibrated


def train_and_persist():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    labelled, imputer = prepare_supervised(raw)
    y = labelled[TARGET_COLUMN]
    X_raw = labelled[FEATURE_COLUMNS]
    X_eng = engineer(X_raw)

    X_tmp, X_test, y_tmp, y_test = train_test_split(X_eng, y, test_size=0.20, random_state=5, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=5, stratify=y_tmp)

    X_raw_tmp, X_raw_test, _, _ = train_test_split(X_raw, y, test_size=0.20, random_state=5, stratify=y)
    X_raw_train, X_raw_val, _, _ = train_test_split(X_raw_tmp, y_tmp, test_size=0.25, random_state=5, stratify=y_tmp)

    spaces, cv = _search_space()
    comparison = {}
    curves = {}
    fitted = {}
    cv_scores = {}

    for name, (estimator, grid) in spaces.items():
        print(f"Training {name}...", flush=True)
        search = RandomizedSearchCV(
            estimator,
            grid,
            n_iter=10,
            scoring="roc_auc",
            cv=cv,
            random_state=42,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        cv_scores[name] = float(search.best_score_)
        model = _fit_calibrated(search.best_estimator_, X_train, y_train, X_val, y_val, is_svm=(name == "SVM"))
        fitted[name] = model
        test_prob = model.predict_proba(X_test)[:, 1]
        val_prob = model.predict_proba(X_val)[:, 1]
        acc_tmp = _accuracy_threshold(y_val, val_prob)
        test_pred = (test_prob >= acc_tmp).astype(int)
        comparison[name] = _metrics(y_test, test_pred, test_prob)
        comparison[name]["cv_roc_auc"] = cv_scores[name]
        comparison[name]["val_roc_auc"] = float(roc_auc_score(y_val, val_prob))
        comparison[name]["confusion_matrix"] = confusion_matrix(y_test, test_pred).tolist()
        comparison[name]["best_params"] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in search.best_params_.items()}
        print(
            f"  {name}: acc={comparison[name]['accuracy']:.3f} auc={comparison[name]['roc_auc']:.3f}",
            flush=True,
        )
        fpr, tpr, _ = roc_curve(y_test, test_prob)
        prec, rec, _ = precision_recall_curve(y_test, test_prob)
        frac_pos, mean_pred = calibration_curve(y_test, test_prob, n_bins=8, strategy="quantile")
        curves[name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "mean_predicted": mean_pred.tolist(),
            "fraction_positive": frac_pos.tolist(),
        }

    best_name = max(comparison, key=lambda n: (comparison[n]["accuracy"], comparison[n]["roc_auc"]))
    best = fitted[best_name]
    val_prob = best.predict_proba(X_val)[:, 1]
    test_prob = best.predict_proba(X_test)[:, 1]
    thresholds = _operating_points(y_val, val_prob)
    thresholds["decision_curve"] = _decision_curve(y_test, test_prob)
    thresholds["pipeline_version"] = PIPELINE_VERSION
    thresholds["split_random_state"] = 5

    # Race ablation: same best family without RAC / engineered RAC-linked terms stay
    X_train_nr = X_train.drop(columns=["RAC"], errors="ignore")
    X_val_nr = X_val.drop(columns=["RAC"], errors="ignore")
    X_test_nr = X_test.drop(columns=["RAC"], errors="ignore")
    base_no_race, grid_no_race = spaces[best_name]
    search_nr = RandomizedSearchCV(base_no_race, grid_no_race, n_iter=10, scoring="roc_auc", cv=cv, random_state=42, n_jobs=1)
    search_nr.fit(X_train_nr, y_train)
    model_nr = _fit_calibrated(search_nr.best_estimator_, X_train_nr, y_train, X_val_nr, y_val, is_svm=(best_name == "SVM"))
    ablation = {
        "with_race_test_auc": comparison[best_name]["roc_auc"],
        "without_race_test_auc": float(roc_auc_score(y_test, model_nr.predict_proba(X_test_nr)[:, 1])),
        "note": "Race removed from predictors to test whether RAC is a proxy. Form still collects race for fairness audit only when the production model includes it.",
    }

    payload = json.dumps(comparison, sort_keys=True).encode()
    model_hash = hashlib.sha256(payload + PIPELINE_VERSION.encode()).hexdigest()[:12]
    version = f"{PIPELINE_VERSION}-{best_name.replace(' ', '').lower()}-{model_hash}"

    joblib.dump(
        {
            "models": fitted,
            "best_name": best_name,
            "feature_columns": FEATURE_COLUMNS,
            "engineered_columns": list(X_eng.columns),
            "imputer": imputer,
            "X_background": X_raw_train.sample(min(160, len(X_raw_train)), random_state=42),
            "X_test": X_raw_test,
            "X_test_eng": X_test,
            "y_test": y_test,
            "X_val": X_raw_val,
            "y_val": y_val,
            "val_prob": val_prob,
            "model_nr": model_nr,
            "version": version,
        },
        ARTIFACTS / "bundle.joblib",
    )
    (ARTIFACTS / "comparison.json").write_text(json.dumps(comparison, indent=2, default=str))
    (ARTIFACTS / "curves.json").write_text(json.dumps(curves))
    (ARTIFACTS / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (ARTIFACTS / "ablation.json").write_text(json.dumps(ablation, indent=2))
    (ARTIFACTS / "best_model.txt").write_text(best_name)
    (ARTIFACTS / "model_version.txt").write_text(version)
    labelled.to_csv(ARTIFACTS / "clean_training.csv", index=False)
    return comparison, best_name, thresholds


def artifacts_ready() -> bool:
    version_file = ARTIFACTS / "model_version.txt"
    return (ARTIFACTS / "bundle.joblib").exists() and version_file.exists() and PIPELINE_VERSION in version_file.read_text()


def load_bundle():
    return joblib.load(ARTIFACTS / "bundle.joblib")


def load_json(name: str):
    return json.loads((ARTIFACTS / name).read_text())


def clear_model_caches():
    for name in ("comparison.json", "curves.json", "thresholds.json", "fairness.json", "global_shap.json", "ablation.json", "bundle.joblib", "best_model.txt", "model_version.txt"):
        path = ARTIFACTS / name
        if path.exists():
            path.unlink()
