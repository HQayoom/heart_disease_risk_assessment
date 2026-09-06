import json

import numpy as np
import pandas as pd

from config import ARTIFACTS
from services.features import engineer
from services.train import load_bundle, load_json

MIN_N = 30


def _group_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = len(y_true)
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    acc = (tp + tn) / n if n else 0.0
    selection = float(y_pred.mean()) if n else 0.0
    return {
        "n": n,
        "accuracy": acc,
        "recall": tpr,
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr,
        "selection_rate": selection,
        "unstable": n < MIN_N,
        "note": f"n<{MIN_N}: interpret with caution" if n < MIN_N else "",
    }


def _slice_report(X, y, pred, col, title):
    rows = []
    for level, idx in X.groupby(col, observed=True).groups.items():
        idx = list(idx)
        rows.append({"group": str(level), **_group_metrics(y[idx], pred[idx])})
    stable = [r for r in rows if not r["unstable"]] or rows
    tprs = [r["tpr"] for r in stable]
    fprs = [r["fpr"] for r in stable]
    sels = [r["selection_rate"] for r in stable]
    return {
        "rows": rows,
        "demographic_parity_gap": float(max(sels) - min(sels)) if sels else 0.0,
        "equal_opportunity_gap": float(max(tprs) - min(tprs)) if tprs else 0.0,
        "equalized_odds_gap": float(max(max(tprs) - min(tprs), max(fprs) - min(fprs))) if tprs else 0.0,
        "gaps_exclude_n_lt": MIN_N,
    }


def evaluate_fairness():
    cache = ARTIFACTS / "fairness.json"
    if cache.exists():
        return json.loads(cache.read_text())
    bundle = load_bundle()
    thr = load_json("thresholds.json")["binary_youden"]
    model = bundle["models"][bundle["best_name"]]
    X = bundle["X_test"].copy().reset_index(drop=True)
    y = np.asarray(bundle["y_test"])
    prob = model.predict_proba(engineer(X))[:, 1]
    pred = (prob >= thr).astype(int)
    X["age_group"] = pd.cut(X["AGE"], bins=[0, 54, 64, 120], labels=["40-54", "55-64", "65-79"])
    X["sex_label"] = X["SEX"].map({1: "Male", 2: "Female"})
    X["race_label"] = X["RAC"].map({0: "Other", 1: "Caucasian", 2: "African American", 3: "Asian"}).fillna("Unknown")

    report = {
        "model": bundle["best_name"],
        "threshold": thr,
        "min_n": MIN_N,
        "groups": {},
    }
    for col, title in [("sex_label", "Sex"), ("age_group", "Age group"), ("race_label", "Race")]:
        report["groups"][title] = _slice_report(X, y, pred, col, title)

    if "model_nr" in bundle:
        X_nr = engineer(bundle["X_test"]).drop(columns=["RAC"], errors="ignore")
        pred_nr = (bundle["model_nr"].predict_proba(X_nr)[:, 1] >= thr).astype(int)
        report["without_race"] = _slice_report(X, y, pred_nr, "race_label", "Race")
        try:
            report["ablation"] = json.loads((ARTIFACTS / "ablation.json").read_text())
        except FileNotFoundError:
            report["ablation"] = {}
    cache.write_text(json.dumps(report, indent=2))
    return report
