import json

import numpy as np
import pandas as pd
import shap
from statsmodels.stats.proportion import proportion_confint

from config import ARTIFACTS, FEATURE_COLUMNS, FEATURE_LABELS
from services.features import engineer
from services.train import load_bundle, load_json


def risk_band(probability: float) -> str:
    thr = load_json("thresholds.json")
    if probability <= thr["low_max"]:
        return "Low"
    if probability <= thr["moderate_max"]:
        return "Moderate"
    return "High"


def local_observed_rate(probability: float, window: float = 0.08):
    bundle = load_bundle()
    val_p = np.asarray(bundle["val_prob"])
    y_val = np.asarray(bundle["y_val"])
    w = window
    mask = np.abs(val_p - probability) <= w
    while mask.sum() < 25 and w < 0.25:
        w += 0.03
        mask = np.abs(val_p - probability) <= w
    n = int(mask.sum())
    k = int(y_val[mask].sum()) if n else 0
    rate = float(k / n) if n else None
    if n:
        lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    else:
        lo = hi = None
    return {
        "observed_rate": rate,
        "ci_low": float(lo) if lo is not None else None,
        "ci_high": float(hi) if hi is not None else None,
        "n": n,
        "window": w,
    }


def _predict_raw_model(model, raw_X: pd.DataFrame):
    return model.predict_proba(engineer(raw_X))[:, 1]


def predict_one(feature_row: dict, model_name: str | None = None):
    bundle = load_bundle()
    name = model_name or bundle["best_name"]
    model = bundle["models"][name]
    X = pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)
    probability = float(_predict_raw_model(model, X)[0])
    shap_values, expected = local_shap(name, X)
    reliability = local_observed_rate(probability)
    return {
        "model_name": name,
        "probability": probability,
        "risk_band": risk_band(probability),
        "shap_values": shap_values,
        "expected_value": expected,
        "ci_low": reliability["ci_low"],
        "ci_high": reliability["ci_high"],
        "observed_rate": reliability["observed_rate"],
        "reliability_n": reliability["n"],
        "model_version": bundle.get("version", ""),
    }


def local_shap(model_name: str, X: pd.DataFrame):
    bundle = load_bundle()
    model = bundle["models"][model_name]
    background = bundle["X_background"]

    def f(data):
        frame = pd.DataFrame(data, columns=FEATURE_COLUMNS)
        return _predict_raw_model(model, frame)

    explainer = shap.Explainer(f, background)
    explanation = explainer(X)
    row = np.array(explanation.values).reshape(-1)[: len(FEATURE_COLUMNS)]
    expected = float(np.array(explanation.base_values).reshape(-1)[0])
    shap_map = {FEATURE_COLUMNS[i]: float(row[i]) for i in range(len(FEATURE_COLUMNS))}
    return shap_map, expected


def global_shap(max_rows: int = 120):
    cache = ARTIFACTS / "global_shap.json"
    if cache.exists():
        return json.loads(cache.read_text())
    bundle = load_bundle()
    name = bundle["best_name"]
    model = bundle["models"][name]
    X = bundle["X_test"].head(max_rows)

    def f(data):
        frame = pd.DataFrame(data, columns=FEATURE_COLUMNS)
        return _predict_raw_model(model, frame)

    explainer = shap.Explainer(f, bundle["X_background"])
    explanation = explainer(X)
    values = np.array(explanation.values)
    mean_abs = np.abs(values).mean(axis=0)
    ranking = sorted(
        (
            {
                "feature": FEATURE_COLUMNS[i],
                "label": FEATURE_LABELS[FEATURE_COLUMNS[i]],
                "mean_abs_shap": float(mean_abs[i]),
            }
            for i in range(len(FEATURE_COLUMNS))
        ),
        key=lambda d: d["mean_abs_shap"],
        reverse=True,
    )
    payload = {
        "model": name,
        "ranking": ranking,
        "shap_matrix": values.tolist(),
        "X": X.to_dict(orient="list"),
        "expected_value": float(np.mean(explanation.base_values)),
    }
    cache.write_text(json.dumps(payload))
    return payload
