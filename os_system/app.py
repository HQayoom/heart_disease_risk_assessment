import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from auth import authenticate, create_user, list_users, seed_users, set_user_active, update_password
from config import (
    DATASET_CITATION,
    DISCLAIMER,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    GH_MAP,
    OSP_MAP,
    PATIENT_FIELDS,
    RACE_MAP,
    SEX_MAP,
    SMOKE_MAP,
    SYSTEM_NAME,
    SYSTEM_SHORT,
    YES_NO_MAP,
)
from db import get_conn, init_db, log_audit
from services.explain import global_shap, predict_one, risk_band
from services.fairness import evaluate_fairness
from services.patients import (
    archive_patient,
    get_patient,
    get_prediction,
    list_patients,
    list_predictions,
    save_prediction,
    seed_demo_patients,
    upsert_patient,
)
from services.reports import build_pdf
from services.train import (
    artifacts_ready,
    clear_model_caches,
    load_bundle,
    load_json,
    load_raw,
    prepare_supervised,
    train_and_persist,
)
from ui import (
    chart,
    dashboard_summary,
    inject_css,
    kpi_cards,
    page_header,
    prediction_outcome,
    risk_badge,
    risk_threshold_legend,
)

st.set_page_config(page_title=SYSTEM_SHORT, layout="wide", page_icon="🩺")


def bootstrap():
    init_db()
    seed_users()
    seed_demo_patients()
    if not artifacts_ready():
        with st.spinner("Training and calibrating Logistic Regression, Random Forest, SVM, and XGBoost..."):
            train_and_persist()


def feature_form(defaults=None, key="form"):
    d = defaults or {}
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", 40, 79, int(d.get("AGE", 58)), key=f"{key}_age")
        sex_label = st.selectbox("Sex", list(SEX_MAP.values()), index=0 if d.get("SEX", 1) == 1 else 1, key=f"{key}_sex")
        bmi = st.number_input("BMI", 15.0, 60.0, float(d.get("BMI", 28.0)), step=0.1, key=f"{key}_bmi")
        race_label = st.selectbox(
            "Race",
            list(RACE_MAP.values()),
            index=list(RACE_MAP.keys()).index(int(d.get("RAC", 1))) if d.get("RAC") in RACE_MAP else 1,
            key=f"{key}_rac",
        )
    with c2:
        smk_label = st.selectbox("Smoking status", list(SMOKE_MAP.values()), index=int(d.get("SMK", 0) or 0), key=f"{key}_smk")
        osp_label = st.selectbox("Osteoporosis", list(OSP_MAP.values()), index=int(d.get("OSP", 0) or 0), key=f"{key}_osp")
        dia_label = st.selectbox("Diabetes", list(YES_NO_MAP.values()), index=int(d.get("DIA", 0) or 0), key=f"{key}_dia")
    with c3:
        htn_label = st.selectbox("Hypertension", list(YES_NO_MAP.values()), index=int(d.get("HTN", 0) or 0), key=f"{key}_htn")
        pa_label = st.selectbox("Regular physical activity", list(YES_NO_MAP.values()), index=int(d.get("PA", 0) or 0), key=f"{key}_pa")
        kpn_label = st.selectbox(
            "Difficulty kneeling / stooping",
            list(YES_NO_MAP.values()),
            index=int(d.get("KPN", 0) or 0),
            key=f"{key}_kpn",
        )
        mob_label = st.selectbox(
            "Walking or stair difficulty",
            list(YES_NO_MAP.values()),
            index=int(d.get("MOB", 0) or 0),
            key=f"{key}_mob",
        )
        gh_label = st.selectbox(
            "Self-rated general health",
            list(GH_MAP.values()),
            index=min(max(int(d.get("GH", 3) or 3) - 1, 0), 4),
            key=f"{key}_gh",
        )
    return {
        "AGE": age,
        "SEX": [k for k, v in SEX_MAP.items() if v == sex_label][0],
        "BMI": bmi,
        "RAC": [k for k, v in RACE_MAP.items() if v == race_label][0],
        "SMK": [k for k, v in SMOKE_MAP.items() if v == smk_label][0],
        "OSP": [k for k, v in OSP_MAP.items() if v == osp_label][0],
        "DIA": [k for k, v in YES_NO_MAP.items() if v == dia_label][0],
        "HTN": [k for k, v in YES_NO_MAP.items() if v == htn_label][0],
        "PA": [k for k, v in YES_NO_MAP.items() if v == pa_label][0],
        "KPN": [k for k, v in YES_NO_MAP.items() if v == kpn_label][0],
        "MOB": [k for k, v in YES_NO_MAP.items() if v == mob_label][0],
        "GH": [k for k, v in GH_MAP.items() if v == gh_label][0],
    }


def shap_waterfall(shap_map: dict, expected: float, probability: float):
    ordered = sorted(shap_map.items(), key=lambda kv: abs(kv[1]), reverse=True)
    fig = go.Figure(
        go.Waterfall(
            x=["Base rate"] + [FEATURE_LABELS[k] for k, _ in ordered] + ["Predicted risk"],
            y=[expected] + [v for _, v in ordered] + [probability],
            measure=["absolute"] + ["relative"] * len(ordered) + ["total"],
            connector={"line": {"color": "#95a5a6"}},
        )
    )
    fig.update_layout(title="SHAP waterfall", showlegend=False)
    chart(fig, 400)


def shap_bar(shap_map: dict, title="Local SHAP contributions"):
    labels = [FEATURE_LABELS[k] for k in shap_map]
    vals = list(shap_map.values())
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=["#B91C1C" if v > 0 else "#0F766E" for v in vals]))
    fig.update_layout(title=title, xaxis_title="Contribution to predicted OA probability")
    chart(fig, 360)


def _pretty_table(rows, columns):
    if not rows:
        st.info("Nothing to show yet.")
        return
    df = pd.DataFrame(rows)
    keep = [c for c in columns if c in df.columns]
    st.dataframe(df[keep], use_container_width=True, hide_index=True)


def page_dashboard(user):
    page_header("Overview", "Dashboard", "Patient counts, stored predictions, and test-set model performance from trained artifacts.")
    preds = list_predictions()
    patients = list_patients()
    comparison = load_json("comparison.json")
    best = Path(ROOT / "artifacts" / "best_model.txt").read_text().strip()
    ranking = global_shap()["ranking"]
    bands = pd.Series([p["risk_band"] for p in preds]).value_counts() if preds else pd.Series(dtype=int)
    dashboard_summary(
        patients=len(patients),
        predictions=len(preds),
        high=int(bands.get("High", 0)),
        moderate=int(bands.get("Moderate", 0)),
        low=int(bands.get("Low", 0)),
        best_model=best,
        roc_auc=float(comparison[best]["roc_auc"]),
        accuracy=float(comparison[best]["accuracy"]),
        top_factor=ranking[0]["label"],
    )
    version = (ROOT / "artifacts" / "model_version.txt").read_text().strip() if (ROOT / "artifacts" / "model_version.txt").exists() else ""
    if version:
        st.caption(f"Model version: `{version}`")
    st.markdown(f'<div class="panel">{DATASET_CITATION}</div>', unsafe_allow_html=True)
    if preds:
        st.subheader("Recent predictions")
        show = pd.DataFrame(preds).head(12)
        show["probability"] = (show["probability"] * 100).round(1).astype(str) + "%"
        _pretty_table(show.to_dict("records"), ["patient_code", "full_name", "created_at", "risk_band", "probability", "model_name"])


def page_patients(user):
    page_header(
        "Records",
        "Patient management",
        "Create, search, edit and archive patients. The form matches the trained NHANES feature set: "
        + ", ".join(FEATURE_LABELS[c] for c in PATIENT_FIELDS)
        + ".",
    )
    read_only = user["role"] == "researcher"
    search = st.text_input("Search", placeholder="ID or name")
    rows = list_patients(search)
    _pretty_table(rows, ["patient_code", "full_name", "AGE", "SEX", "BMI", "DIA", "HTN", "KPN", "MOB", "updated_at"])
    st.subheader("Add or edit")
    if read_only:
        st.info("Researcher role is read-only.")
        return
    names = {f"{r['patient_code']} — {r['full_name']}": r["id"] for r in rows}
    choice = st.selectbox("Existing record", ["<new>"] + list(names))
    existing = None if choice == "<new>" else get_patient(names[choice])
    full_name = st.text_input("Full name", existing["full_name"] if existing else "")
    notes = st.text_area("Clinical notes", existing["notes"] if existing else "")
    feats = feature_form(existing, key="pat")
    if st.button("Save patient"):
        if not full_name.strip():
            st.error("Name is required.")
        else:
            pid = upsert_patient({**feats, "full_name": full_name.strip(), "notes": notes}, user["username"], existing["id"] if existing else None)
            st.success(f"Saved patient #{pid}")
            st.rerun()
    if existing and st.button("Archive patient"):
        archive_patient(existing["id"], user["username"])
        st.rerun()
    if existing:
        st.subheader("Prediction history")
        hist = list_predictions(existing["id"])
        st.dataframe(pd.DataFrame(hist) if hist else pd.DataFrame())


def page_predict(user):
    page_header(
        "Assessment",
        "OA risk prediction",
        "Enter the trained-model features, run the calibrated classifier, and store the prediction with its SHAP explanation.",
    )
    if user["role"] == "researcher":
        st.info("Researchers have read-only access and cannot create new predictions.")
        return
    thr = load_json("thresholds.json")
    risk_threshold_legend(thr["low_max"], thr["moderate_max"], thr["binary_youden"])
    patients = list_patients()
    if not patients:
        st.warning("Add a patient first.")
        return
    labels = {f"{p['patient_code']} — {p['full_name']}": p for p in patients}
    selected = labels[st.selectbox("Patient", list(labels))]
    st.caption("Production form — NHANES-trained features (demographics, comorbidities, activity, kneeling difficulty).")
    feats = feature_form(selected, key="pred")
    bundle = load_bundle()
    model_name = st.selectbox("Model", [bundle["best_name"]] + [m for m in bundle["models"] if m != bundle["best_name"]])
    if st.button("Predict OA risk", type="primary"):
        result = predict_one(feats, model_name)
        upsert_patient({**feats, "full_name": selected["full_name"], "notes": selected.get("notes") or ""}, user["username"], selected["id"])
        pred_id = save_prediction(selected["id"], result, user["username"])
        st.session_state["last_pred_id"] = pred_id
        st.session_state["nav_page"] = "Prediction explanation"
        st.rerun()


def page_explanation(user):
    page_header(
        "Explainability",
        "Prediction explanation",
        "Local SHAP answers why this patient received their risk band; global plots show dataset-level drivers.",
    )
    pred_id = st.session_state.get("last_pred_id")
    hist = list_predictions()
    if not hist:
        st.info("No predictions yet.")
        return
    options = {f"#{r['id']} {r['patient_code']} {r['risk_band']} {r['created_at']}": r["id"] for r in hist}
    default_ix = 0
    if pred_id:
        keys = list(options.values())
        if pred_id in keys:
            default_ix = keys.index(pred_id)
    chosen = st.selectbox("Stored prediction", list(options), index=default_ix)
    row = get_prediction(options[chosen])
    shap_map = json.loads(row["shap_json"])
    prediction_outcome(row["risk_band"], float(row["probability"]), row["model_name"])
    thr = load_json("thresholds.json")
    risk_threshold_legend(thr["low_max"], thr["moderate_max"], thr["binary_youden"])
    if row.get("ci_low") is not None:
        st.caption(
            f"Validation-neighbourhood observed KOA rate: {row['ci_low']*100:.0f}–{row['ci_high']*100:.0f}% "
            f"(Wilson 95%, n={row.get('reliability_n') or 'n/a'}). Model version: {row.get('model_version') or 'n/a'}"
        )
    st.subheader("Main contributing factors (local SHAP)")
    shap_bar(shap_map, "Why did the model make this prediction?")
    shap_waterfall(shap_map, float(row.get("expected_value") or 0.5), float(row["probability"]))
    g = global_shap()
    st.subheader("Global explainability (validation sample)")
    rank_df = pd.DataFrame(g["ranking"])
    chart(px.bar(rank_df, x="mean_abs_shap", y="label", orientation="h", title="Mean |SHAP| feature importance"))
    X = pd.DataFrame(g["X"])
    shap_mat = np.array(g["shap_matrix"])
    long = []
    for i, feat in enumerate(FEATURE_COLUMNS):
        for j in range(len(X)):
            long.append({"feature": FEATURE_LABELS[feat], "shap": shap_mat[j, i], "value": X.iloc[j][feat]})
    long_df = pd.DataFrame(long)
    chart(px.scatter(long_df, x="shap", y="feature", color="value", title="SHAP beeswarm / summary"))
    dep = st.selectbox("SHAP dependence feature", FEATURE_COLUMNS, format_func=lambda x: FEATURE_LABELS[x])
    idx = FEATURE_COLUMNS.index(dep)
    dep_df = pd.DataFrame({"feature_value": X[dep], "shap": shap_mat[:, idx]})
    chart(px.scatter(dep_df, x="feature_value", y="shap", trendline="ols", title=f"SHAP dependence: {FEATURE_LABELS[dep]}"))


def page_risk_factors(user):
    page_header(
        "Research",
        "Risk factor analytics",
        "Rankings from mean |SHAP| on the held-out test sample — strongest predictors of follow-up knee OA in this dataset.",
    )
    g = global_shap()
    st.subheader("Top OA risk factors")
    for i, row in enumerate(g["ranking"], start=1):
        st.write(f"{i}. **{row['label']}** — mean |SHAP| {row['mean_abs_shap']:.4f}")
    X = pd.DataFrame(g["X"])
    shap_mat = np.array(g["shap_matrix"])
    feat = st.selectbox("How does this variable influence predicted OA risk?", FEATURE_COLUMNS, format_func=lambda x: FEATURE_LABELS[x])
    idx = FEATURE_COLUMNS.index(feat)
    df = pd.DataFrame({"value": X[feat], "shap": shap_mat[:, idx]})
    chart(px.scatter(df, x="value", y="shap", title=FEATURE_LABELS[feat]))


def page_models(user):
    page_header(
        "Machine learning",
        "Model comparison",
        "Logistic Regression, Random Forest, SVM and XGBoost — metrics and curves from the held-out test set.",
    )
    comparison = load_json("comparison.json")
    curves = load_json("curves.json")
    table = pd.DataFrame(comparison).T.reset_index().rename(columns={"index": "Model"})
    metric_cols = [c for c in ["accuracy", "precision", "recall", "f1", "roc_auc", "cv_roc_auc", "val_roc_auc"] if c in table.columns]
    st.subheader("Classification metrics (test set)")
    st.dataframe(table[["Model"] + metric_cols].style.format({c: "{:.3f}" for c in metric_cols}), use_container_width=True, hide_index=True)
    best = Path(ROOT / "artifacts" / "best_model.txt").read_text().strip()
    st.success(f"Best model by held-out **test** ROC-AUC: {best}. CV/validation scores were used for tuning and thresholds, not for the headline number.")
    fig = go.Figure()
    for name, c in curves.items():
        fig.add_trace(go.Scatter(x=c["fpr"], y=c["tpr"], name=name, mode="lines"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Chance", line=dict(dash="dash"), mode="lines"))
    fig.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate")
    chart(fig)
    figp = go.Figure()
    for name, c in curves.items():
        figp.add_trace(go.Scatter(x=c["recall"], y=c["precision"], name=name, mode="lines"))
    figp.update_layout(title="Precision-recall curves", xaxis_title="Recall", yaxis_title="Precision")
    chart(figp)
    figc = go.Figure()
    for name, c in curves.items():
        figc.add_trace(go.Scatter(x=c["mean_predicted"], y=c["fraction_positive"], name=name, mode="lines+markers"))
    figc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration", line=dict(dash="dash"), mode="lines"))
    figc.update_layout(title="Calibration curves", xaxis_title="Mean predicted probability", yaxis_title="Fraction of positives")
    chart(figc)
    for i, (name, m) in enumerate(comparison.items()):
        cm = np.array(m["confusion_matrix"])
        heat = px.imshow(cm, text_auto=True, title=f"Confusion matrix — {name}", labels=dict(x="Predicted", y="Actual"))
        chart(heat)
    thr = load_json("thresholds.json")
    if "decision_curve" in thr:
        dc = pd.DataFrame(thr["decision_curve"])
        figd = go.Figure()
        figd.add_trace(go.Scatter(x=dc["threshold"], y=dc["net_benefit"], name="Model"))
        figd.add_trace(go.Scatter(x=dc["threshold"], y=dc["treat_all"], name="Treat all", line=dict(dash="dash")))
        figd.add_trace(go.Scatter(x=dc["threshold"], y=dc["treat_none"], name="Treat none", line=dict(dash="dot")))
        figd.update_layout(title="Decision curve (test set net benefit)", xaxis_title="Threshold", yaxis_title="Net benefit")
        chart(figd)


def page_fairness(user):
    page_header("Equity", "Fairness audit", "Group performance and gaps. Small samples are flagged rather than hidden.")
    report = evaluate_fairness()
    st.write(f"Best model **{report['model']}** at the Youden threshold {report['threshold']:.3f}.")
    st.write(
        f"Gaps use groups with n≥{report.get('min_n', 30)}. Smaller groups are still shown but flagged as unstable. "
        "This is a disparity audit, not a single fairness score."
    )
    if report.get("ablation"):
        ab = report["ablation"]
        st.info(
            f"Race ablation: test AUC with race {ab.get('with_race_test_auc', float('nan')):.3f} vs "
            f"without race {ab.get('without_race_test_auc', float('nan')):.3f}."
        )
    for title, block in report["groups"].items():
        st.subheader(title)
        st.caption(
            f"Demographic parity gap (selection rate): {block['demographic_parity_gap']:.3f} · "
            f"Equal opportunity gap (TPR): {block['equal_opportunity_gap']:.3f} · "
            f"Equalized odds gap: {block['equalized_odds_gap']:.3f}"
        )
        df = pd.DataFrame(block["rows"])
        if not df.empty:
            show = df.rename(
                columns={
                    "group": "Group",
                    "accuracy": "Accuracy",
                    "recall": "Recall",
                    "tpr": "TPR",
                    "fpr": "FPR",
                    "fnr": "FNR",
                    "selection_rate": "Selection rate",
                    "n": "n",
                    "note": "Note",
                }
            )
            cols = [c for c in ["Group", "n", "Accuracy", "Recall", "FPR", "FNR", "Selection rate", "Note"] if c in show.columns]
            st.dataframe(show[cols].style.format({"Accuracy": "{:.3f}", "Recall": "{:.3f}", "FPR": "{:.3f}", "FNR": "{:.3f}", "Selection rate": "{:.3f}"}), use_container_width=True, hide_index=True)


def page_analytics(user):
    page_header("Dataset", "Healthcare analytics", "Distributions, missingness and correlations in the NHANES training file.")
    raw = load_raw()
    if "Unnamed: 0" in raw.columns:
        raw = raw.drop(columns=["Unnamed: 0"])
    clean, _ = prepare_supervised(raw)
    st.metric("Raw rows", len(raw))
    st.metric("Supervised rows (KOA observed)", len(clean))
    miss = raw.isna().sum().rename("missing_count").to_frame()
    miss["missing_pct"] = 100 * miss["missing_count"] / len(raw)
    st.subheader("Missing values")
    st.dataframe(miss)
    st.subheader("Class distribution (KOA)")
    chart(px.histogram(clean, x="KOA", title="0 = at risk, 1 = diagnosed at follow-up"))
    st.subheader("Distributions")
    feat = st.selectbox("Variable", FEATURE_COLUMNS, format_func=lambda x: FEATURE_LABELS[x])
    chart(px.histogram(clean, x=feat, color=clean["KOA"].astype(str), barmode="overlay", opacity=0.7))
    chart(px.box(clean, x=clean["KOA"].astype(str), y=feat))
    st.subheader("Correlations")
    corr = clean[FEATURE_COLUMNS + ["KOA"]].corr()
    chart(px.imshow(corr, text_auto=".2f", title="Pearson correlation heatmap"))
    chart(px.scatter(clean, x="AGE", y="BMI", color=clean["KOA"].astype(str), title="Age vs BMI"))


def page_history(user):
    page_header("Audit trail", "Prediction history", "Every stored assessment keeps its SHAP vector and model version.")
    rows = list_predictions()
    if not rows:
        st.info("No predictions stored.")
        return
    df = pd.DataFrame(rows)
    df["probability_pct"] = (df["probability"] * 100).round(1)
    show = ["patient_code", "created_at", "risk_band", "probability_pct", "model_name"]
    if "model_version" in df.columns:
        show.append("model_version")
    st.dataframe(df[show])
    options = {f"#{r['id']} {r['patient_code']}": r["id"] for r in rows}
    chosen = st.selectbox("Open SHAP explanation", list(options))
    row = get_prediction(options[chosen])
    shap_bar(json.loads(row["shap_json"]))


def page_reports(user):
    page_header("Documents", "Clinical report", "Download a dated PDF with prediction, SHAP drivers and the prototype disclaimer.")
    rows = list_predictions()
    if not rows:
        st.info("Run a prediction first.")
        return
    options = {f"#{r['id']} {r['patient_code']} {r['risk_band']}": r["id"] for r in rows}
    row = get_prediction(options[st.selectbox("Prediction", list(options))])
    patient = get_patient(row["patient_id"])
    payload = {
        **row,
        "shap_values": json.loads(row["shap_json"]),
    }
    pdf = build_pdf(patient, payload)
    st.download_button("Download PDF report", data=pdf, file_name=f"oa_report_{patient['patient_code']}.pdf", mime="application/pdf")
    st.warning(DISCLAIMER)


def page_admin(user):
    page_header("Control plane", "Administration", "Users, roles, passwords, model retraining, audit logs and dataset information.")
    if user["role"] != "administrator":
        st.error("Administrators only.")
        return
    st.subheader("Users and roles")
    st.dataframe(pd.DataFrame(list_users()))
    with st.form("new_user"):
        username = st.text_input("Username")
        full_name = st.text_input("Full name")
        role = st.selectbox("Role", ["clinician", "administrator", "researcher"])
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Create user") and username and password:
            create_user(username, password, full_name or username, role, user["username"])
            st.success("User created")
            st.rerun()
    st.subheader("Password management")
    uname = st.text_input("Username to reset")
    newpw = st.text_input("New password", type="password")
    if st.button("Update password") and uname and newpw:
        update_password(uname, newpw, user["username"])
        st.success("Password updated")
    deactivate = st.number_input("User id to deactivate", min_value=1, step=1)
    if st.button("Deactivate user"):
        set_user_active(int(deactivate), False, user["username"])
        st.success("Updated")
    st.subheader("Model versions")
    st.write("Current production model:", Path(ROOT / "artifacts" / "best_model.txt").read_text().strip())
    st.json(load_json("thresholds.json"))
    if st.button("Retrain all models"):
        clear_model_caches()
        train_and_persist()
        st.success("Retrained.")
        st.rerun()
    st.subheader("Audit log")
    with get_conn() as conn:
        logs = [dict(r) for r in conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT 200")]
    st.dataframe(pd.DataFrame(logs))
    st.subheader("Dataset information")
    st.write(DATASET_CITATION)


def login_view():
    left, right = st.columns((1.05, 1), gap="large")
    with left:
        st.markdown(
            f"""
            <div class="login-hero">
              <div class="brand-mark">{SYSTEM_SHORT}</div>
              <h1>{SYSTEM_NAME}</h1>
              <p>Complete pipeline: patient data → calibrated prediction → SHAP explanation → fairness audit → downloadable report.</p>
              <p>Prototype decision support. Not a diagnostic device.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("#### Login")
        st.caption("Controlled access for clinicians, administrators and researchers (read-only).")
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Continue", use_container_width=True)
        if submitted:
            user = authenticate(username, password)
            if user:
                st.session_state["user"] = {
                    "id": user["id"],
                    "username": user["username"],
                    "full_name": user["full_name"],
                    "role": user["role"],
                }
                st.rerun()
            else:
                st.error("Those credentials were not recognised.")
        with st.expander("Demo accounts"):
            st.markdown(
                """
                | Role | Username | Password |
                | --- | --- | --- |
                | Clinician / Doctor | `clinician` | `Clinic#2026` |
                | Healthcare professional | `healthcare` | `Health#2026` |
                | Administrator | `admin` | `Admin#2026` |
                | Researcher (read-only) | `researcher` | `Research#2026` |
                """
            )


def page_architecture(user):
    page_header(
        "System design",
        "Architecture",
        f"Layered view of the {SYSTEM_NAME}.",
    )
    st.markdown(
        f"""
        <div class="arch-stack">
          <div class="arch-layer"><h4>User</h4>Clinician / Doctor · Administrator · Researcher (read-only)</div>
          <div class="arch-arrow">▼</div>
          <div class="arch-layer"><h4>Web dashboard</h4>Streamlit — 12 workspace pages</div>
          <div class="arch-arrow">▼</div>
          <div class="arch-layer"><h4>Functional modules</h4>Patient management · OA prediction · Data analytics · Model comparison · Fairness</div>
          <div class="arch-arrow">▼</div>
          <div class="arch-layer"><h4>Data preprocessing</h4>Iterative imputation · BMI feature engineering (`services/features.py`)</div>
          <div class="arch-arrow">▼</div>
          <div class="arch-layer"><h4>Machine learning engine</h4>Logistic Regression · Random Forest · SVM · XGBoost (`services/train.py`)</div>
          <div class="arch-arrow">▼</div>
          <div class="arch-layer"><h4>Best model (test ROC-AUC)</h4>Calibrated classifier persisted in `artifacts/bundle.joblib`</div>
          <div class="arch-arrow">▼</div>
          <div class="arch-layer"><h4>SHAP explainer + fairness engine</h4>`services/explain.py` · `services/fairness.py`</div>
          <div class="arch-arrow">▼</div>
          <div class="arch-layer"><h4>Clinical insights + report generation</h4>SQLite prediction history · PDF reports · audit log</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="arch-wrap">
          <div class="arch-col">
            <div class="arch-box">
              <h4>SQLite tables</h4>
              <ul>
                <li>users · patients · predictions</li>
                <li>audit_log · settings</li>
              </ul>
            </div>
          </div>
          <div>
            <div class="arch-row">
              <h4>Backend services</h4>
              <div class="mods">
                <div class="mod">Authentication</div>
                <div class="mod">Patients</div>
                <div class="mod">Preprocessing</div>
                <div class="mod">Prediction</div>
                <div class="mod">SHAP</div>
                <div class="mod">Fairness</div>
                <div class="mod">Reporting</div>
              </div>
            </div>
          </div>
          <div class="arch-col">
            <div class="arch-box">
              <h4>Artifacts</h4>
              <ul>
                <li>comparison.json · curves.json</li>
                <li>thresholds.json · global_shap.json</li>
                <li>fairness.json · bundle.joblib</li>
              </ul>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Email and cloud storage are specified for a production deployment and are not wired in this prototype. "
        "All metrics displayed in the UI are loaded from artifacts generated by `train_and_persist()` — not hard-coded."
    )


PAGES = {
    "Dashboard": page_dashboard,
    "Patient management": page_patients,
    "OA risk prediction": page_predict,
    "Prediction explanation": page_explanation,
    "Risk factor analytics": page_risk_factors,
    "Model comparison": page_models,
    "Fairness dashboard": page_fairness,
    "Data analytics": page_analytics,
    "Prediction history": page_history,
    "Reports": page_reports,
    "Architecture": page_architecture,
    "Administration": page_admin,
}

ROLE_PAGES = {
    "clinician": [k for k in PAGES if k != "Administration"],
    "researcher": [
        "Dashboard",
        "Patient management",
        "Prediction explanation",
        "Risk factor analytics",
        "Model comparison",
        "Fairness dashboard",
        "Data analytics",
        "Prediction history",
        "Reports",
        "Architecture",
    ],
    "administrator": list(PAGES.keys()),
}


def main():
    inject_css()
    bootstrap()
    user = st.session_state.get("user")
    if not user:
        login_view()
        return
    with st.sidebar:
        st.markdown(
            f"""
            <div class="brand-mark">Healthcare analytics</div>
            <div class="brand-title">{SYSTEM_SHORT}</div>
            <div class="brand-sub">Explainable OA risk pipeline</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="user-chip"><strong>{user["full_name"]}</strong><br/>{user["role"].title()}</div>',
            unsafe_allow_html=True,
        )
        pages = ROLE_PAGES[user["role"]]
        current = st.session_state.get("nav_page", pages[0])
        if current not in pages:
            current = pages[0]
        page = st.radio("Workspace", pages, index=pages.index(current), label_visibility="collapsed")
        st.session_state["nav_page"] = page
        if st.button("Sign out", use_container_width=True):
            log_audit(user["username"], "logout", "")
            st.session_state.clear()
            st.rerun()
        st.markdown(f'<p class="disclaimer">{DISCLAIMER}</p>', unsafe_allow_html=True)
    PAGES[page](user)


if __name__ == "__main__":
    main()
