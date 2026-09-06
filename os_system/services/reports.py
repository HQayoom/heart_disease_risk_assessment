import io
import json
from datetime import datetime, timezone

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from config import DISCLAIMER, FEATURE_LABELS, SEX_MAP, RACE_MAP, SMOKE_MAP, SYSTEM_NAME, OSP_MAP, YES_NO_MAP


def build_pdf(patient: dict, prediction: dict) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="OA Risk Report")
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(SYSTEM_NAME, styles["Title"]))
    story.append(Paragraph("Clinical / research decision-support report", styles["Normal"]))
    story.append(Paragraph(datetime.now(timezone.utc).strftime("Generated %Y-%m-%d %H:%M UTC"), styles["Normal"]))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Patient information", styles["Heading2"]))
    story.append(
        Paragraph(
            f"ID: {patient['patient_code']} &nbsp;&nbsp; Name: {patient['full_name']}<br/>"
            f"Age: {patient['AGE']} &nbsp; Sex: {SEX_MAP.get(patient['SEX'], patient['SEX'])} "
            f"&nbsp; BMI: {patient['BMI']}<br/>"
            f"Race: {RACE_MAP.get(patient['RAC'], patient['RAC'])} &nbsp; "
            f"Smoking: {SMOKE_MAP.get(patient['SMK'], patient['SMK'])} &nbsp; "
            f"Osteoporosis: {OSP_MAP.get(patient['OSP'], patient['OSP'])}<br/>"
            f"Diabetes: {YES_NO_MAP.get(patient.get('DIA', 0), patient.get('DIA', 0))} &nbsp; "
            f"Hypertension: {YES_NO_MAP.get(patient.get('HTN', 0), patient.get('HTN', 0))} &nbsp; "
            f"Activity: {YES_NO_MAP.get(patient.get('PA', 0), patient.get('PA', 0))} &nbsp; "
            f"Kneeling difficulty: {YES_NO_MAP.get(patient.get('KPN', 0), patient.get('KPN', 0))} &nbsp; "
            f"Walking/stair difficulty: {YES_NO_MAP.get(patient.get('MOB', 0), patient.get('MOB', 0))}<br/>"
            f"General health: {patient.get('GH', '')}",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("OA risk prediction", styles["Heading2"]))
    ci = ""
    if prediction.get("ci_low") is not None:
        ci = f"<br/>Validation-neighbourhood observed KOA rate: {prediction['ci_low']*100:.1f}–{prediction['ci_high']*100:.1f}% (Wilson 95%)"
    story.append(
        Paragraph(
            f"Risk classification: <b>{prediction['risk_band']}</b><br/>"
            f"Predicted probability of follow-up knee OA: <b>{float(prediction['probability'])*100:.1f}%</b>{ci}<br/>"
            f"Model: {prediction['model_name']}<br/>"
            f"Model version: {prediction.get('model_version') or 'n/a'}<br/>"
            f"Prediction date: {prediction.get('created_at','')}",
            styles["Normal"],
        )
    )
    shap_map = prediction.get("shap_values") or {}
    if isinstance(shap_map, str):
        shap_map = json.loads(shap_map)
    ranked = sorted(shap_map.items(), key=lambda kv: abs(kv[1]), reverse=True)
    story.append(Paragraph("Top contributing factors (local SHAP)", styles["Heading2"]))
    for feat, val in ranked:
        direction = "increases" if val > 0 else "decreases"
        story.append(
            Paragraph(
                f"{FEATURE_LABELS.get(feat, feat)}: SHAP {val:+.3f} ({direction} predicted OA risk)",
                styles["Normal"],
            )
        )
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph("Disclaimer", styles["Heading2"]))
    story.append(Paragraph(DISCLAIMER, styles["Normal"]))
    doc.build(story)
    return buf.getvalue()
