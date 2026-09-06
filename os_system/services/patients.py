from db import get_conn, log_audit, utcnow


def next_code() -> str:
    with get_conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0] + 1
    return f"P{n:04d}"


def upsert_patient(data: dict, actor: str, patient_id: int | None = None) -> int:
    now = utcnow()
    with get_conn() as conn:
        if patient_id:
            conn.execute(
                """UPDATE patients SET full_name=?, AGE=?, SEX=?, BMI=?, RAC=?, SMK=?, OSP=?,
                   DIA=?, HTN=?, PA=?, KPN=?, MOB=?, GH=?, notes=?, updated_at=? WHERE id=?""",
                (
                    data["full_name"], data["AGE"], data["SEX"], data["BMI"], data["RAC"],
                    data["SMK"], data["OSP"], data.get("DIA", 0), data.get("HTN", 0),
                    data.get("PA", 0), data.get("KPN", 0), data.get("MOB", 0), data.get("GH", 3),
                    data.get("notes", ""), now, patient_id,
                ),
            )
            log_audit(actor, "edit_patient", str(patient_id))
            return patient_id
        code = data.get("patient_code") or next_code()
        cur = conn.execute(
            """INSERT INTO patients (patient_code, full_name, AGE, SEX, BMI, RAC, SMK, OSP, DIA, HTN, PA, KPN, MOB, GH, notes,
               archived, created_by, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?)""",
            (
                code, data["full_name"], data["AGE"], data["SEX"], data["BMI"], data["RAC"],
                data["SMK"], data["OSP"], data.get("DIA", 0), data.get("HTN", 0),
                data.get("PA", 0), data.get("KPN", 0), data.get("MOB", 0), data.get("GH", 3),
                data.get("notes", ""), actor, now, now,
            ),
        )
        log_audit(actor, "add_patient", code)
        return int(cur.lastrowid)


def list_patients(search: str = "", include_archived=False):
    q = "SELECT * FROM patients WHERE 1=1"
    params = []
    if not include_archived:
        q += " AND archived=0"
    if search:
        q += " AND (patient_code LIKE ? OR full_name LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])
    q += " ORDER BY id DESC"
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(q, params)]


def get_patient(patient_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone()
    return dict(row) if row else None


def archive_patient(patient_id: int, actor: str):
    with get_conn() as conn:
        conn.execute("UPDATE patients SET archived=1, updated_at=? WHERE id=?", (utcnow(), patient_id))
    log_audit(actor, "archive_patient", str(patient_id))


def save_prediction(patient_id: int, result: dict, actor: str) -> int:
    import json

    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO predictions (patient_id, model_name, probability, risk_band, shap_json, created_by, created_at,
               model_version, ci_low, ci_high, expected_value)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                patient_id,
                result["model_name"],
                result["probability"],
                result["risk_band"],
                json.dumps(result["shap_values"]),
                actor,
                utcnow(),
                result.get("model_version"),
                result.get("ci_low"),
                result.get("ci_high"),
                result.get("expected_value"),
            ),
        )
        pred_id = int(cur.lastrowid)
    log_audit(actor, "predict", f"patient={patient_id} {result['risk_band']}")
    return pred_id


def list_predictions(patient_id: int | None = None):
    q = """SELECT p.*, pt.patient_code, pt.full_name
           FROM predictions p JOIN patients pt ON pt.id = p.patient_id"""
    params = []
    if patient_id:
        q += " WHERE p.patient_id=?"
        params.append(patient_id)
    q += " ORDER BY p.id DESC"
    with get_conn() as conn:
        return [dict(r) for r in conn.execute(q, params)]


def get_prediction(pred_id: int):
    with get_conn() as conn:
        row = conn.execute(
            """SELECT p.*, pt.patient_code, pt.full_name, pt.AGE, pt.SEX, pt.BMI, pt.RAC, pt.SMK, pt.OSP,
                      pt.DIA, pt.HTN, pt.PA, pt.KPN, pt.MOB, pt.GH
               FROM predictions p JOIN patients pt ON pt.id = p.patient_id WHERE p.id=?""",
            (pred_id,),
        ).fetchone()
    return dict(row) if row else None


DEMO_PATIENTS = [
    {"patient_code": "P0001", "full_name": "Demo A — higher risk", "AGE": 72, "SEX": 2, "BMI": 36.2, "RAC": 1, "SMK": 1, "OSP": 1, "DIA": 1, "HTN": 1, "PA": 0, "KPN": 1, "MOB": 1, "GH": 4, "notes": "Seeded demo"},
    {"patient_code": "P0002", "full_name": "Demo B — lower risk", "AGE": 48, "SEX": 1, "BMI": 22.4, "RAC": 1, "SMK": 0, "OSP": 0, "DIA": 0, "HTN": 0, "PA": 1, "KPN": 0, "MOB": 0, "GH": 1, "notes": "Seeded demo"},
    {"patient_code": "P0003", "full_name": "Demo C — mixed profile", "AGE": 61, "SEX": 2, "BMI": 29.1, "RAC": 2, "SMK": 0, "OSP": 0, "DIA": 0, "HTN": 1, "PA": 1, "KPN": 0, "MOB": 0, "GH": 3, "notes": "Seeded demo"},
]


def seed_demo_patients():
    with get_conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        if n:
            return
    for row in DEMO_PATIENTS:
        upsert_patient(row, "system")
