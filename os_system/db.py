import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from config import DB_PATH


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _add_column(conn, table: str, column: str, ddl: str):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db():
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('clinician','administrator','researcher')),
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                action TEXT NOT NULL,
                detail TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_code TEXT UNIQUE NOT NULL,
                full_name TEXT NOT NULL,
                AGE INTEGER NOT NULL,
                SEX INTEGER NOT NULL,
                BMI REAL NOT NULL,
                RAC INTEGER NOT NULL,
                SMK INTEGER NOT NULL,
                OSP INTEGER NOT NULL,
                DIA INTEGER NOT NULL DEFAULT 0,
                HTN INTEGER NOT NULL DEFAULT 0,
                PA INTEGER NOT NULL DEFAULT 0,
                KPN INTEGER NOT NULL DEFAULT 0,
                MOB INTEGER NOT NULL DEFAULT 0,
                GH INTEGER NOT NULL DEFAULT 3,
                notes TEXT,
                archived INTEGER NOT NULL DEFAULT 0,
                created_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                probability REAL NOT NULL,
                risk_band TEXT NOT NULL,
                shap_json TEXT,
                created_by TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(patient_id) REFERENCES patients(id)
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        _add_column(conn, "predictions", "model_version", "TEXT")
        _add_column(conn, "predictions", "ci_low", "REAL")
        _add_column(conn, "predictions", "ci_high", "REAL")
        _add_column(conn, "predictions", "expected_value", "REAL")
        for col, ddl in (("DIA", "INTEGER NOT NULL DEFAULT 0"), ("HTN", "INTEGER NOT NULL DEFAULT 0"),
                         ("PA", "INTEGER NOT NULL DEFAULT 0"), ("KPN", "INTEGER NOT NULL DEFAULT 0"),
                         ("MOB", "INTEGER NOT NULL DEFAULT 0"), ("GH", "INTEGER NOT NULL DEFAULT 3")):
            _add_column(conn, "patients", col, ddl)


def log_audit(username: str | None, action: str, detail: str = ""):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO audit_log (username, action, detail, created_at) VALUES (?,?,?,?)",
            (username, action, detail, utcnow()),
        )
