import bcrypt
from db import get_conn, init_db, log_audit, utcnow

DEFAULT_USERS = [
    ("admin", "Admin#2026", "System Administrator", "administrator"),
    ("clinician", "Clinic#2026", "Lead Clinician", "clinician"),
    ("healthcare", "Health#2026", "Healthcare Professional", "clinician"),
    ("researcher", "Research#2026", "Read-only Researcher", "researcher"),
]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def seed_users():
    init_db()
    with get_conn() as conn:
        for username, password, full_name, role in DEFAULT_USERS:
            exists = conn.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone()
            if exists:
                continue
            conn.execute(
                """INSERT INTO users (username, password_hash, full_name, role, is_active, created_at)
                   VALUES (?,?,?,?,1,?)""",
                (username, hash_password(password), full_name, role, utcnow()),
            )


def authenticate(username: str, password: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username=? AND is_active=1",
            (username.strip(),),
        ).fetchone()
    if row and verify_password(password, row["password_hash"]):
        log_audit(username, "login", "successful")
        return dict(row)
    log_audit(username, "login_failed", "invalid credentials")
    return None


def list_users():
    with get_conn() as conn:
        return [dict(r) for r in conn.execute("SELECT id, username, full_name, role, is_active, created_at FROM users ORDER BY id")]


def create_user(username, password, full_name, role, actor):
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO users (username, password_hash, full_name, role, is_active, created_at)
               VALUES (?,?,?,?,1,?)""",
            (username, hash_password(password), full_name, role, utcnow()),
        )
    log_audit(actor, "create_user", username)


def update_password(username, new_password, actor):
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET password_hash=? WHERE username=?",
            (hash_password(new_password), username),
        )
    log_audit(actor, "password_change", username)


def set_user_active(user_id, active, actor):
    with get_conn() as conn:
        conn.execute("UPDATE users SET is_active=? WHERE id=?", (1 if active else 0, user_id))
    log_audit(actor, "set_active", f"user_id={user_id} active={active}")
