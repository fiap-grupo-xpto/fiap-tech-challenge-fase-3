import os
import sqlite3


def get_database_path() -> str:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.getenv(
        "ASSISTANT_DB_PATH",
        os.path.join(base_dir, "data", "hospital.db"),
    )


def ensure_database_exists() -> str:
    db_path = get_database_path()
    if os.path.exists(db_path):
        return db_path

    try:
        from backend.data.bootstrap_hospital_db import bootstrap
    except Exception as exc:
        raise RuntimeError(
            "Assistant database not found and bootstrap script could not be imported."
        ) from exc

    return bootstrap(db_path)


def get_connection() -> sqlite3.Connection:
    db_path = ensure_database_exists()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

