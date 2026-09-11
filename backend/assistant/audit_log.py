from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from backend.assistant.db import get_connection

logger = logging.getLogger("assistant.audit")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(os.getenv("ASSISTANT_LOG_LEVEL", "INFO"))
logger.propagate = False

_AUDIT_LOG_COLUMNS = (
    "request_id",
    "created_at",
    "patient_id",
    "question",
    "status",
    "assistant_answer",
    "sources_used",
    "llm_backend_used",
    "fallback_used",
    "blocked",
    "block_reason",
    "requires_human_review",
    "input_validation",
    "output_validation",
    "error_message",
    "system_prompt",
    "user_prompt",
    "attempted_backend",
    "attempted_backend_error",
    "raw_answer",
)

# Colunas adicionadas depois da criação inicial da tabela — aplicadas via ALTER TABLE
# para bancos `hospital.db` já existentes que não passaram pelo bootstrap mais recente.
_MIGRATED_COLUMNS = ("system_prompt", "user_prompt", "attempted_backend", "attempted_backend_error", "raw_answer")


def ensure_audit_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            request_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            patient_id TEXT,
            question TEXT,
            status TEXT,
            assistant_answer TEXT,
            sources_used TEXT,
            llm_backend_used TEXT,
            fallback_used INTEGER,
            blocked INTEGER,
            block_reason TEXT,
            requires_human_review INTEGER,
            input_validation TEXT,
            output_validation TEXT,
            error_message TEXT,
            system_prompt TEXT,
            user_prompt TEXT,
            attempted_backend TEXT,
            attempted_backend_error TEXT,
            raw_answer TEXT
        );
        """
    )
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(audit_log);").fetchall()}
    for column in _MIGRATED_COLUMNS:
        if column not in existing_columns:
            conn.execute(f"ALTER TABLE audit_log ADD COLUMN {column} TEXT;")


def log_interaction(entry: Dict[str, Any]) -> None:
    """Registra uma interação do assistente para rastreamento e auditoria.

    Emite um log técnico (stdout, via `logging`) e persiste um registro estruturado
    na tabela `audit_log` do SQLite, consultável posteriormente por `get_interaction`.
    """
    request_id = entry.get("request_id")

    logger.info(
        "assistant_interaction request_id=%s patient_id=%s status=%s blocked=%s "
        "requires_human_review=%s backend=%s",
        request_id,
        entry.get("patient_id"),
        entry.get("status"),
        entry.get("blocked"),
        entry.get("requires_human_review"),
        entry.get("llm_backend_used"),
    )
    if entry.get("blocked"):
        logger.warning(
            "assistant_response_blocked request_id=%s reason=%s",
            request_id,
            entry.get("block_reason"),
        )
    if entry.get("attempted_backend_error"):
        logger.warning(
            "assistant_primary_backend_failed request_id=%s attempted_backend=%s error=%s "
            "(final backend %s)",
            request_id,
            entry.get("attempted_backend"),
            entry.get("attempted_backend_error"),
            entry.get("llm_backend_used"),
        )
    if entry.get("error_message"):
        logger.error(
            "assistant_error request_id=%s error=%s",
            request_id,
            entry.get("error_message"),
        )

    try:
        conn = get_connection()
        try:
            ensure_audit_table(conn)
            conn.execute(
                f"""
                INSERT OR REPLACE INTO audit_log ({", ".join(_AUDIT_LOG_COLUMNS)})
                VALUES ({", ".join(["?"] * len(_AUDIT_LOG_COLUMNS))});
                """,
                (
                    request_id,
                    entry.get("created_at"),
                    entry.get("patient_id"),
                    entry.get("question"),
                    entry.get("status"),
                    entry.get("assistant_answer"),
                    json.dumps(entry.get("sources_used", []), ensure_ascii=False),
                    entry.get("llm_backend_used"),
                    int(bool(entry.get("fallback_used"))),
                    int(bool(entry.get("blocked"))),
                    entry.get("block_reason"),
                    int(bool(entry.get("requires_human_review"))),
                    json.dumps(entry.get("input_validation"), ensure_ascii=False)
                    if entry.get("input_validation") is not None
                    else None,
                    json.dumps(entry.get("output_validation"), ensure_ascii=False)
                    if entry.get("output_validation") is not None
                    else None,
                    entry.get("error_message"),
                    entry.get("system_prompt"),
                    entry.get("user_prompt"),
                    entry.get("attempted_backend"),
                    entry.get("attempted_backend_error"),
                    entry.get("raw_answer"),
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover - defensive, audit must never break the request
        logger.error("audit_log_persist_failed request_id=%s error=%s", request_id, exc)
        # Separate durable destination when the patient database is unavailable.
        path = Path(os.getenv("ASSISTANT_AUDIT_FALLBACK", "backend/data/audit_fallback.jsonl"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())



def get_interaction(request_id: str) -> Optional[Dict[str, Any]]:
    """Recupera uma interação previamente registrada, para auditoria/testes."""
    conn = get_connection()
    try:
        ensure_audit_table(conn)
        row = conn.execute(
            "SELECT * FROM audit_log WHERE request_id = ?;",
            (request_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    result = dict(row)
    for field_name in ("sources_used", "input_validation", "output_validation"):
        if result.get(field_name):
            result[field_name] = json.loads(result[field_name])
    for field_name in ("fallback_used", "blocked", "requires_human_review"):
        result[field_name] = bool(result[field_name])
    return result
