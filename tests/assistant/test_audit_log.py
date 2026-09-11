import pytest

from backend.assistant.audit_log import get_interaction, log_interaction
from backend.data.bootstrap_hospital_db import bootstrap


@pytest.fixture()
def assistant_db(tmp_path, monkeypatch):
    db_path = tmp_path / "hospital.db"
    bootstrap(str(db_path))
    monkeypatch.setenv("ASSISTANT_DB_PATH", str(db_path))
    return db_path


def test_log_interaction_persists_and_can_be_read_back(assistant_db):
    entry = {
        "request_id": "test-request-1",
        "created_at": "2026-09-10T10:00:00+00:00",
        "patient_id": "P001",
        "question": "What should be reviewed next?",
        "status": "success",
        "assistant_answer": "Resumo: ok",
        "sources_used": [
            {
                "source_id": "P001",
                "title": "Patient Record",
                "type": "patient_record",
                "snippet": "...",
            }
        ],
        "llm_backend_used": "fake_provider",
        "fallback_used": False,
        "blocked": False,
        "block_reason": None,
        "requires_human_review": True,
        "input_validation": {
            "blocked": False,
            "requires_human_review": False,
            "reason": None,
            "matched_rules": [],
        },
        "output_validation": {
            "blocked": False,
            "requires_human_review": False,
            "reason": None,
            "matched_rules": [],
        },
        "error_message": None,
    }

    log_interaction(entry)

    stored = get_interaction("test-request-1")
    assert stored is not None
    assert stored["patient_id"] == "P001"
    assert stored["status"] == "success"
    assert stored["requires_human_review"] is True
    assert stored["sources_used"][0]["source_id"] == "P001"


def test_log_interaction_records_blocked_reason(assistant_db):
    entry = {
        "request_id": "test-request-2",
        "created_at": "2026-09-10T10:05:00+00:00",
        "patient_id": "P001",
        "question": "Prescreva um remedio",
        "status": "blocked",
        "assistant_answer": "...",
        "sources_used": [],
        "llm_backend_used": "",
        "fallback_used": False,
        "blocked": True,
        "block_reason": "input guardrail triggered",
        "requires_human_review": True,
        "input_validation": None,
        "output_validation": None,
        "error_message": None,
    }

    log_interaction(entry)

    stored = get_interaction("test-request-2")
    assert stored is not None
    assert stored["blocked"] is True
    assert stored["block_reason"] == "input guardrail triggered"


def test_get_interaction_returns_none_for_unknown_request_id(assistant_db):
    assert get_interaction("does-not-exist") is None


def test_database_failure_uses_durable_file(tmp_path, monkeypatch):
    import json
    from backend.assistant import audit_log
    destination = tmp_path / "fallback.jsonl"
    monkeypatch.setenv("ASSISTANT_AUDIT_FALLBACK", str(destination))
    def fail():
        raise OSError("database offline")
    monkeypatch.setattr(audit_log, "get_connection", fail)
    audit_log.log_interaction({"request_id": "offline", "status": "blocked", "raw_answer": "original"})
    assert json.loads(destination.read_text(encoding="utf-8"))["raw_answer"] == "original"


def test_both_audit_destinations_fail_explicitly(tmp_path, monkeypatch):
    import pytest
    from backend.assistant import audit_log
    monkeypatch.setenv("ASSISTANT_AUDIT_FALLBACK", str(tmp_path))
    def fail():
        raise OSError("database offline")
    monkeypatch.setattr(audit_log, "get_connection", fail)
    with pytest.raises(OSError):
        audit_log.log_interaction({"request_id": "offline"})
