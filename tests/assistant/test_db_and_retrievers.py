import os

import pytest

from backend.assistant.retrievers import get_patient_context, get_protocols_for_context
from backend.data.bootstrap_hospital_db import bootstrap


@pytest.fixture()
def assistant_db(tmp_path, monkeypatch):
    db_path = tmp_path / "hospital.db"
    bootstrap(str(db_path))
    monkeypatch.setenv("ASSISTANT_DB_PATH", str(db_path))
    return db_path


def test_get_patient_context_returns_patient(assistant_db):
    patient = get_patient_context("P001")
    assert patient is not None
    assert patient.patient_id == "P001"
    assert patient.sex == "M"
    assert len(patient.symptoms) > 0


def test_protocol_retrieval_varies_by_patient(assistant_db):
    patient_high = get_patient_context("P001")
    patient_low = get_patient_context("P002")
    assert patient_high is not None
    assert patient_low is not None

    protocols_high = get_protocols_for_context(
        "The patient has persistent cough and dyspnea. What should be reviewed next?",
        patient_high,
    )
    protocols_low = get_protocols_for_context(
        "The patient has intermittent cough. What should be reviewed next?",
        patient_low,
    )

    assert len(protocols_high) >= 1
    assert any(protocol.source_id == "PROTO-TRIAGE-001" for protocol in protocols_high)
    assert protocols_high != protocols_low

