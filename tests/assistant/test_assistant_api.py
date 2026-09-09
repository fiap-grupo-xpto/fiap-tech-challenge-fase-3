from fastapi.testclient import TestClient

import pytest

import backend.main as backend_main
from backend.assistant.schemas import AssistantProviderResult
from backend.data.bootstrap_hospital_db import bootstrap


class FakeProvider:
    def is_available(self) -> bool:
        return True

    def generate_prompt(self, prompt: str, system_prompt=None, user_prompt=None):
        return AssistantProviderResult(
            answer_text="Resumo: resposta simulada\nContexto do paciente: ok\nConduta sugerida: ok\nJustificativa: ok\nFontes utilizadas: ok\nObservação: ok",
            backend_used="fake_provider",
            custom_llm_available=True,
            fallback_used=False,
            provider_error=None,
        )


@pytest.fixture()
def assistant_db(tmp_path, monkeypatch):
    db_path = tmp_path / "hospital.db"
    bootstrap(str(db_path))
    monkeypatch.setenv("ASSISTANT_DB_PATH", str(db_path))
    return db_path


def test_assistant_query_endpoint_success(assistant_db, monkeypatch):
    from backend.assistant import workflow as workflow_module

    monkeypatch.setattr(
        workflow_module.AssistantProviderSelector,
        "select",
        lambda self, mode: FakeProvider(),
    )

    with TestClient(backend_main.app) as client:
        response = client.post(
            "/assistant/query",
            json={
                "patient_id": "P001",
                "question": "What should be reviewed next?",
                "include_protocols": True,
                "include_pending_exams": True,
                "force_llm_mode": "auto",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "success"
    assert payload["llm_backend_used"] == "fake_provider"
    assert payload["patient_id"] == "P001"
    assert len(payload["sources_used"]) >= 1


def test_assistant_query_item1_only_fails_when_artifacts_missing(assistant_db, monkeypatch):
    from pathlib import Path

    missing_model_dir = Path(assistant_db).parent / "missing_item1_model"
    missing_tokenizer_dir = Path(assistant_db).parent / "missing_item1_tokenizer"

    monkeypatch.setenv("ITEM1_MODEL_DIR", str(missing_model_dir))
    monkeypatch.setenv("ITEM1_TOKENIZER_DIR", str(missing_tokenizer_dir))

    with TestClient(backend_main.app) as client:
        response = client.post(
            "/assistant/query",
            json={
                "patient_id": "P001",
                "question": "What should be reviewed next?",
                "include_protocols": True,
                "include_pending_exams": True,
                "force_llm_mode": "item1_only",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "error"
    assert "custom llm" in (payload.get("message", "") or "").lower()


def test_assistant_query_item1_only_uses_item1_provider_when_artifacts_exist(assistant_db, monkeypatch):
    from backend.assistant import llm_adapter as llm_adapter_module

    def fake_generate_prompt(self, prompt: str, system_prompt=None, user_prompt=None):
        return AssistantProviderResult(
            answer_text="Resumo: resposta item1 simulada\nContexto do paciente: ok\nConduta sugerida: ok\nJustificativa: ok\nFontes utilizadas: ok\nObservação: ok",
            backend_used="item1_custom_llm",
            custom_llm_available=True,
            fallback_used=False,
            provider_error=None,
        )

    monkeypatch.setattr(llm_adapter_module.Item1LocalProvider, "generate_prompt", fake_generate_prompt)

    with TestClient(backend_main.app) as client:
        response = client.post(
            "/assistant/query",
            json={
                "patient_id": "P001",
                "question": "What should be reviewed next?",
                "include_protocols": True,
                "include_pending_exams": True,
                "force_llm_mode": "item1_only",
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "success"
    assert payload["llm_backend_used"] == "item1_custom_llm"
    assert payload["custom_llm_available"] is True
    assert payload["fallback_used"] is False
