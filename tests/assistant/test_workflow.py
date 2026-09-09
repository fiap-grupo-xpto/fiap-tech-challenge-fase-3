import pytest

from backend.assistant.schemas import AssistantProviderResult, AssistantQueryRequest
from backend.assistant.service import run_assistant_query
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


def test_run_assistant_query_success(assistant_db, monkeypatch):
    from backend.assistant import workflow as workflow_module

    monkeypatch.setattr(
        workflow_module.AssistantProviderSelector,
        "select",
        lambda self, mode: FakeProvider(),
    )

    request = AssistantQueryRequest(
        patient_id="P001",
        question="What should be reviewed next for this patient?",
        include_protocols=True,
        include_pending_exams=True,
    )
    response = run_assistant_query(request)
    assert response.status == "success"
    assert response.patient_id == "P001"
    assert response.llm_backend_used == "fake_provider"
    assert len(response.alerts) > 0
    assert len(response.recommended_actions) > 0


def test_run_assistant_query_patient_not_found(assistant_db, monkeypatch):
    from backend.assistant import workflow as workflow_module

    monkeypatch.setattr(
        workflow_module.AssistantProviderSelector,
        "select",
        lambda self, mode: FakeProvider(),
    )

    request = AssistantQueryRequest(
        patient_id="P999",
        question="What should be reviewed next?",
        include_protocols=True,
        include_pending_exams=True,
    )
    response = run_assistant_query(request)
    assert response.status == "error"
    assert "patient not found" in (response.message or "").lower()

