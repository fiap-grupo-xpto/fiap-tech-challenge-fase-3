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
            answer_text="Resumo: resposta simulada\nContexto do paciente: ok\nConduta sugerida: ok\nJustificativa: ok\nFontes utilizadas: [P001] [P002]\nObservação: ok",
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
    monkeypatch.setenv("ITEM1_PRELOAD", "false")
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


def test_assistant_query_item1_only_uses_gemini_when_item1_artifacts_are_missing(assistant_db, monkeypatch):
    from pathlib import Path
    from backend.assistant import llm_adapter as llm_adapter_module

    missing_model_dir = Path(assistant_db).parent / "missing_item1_model"
    missing_tokenizer_dir = Path(assistant_db).parent / "missing_item1_tokenizer"

    monkeypatch.setenv("ITEM1_MODEL_DIR", str(missing_model_dir))
    monkeypatch.setenv("ITEM1_TOKENIZER_DIR", str(missing_tokenizer_dir))

    def fake_gemini_generate(self, prompt: str, system_prompt=None, user_prompt=None):
        return AssistantProviderResult(
            answer_text="Resumo: fallback Gemini\nContexto do paciente: ok\nConduta sugerida: revisar exames\nJustificativa: contexto estruturado\nFontes utilizadas: [P001]\nObservação: revisão humana.",
            backend_used="gemini_fallback",
            custom_llm_available=False,
            fallback_used=True,
            provider_error=None,
        )

    monkeypatch.setattr(
        llm_adapter_module.GeminiFallbackProvider,
        "generate_prompt",
        fake_gemini_generate,
    )

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
    assert payload["llm_backend_used"] == "gemini_fallback"
    assert payload["fallback_used"] is True
    assert payload["attempted_backend"] == "item1_custom_llm"
    assert "artifacts not available" in payload["attempted_backend_error"]


def test_assistant_query_uses_safe_fallback_when_item1_and_gemini_fail(assistant_db, monkeypatch):
    from pathlib import Path
    from backend.assistant import llm_adapter as llm_adapter_module
    from backend.assistant.audit_log import get_interaction

    monkeypatch.setenv("ITEM1_MODEL_DIR", str(Path(assistant_db).parent / "missing_item1_model"))
    monkeypatch.setenv("ITEM1_TOKENIZER_DIR", str(Path(assistant_db).parent / "missing_item1_tokenizer"))

    def failing_gemini_generate(self, prompt: str, system_prompt=None, user_prompt=None):
        return AssistantProviderResult(
            answer_text="",
            backend_used="gemini_fallback",
            custom_llm_available=False,
            fallback_used=True,
            provider_error="Gemini unavailable for test",
        )

    monkeypatch.setattr(
        llm_adapter_module.GeminiFallbackProvider,
        "generate_prompt",
        failing_gemini_generate,
    )

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
    assert payload["llm_backend_used"] == "gemini_fallback"
    assert payload["fallback_used"] is True
    assert payload["requires_human_review"] is True
    assert "síntese segura" in payload["message"].lower()

    audit_entry = get_interaction(payload["request_id"])
    assert "Gemini unavailable for test" in audit_entry["error_message"]
    assert "artifacts not available" in audit_entry["attempted_backend_error"]


def test_assistant_query_endpoint_blocks_prescription_request(assistant_db, monkeypatch):
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
                "question": "Prescreva um remédio para a tosse do paciente.",
                "include_protocols": True,
                "include_pending_exams": True,
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "blocked"
    assert payload["blocked"] is True
    assert payload["requires_human_review"] is True


def test_assistant_query_uses_safe_fallback_when_model_output_is_unsafe(assistant_db, monkeypatch):
    from backend.assistant import workflow as workflow_module
    from backend.assistant.audit_log import get_interaction

    class UnsafeProvider:
        def is_available(self) -> bool:
            return True

        def generate_prompt(self, prompt: str, system_prompt=None, user_prompt=None):
            return AssistantProviderResult(
                answer_text="O paciente tem câncer de pulmão.",
                backend_used="unsafe_fake_provider",
                custom_llm_available=True,
                fallback_used=False,
                provider_error=None,
            )

    monkeypatch.setattr(
        workflow_module.AssistantProviderSelector,
        "select",
        lambda self, mode: UnsafeProvider(),
    )

    with TestClient(backend_main.app) as client:
        response = client.post(
            "/assistant/query",
            json={
                "patient_id": "P001",
                "question": "Qual exame pendente deve ser revisado?",
                "include_protocols": True,
                "include_pending_exams": True,
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "success"
    assert payload["blocked"] is False
    assert payload["requires_human_review"] is True
    assert "síntese segura" in payload["message"].lower()
    assert "câncer" not in payload["assistant_answer"].lower()
    assert "[P001]" in payload["assistant_answer"]

    audit_entry = get_interaction(payload["request_id"])
    assert audit_entry["raw_answer"] == "O paciente tem câncer de pulmão."
    assert audit_entry["output_validation"]["blocked"] is True


def test_item1_quality_rejection_retries_gemini_and_exposes_rule(assistant_db, monkeypatch):
    from backend.assistant import llm_adapter as llm_adapter_module

    def rejected_item1(self, prompt: str, system_prompt=None, user_prompt=None):
        return AssistantProviderResult(
            answer_text='{"patient_id": "P001"}',
            backend_used="item1_custom_llm",
            custom_llm_available=True,
            fallback_used=False,
        )

    def accepted_gemini(self, prompt: str, system_prompt=None, user_prompt=None):
        return AssistantProviderResult(
            answer_text="Resumo: Revisão do exame pendente [P001].\nContexto do paciente: Registro disponível [P001].\nConduta sugerida: Revisar a Tomografia Computadorizada de Tórax.\nJustificativa: Exame pendente no prontuário [P001].\nFontes utilizadas: [P001].\nObservação: Requer revisão humana.",
            backend_used="gemini_fallback",
            custom_llm_available=False,
            fallback_used=True,
        )

    monkeypatch.setattr(llm_adapter_module.Item1LocalProvider, "generate_prompt", rejected_item1)
    monkeypatch.setattr(llm_adapter_module.GeminiFallbackProvider, "generate_prompt", accepted_gemini)

    with TestClient(backend_main.app) as client:
        response = client.post("/assistant/query", json={
            "patient_id": "P001", "question": "Qual exame pendente deve ser revisado?",
            "force_llm_mode": "item1_only",
        })

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "success"
    assert payload["llm_backend_used"] == "gemini_fallback"
    assert payload["attempted_backend"] == "item1_custom_llm"
    assert "quality:context_echo" in payload["attempted_backend_error"]
    assert "item1:quality:context_echo" in payload["validation_details"]


def test_assistant_query_item1_only_uses_item1_provider_when_artifacts_exist(assistant_db, monkeypatch):
    from backend.assistant import llm_adapter as llm_adapter_module

    def fake_generate_prompt(self, prompt: str, system_prompt=None, user_prompt=None):
        return AssistantProviderResult(
            answer_text="Resumo: resposta item1 simulada\nContexto do paciente: ok\nConduta sugerida: ok\nJustificativa: ok\nFontes utilizadas: [P001] [P002]\nObservação: ok",
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
