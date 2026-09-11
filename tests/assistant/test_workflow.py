import pytest

from backend.assistant.schemas import AssistantProviderResult, AssistantQueryRequest
from backend.assistant.service import run_assistant_query
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


def test_run_assistant_query_flags_human_review_for_high_risk_patient(assistant_db, monkeypatch):
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
    assert response.requires_human_review is True
    assert any(source["type"] == "patient_record" for source in response.sources_used)


def test_run_assistant_query_blocks_prescription_request(assistant_db, monkeypatch):
    from backend.assistant import workflow as workflow_module

    monkeypatch.setattr(
        workflow_module.AssistantProviderSelector,
        "select",
        lambda self, mode: FakeProvider(),
    )

    request = AssistantQueryRequest(
        patient_id="P001",
        question="Prescreva um remédio para a tosse do paciente.",
        include_protocols=True,
        include_pending_exams=True,
    )
    response = run_assistant_query(request)
    assert response.status == "blocked"
    assert response.blocked is True
    assert response.requires_human_review is True
    assert response.patient_context_used == {}


class PrescriptiveProvider:
    def is_available(self) -> bool:
        return True

    def generate_prompt(self, prompt: str, system_prompt=None, user_prompt=None):
        return AssistantProviderResult(
            answer_text="Conduta sugerida: tome 500 mg de amoxicilina a cada 8 horas.",
            backend_used="fake_provider",
            custom_llm_available=True,
            fallback_used=False,
            provider_error=None,
        )


def test_run_assistant_query_blocks_prescriptive_answer(assistant_db, monkeypatch):
    from backend.assistant import workflow as workflow_module

    monkeypatch.setattr(
        workflow_module.AssistantProviderSelector,
        "select",
        lambda self, mode: PrescriptiveProvider(),
    )

    request = AssistantQueryRequest(
        patient_id="P001",
        question="What should be reviewed next for this patient?",
        include_protocols=True,
        include_pending_exams=True,
    )
    response = run_assistant_query(request)
    assert response.status == "blocked"
    assert response.blocked is True
    assert response.requires_human_review is True
    assert "500" not in response.assistant_answer


def test_run_assistant_query_persists_audit_log_entry(assistant_db, monkeypatch):
    from backend.assistant import workflow as workflow_module
    from backend.assistant.audit_log import get_interaction

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
    assert response.request_id is not None

    stored = get_interaction(response.request_id)
    assert stored is not None
    assert stored["status"] == "success"
    assert stored["patient_id"] == "P001"
    assert stored["requires_human_review"] is True
    assert stored["user_prompt"]
    assert request.question in stored["user_prompt"]


def test_run_assistant_query_requires_human_review_even_for_low_risk_patient(assistant_db, monkeypatch):
    """PROTO-HITL-001 exige validação humana para TODA sugestão do assistente, não só
    para pacientes de alto risco — regressão do achado de que P002 (baixo risco) saía
    com requires_human_review=False."""
    from backend.assistant import workflow as workflow_module

    monkeypatch.setattr(
        workflow_module.AssistantProviderSelector,
        "select",
        lambda self, mode: FakeProvider(),
    )

    request = AssistantQueryRequest(
        patient_id="P002",
        question="Paciente relata fadiga leve. O que revisar?",
        include_protocols=True,
        include_pending_exams=True,
    )
    response = run_assistant_query(request)
    assert response.status == "success"
    assert response.requires_human_review is True


def test_run_assistant_query_logs_and_recovers_from_unhandled_node_exception(assistant_db, monkeypatch):
    """Uma exceção não tratada em qualquer nó do grafo (ex.: falha de banco) não pode
    derrubar a auditoria nem propagar como erro 500 sem registro."""
    from backend.assistant import workflow as workflow_module
    from backend.assistant.audit_log import get_interaction

    monkeypatch.setattr(
        workflow_module.AssistantProviderSelector,
        "select",
        lambda self, mode: FakeProvider(),
    )

    def broken_get_patient_context(patient_id):
        raise RuntimeError("simulated unexpected database outage")

    monkeypatch.setattr(workflow_module, "get_patient_context", broken_get_patient_context)

    request = AssistantQueryRequest(
        patient_id="P001",
        question="What should be reviewed next for this patient?",
        include_protocols=True,
        include_pending_exams=True,
    )
    response = run_assistant_query(request)

    assert response.status == "error"
    assert response.request_id is not None

    stored = get_interaction(response.request_id)
    assert stored is not None
    assert stored["status"] == "error"
    assert "simulated unexpected database outage" in (stored["error_message"] or "")


class FailingProvider:
    """Simula o cenário real observado em testes manuais: a pasta de artefatos do
    Item 1 existe (is_available()==True), mas a geração falha internamente."""

    def is_available(self) -> bool:
        return True

    def generate_prompt(self, prompt: str, system_prompt=None, user_prompt=None):
        return AssistantProviderResult(
            answer_text="Não foi possível gerar resposta com o modelo customizado.",
            backend_used="item1_custom_llm",
            custom_llm_available=True,
            fallback_used=False,
            provider_error="simulated model load failure",
        )


def test_run_assistant_query_reports_error_instead_of_fake_success_on_generation_failure(
    assistant_db, monkeypatch
):
    from backend.assistant import workflow as workflow_module

    monkeypatch.setattr(
        workflow_module.AssistantProviderSelector,
        "select",
        lambda self, mode: FailingProvider(),
    )

    request = AssistantQueryRequest(
        patient_id="P001",
        question="What should be reviewed next for this patient?",
        include_protocols=True,
        include_pending_exams=True,
    )
    response = run_assistant_query(request)
    assert response.status == "error"
    assert "simulated model load failure" in (response.message or "")
    assert response.llm_backend_used == "item1_custom_llm"


def test_run_assistant_query_preserves_primary_failure_reason_through_fallback(
    assistant_db, monkeypatch
):
    """Quando o modelo local falha mas o Gemini responde, o motivo da falha local não
    pode sumir da resposta nem do log de auditoria — só o "success" final não basta
    para reconstruir o que realmente aconteceu."""
    from backend.assistant import workflow as workflow_module
    from backend.assistant.audit_log import get_interaction
    from backend.assistant.llm_adapter import AutoFallbackProvider

    class FailingLocalProvider:
        def is_available(self) -> bool:
            return True

        def generate_prompt(self, prompt: str, system_prompt=None, user_prompt=None):
            return AssistantProviderResult(
                answer_text="Não foi possível gerar resposta com o modelo customizado.",
                backend_used="item1_custom_llm",
                custom_llm_available=True,
                fallback_used=False,
                provider_error="simulated local model load failure",
            )

    class WorkingFallbackProvider:
        def is_available(self) -> bool:
            return True

        def generate_prompt(self, prompt: str, system_prompt=None, user_prompt=None):
            return AssistantProviderResult(
                answer_text="Resumo: resposta via fallback\nFontes utilizadas: [P001] [P002]",
                backend_used="gemini_fallback",
                custom_llm_available=False,
                fallback_used=True,
                provider_error=None,
            )

    monkeypatch.setattr(
        workflow_module.AssistantProviderSelector,
        "select",
        lambda self, mode: AutoFallbackProvider(FailingLocalProvider(), WorkingFallbackProvider()),
    )

    request = AssistantQueryRequest(
        patient_id="P001",
        question="What should be reviewed next for this patient?",
        include_protocols=True,
        include_pending_exams=True,
    )
    response = run_assistant_query(request)

    assert response.status == "success"
    assert response.llm_backend_used == "gemini_fallback"
    assert response.attempted_backend == "item1_custom_llm"
    assert response.attempted_backend_error == "simulated local model load failure"

    stored = get_interaction(response.request_id)
    assert stored["attempted_backend"] == "item1_custom_llm"
    assert stored["attempted_backend_error"] == "simulated local model load failure"


class HallucinatedCitationProvider:
    def is_available(self) -> bool:
        return True

    def generate_prompt(self, prompt: str, system_prompt=None, user_prompt=None):
        return AssistantProviderResult(
            answer_text=(
                "Resumo: revisar exames pendentes\n"
                "Fontes utilizadas: PROTO-DOES-NOT-EXIST-999\n"
                "Observação: apoio à decisão clínica"
            ),
            backend_used="fake_provider",
            custom_llm_available=True,
            fallback_used=False,
            provider_error=None,
        )


def test_run_assistant_query_flags_uncited_protocol_reference_without_blocking(
    assistant_db, monkeypatch
):
    from backend.assistant import workflow as workflow_module

    monkeypatch.setattr(
        workflow_module.AssistantProviderSelector,
        "select",
        lambda self, mode: HallucinatedCitationProvider(),
    )

    request = AssistantQueryRequest(
        patient_id="P001",
        question="What should be reviewed next for this patient?",
        include_protocols=True,
        include_pending_exams=True,
    )
    response = run_assistant_query(request)

    assert response.status == "blocked"
    assert response.blocked is True
    assert response.requires_human_review is True



def test_blocked_derived_fields_do_not_escape(assistant_db, monkeypatch):
    from backend.assistant import workflow as w
    from backend.assistant.audit_log import get_interaction
    monkeypatch.setattr(w.AssistantProviderSelector, "select", lambda self, mode: FakeProvider())
    monkeypatch.setattr(w, "derive_recommended_actions", lambda *args: ["Use amoxicilina por via oral."])
    result = run_assistant_query(AssistantQueryRequest(patient_id="P001", question="Revisar exames?"))
    assert result.status == "blocked"
    assert result.recommended_actions == []
    assert result.alerts == []
    assert get_interaction(result.request_id)["raw_answer"]


@pytest.mark.parametrize("failed,answer", [(True, ""), (False, "Tome dois comprimidos por dia.")])
def test_fallback_error_survives_non_success(assistant_db, monkeypatch, failed, answer):
    from backend.assistant import workflow as w
    from backend.assistant.audit_log import get_interaction
    class Provider:
        def generate_prompt(self, *args, **kwargs):
            return AssistantProviderResult(answer_text=answer, backend_used="fallback", custom_llm_available=False,
                fallback_used=True, provider_error="fallback failed" if failed else None,
                attempted_backend="local", attempted_backend_error="local failed")
    monkeypatch.setattr(w.AssistantProviderSelector, "select", lambda self, mode: Provider())
    response = run_assistant_query(AssistantQueryRequest(patient_id="P001", question="Revisar exames?"))
    assert response.status == ("error" if failed else "blocked")
    assert response.attempted_backend_error == "local failed"
    assert get_interaction(response.request_id)["attempted_backend_error"] == "local failed"


def test_missing_sources_blocked_and_audited(assistant_db, monkeypatch):
    from backend.assistant import workflow as w
    from backend.assistant.audit_log import get_interaction
    class Uncited:
        def generate_prompt(self, *args, **kwargs):
            return AssistantProviderResult(answer_text="Revisar chest_ct pendente.", backend_used="fake",
                custom_llm_available=False, fallback_used=False)
    monkeypatch.setattr(w.AssistantProviderSelector, "select", lambda self, mode: Uncited())
    result = run_assistant_query(AssistantQueryRequest(patient_id="P001", question="Qual exame está pendente?"))
    assert result.status == "blocked"
    assert result.recommended_actions == []
    stored = get_interaction(result.request_id)
    assert "quality:missing_source" in stored["output_validation"]["matched_rules"]
    assert stored["raw_answer"] == "Revisar chest_ct pendente."
