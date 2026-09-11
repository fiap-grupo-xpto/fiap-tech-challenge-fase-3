from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from backend.assistant.audit_log import log_interaction
from backend.assistant.quality import validate_answer_quality
from backend.assistant.guardrails import (
    check_citation_consistency,
    check_input_safety,
    check_output_safety,
)
from backend.assistant.llm_adapter import AssistantProviderSelector
from backend.assistant.prompts import ASSISTANT_SYSTEM_PROMPT, build_chat_prompt_template, build_user_prompt
from backend.assistant.retrievers import (
    derive_alerts,
    derive_recommended_actions,
    get_patient_context,
    get_pending_exams_reviewed,
    get_protocols_for_context,
    requires_human_validation,
)
from backend.assistant.schemas import (
    AssistantProviderResult,
    AssistantQueryRequest,
    AssistantQueryResponse,
    AssistantWorkflowState,
    GuardrailResult,
)


def _route_after_validation(state: AssistantWorkflowState) -> str:
    if state.get("error_message"):
        return "format_error"
    if state.get("blocked"):
        return "format_blocked"
    return "load_patient_context"


def _route_after_output_validation(state: AssistantWorkflowState) -> str:
    if state.get("error_message"):
        return "format_error"
    if state.get("blocked"):
        return "format_blocked"
    return "format_response"


def validate_input(state: AssistantWorkflowState) -> Dict[str, Any]:
    request = state.get("request")
    if request is None:
        return {"error_message": "Missing request"}

    if not request.patient_id.strip():
        return {"error_message": "patient_id must not be empty"}

    if not request.question.strip():
        return {"error_message": "question must not be empty"}

    guardrail_result = check_input_safety(request.question)
    result: Dict[str, Any] = {"input_validation": guardrail_result}
    if guardrail_result.blocked:
        result["blocked"] = True
        result["block_reason"] = guardrail_result.reason
    return result


def load_patient_context_node(state: AssistantWorkflowState) -> Dict[str, Any]:
    request: AssistantQueryRequest = state["request"]
    patient_context = get_patient_context(request.patient_id)
    if patient_context is None:
        return {"error_message": f"Patient not found: {request.patient_id}"}
    return {"patient_context": patient_context}


def retrieve_protocols_node(state: AssistantWorkflowState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {"protocols": []}
    request: AssistantQueryRequest = state["request"]
    patient_context = state["patient_context"]
    if not request.include_protocols:
        return {"protocols": []}
    protocols = get_protocols_for_context(request.question, patient_context)
    return {"protocols": protocols}


def review_pending_exams_node(state: AssistantWorkflowState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {"pending_exams_reviewed": []}
    request: AssistantQueryRequest = state["request"]
    if not request.include_pending_exams:
        return {"pending_exams_reviewed": []}
    patient_context = state["patient_context"]
    return {"pending_exams_reviewed": get_pending_exams_reviewed(patient_context)}


def evaluate_alerts_node(state: AssistantWorkflowState) -> Dict[str, Any]:
    patient_context = state.get("patient_context")
    if patient_context is None:
        return {"alerts": []}
    pending_exams = state.get("pending_exams_reviewed", [])
    return {"alerts": derive_alerts(patient_context, pending_exams)}


def suggest_actions_node(state: AssistantWorkflowState) -> Dict[str, Any]:
    patient_context = state.get("patient_context")
    if patient_context is None:
        return {"recommended_actions": []}
    pending_exams = state.get("pending_exams_reviewed", [])
    protocols = state.get("protocols", [])
    return {
        "recommended_actions": derive_recommended_actions(patient_context, pending_exams, protocols)
    }


def build_prompt_node(state: AssistantWorkflowState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {}
    request: AssistantQueryRequest = state["request"]
    patient_context = state["patient_context"]
    protocols = state.get("protocols", [])
    pending_exams = state.get("pending_exams_reviewed", [])
    alerts = state.get("alerts", [])
    recommended_actions = state.get("recommended_actions", [])

    user_prompt = build_user_prompt(
        question=request.question,
        patient_context=patient_context,
        protocols=protocols,
        pending_exams_reviewed=pending_exams,
        alerts=alerts,
        recommended_actions=recommended_actions,
    )
    return {"system_prompt": ASSISTANT_SYSTEM_PROMPT, "user_prompt": user_prompt}


def generate_answer_node(state: AssistantWorkflowState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {}
    request: AssistantQueryRequest = state["request"]
    mode = request.force_llm_mode or None
    if mode is None:
        mode = ""
    mode = mode.strip() or ""
    mode = mode or "auto"

    selector = AssistantProviderSelector()
    provider = selector.select(mode if mode else "auto")
    if mode.lower() == "item1_only" and not provider.is_available():
        return {"error_message": "Custom LLM artifacts are not available for item1_only mode"}

    prompt_template = build_chat_prompt_template()
    prompt_value = prompt_template.invoke(
        {"system_prompt": state["system_prompt"], "user_prompt": state["user_prompt"]}
    )
    prompt_text = prompt_value.to_string()

    provider_result: AssistantProviderResult = provider.generate_prompt(
        prompt_text,
        system_prompt=state["system_prompt"],
        user_prompt=state["user_prompt"],
    )
    if not provider_result.answer_text.strip() and not provider_result.provider_error:
        provider_result.provider_error = "Empty model response"
    if provider_result.provider_error:
        # Uma falha real de geração (modelo não carregou, API indisponível, etc.) não
        # pode virar "success" com um texto de fallback genérico — isso mascarava a
        # falha tanto para o cliente quanto para a auditoria.
        return {
            "error_message": (
                f"LLM generation failed ({provider_result.backend_used}): "
                f"{provider_result.provider_error}"
            ),
            "provider_result": provider_result,
        }
    return {"provider_result": provider_result}


def validate_output_node(state: AssistantWorkflowState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {}
    provider_result = state.get("provider_result")
    answer_text = provider_result.answer_text if provider_result else ""

    # Valida o texto da LLM (prescrição/diagnóstico) e também os campos derivados
    # (recommended_actions, alerts) — mesmo sendo determinísticos hoje, não passavam
    # por nenhum controle antes; se um dia passarem a ser influenciados pela LLM, já
    # ficam cobertos pela mesma checagem.
    derived_text = "\n".join([*state.get("recommended_actions", []), *state.get("alerts", [])])

    known_source_ids = [protocol.source_id for protocol in state.get("protocols", [])]

    safety_result = check_output_safety(answer_text)
    derived_result = check_output_safety(derived_text) if derived_text else GuardrailResult(
        blocked=False, requires_human_review=False
    )
    citation_result = check_citation_consistency(answer_text, known_source_ids)

    quality_result = validate_answer_quality(answer_text, state["request"].question,
                                             state["patient_context"], state.get("protocols", []))
    guardrail_result = GuardrailResult(
        blocked=safety_result.blocked or derived_result.blocked or citation_result.blocked or quality_result.blocked,
        requires_human_review=(
            safety_result.requires_human_review
            or derived_result.requires_human_review
            or citation_result.requires_human_review
        ),
        reason=safety_result.reason or derived_result.reason or citation_result.reason or quality_result.reason,
        matched_rules=[
            *safety_result.matched_rules,
            *derived_result.matched_rules,
            *citation_result.matched_rules,
            *quality_result.matched_rules,
        ],
    )

    result: Dict[str, Any] = {"output_validation": guardrail_result}
    if guardrail_result.blocked:
        result["blocked"] = True
        result["block_reason"] = guardrail_result.reason
    return result


def format_response_node(state: AssistantWorkflowState) -> Dict[str, Any]:
    request: AssistantQueryRequest = state["request"]
    patient_context = state.get("patient_context")

    if state.get("error_message"):
        provider_result = state.get("provider_result")
        response = AssistantQueryResponse(
            status="error",
            patient_id=request.patient_id if request else "",
            question=request.question if request else "",
            assistant_answer="",
            patient_context_used={},
            sources_used=[],
            pending_exams_reviewed=[],
            alerts=[],
            recommended_actions=[],
            llm_backend_used=provider_result.backend_used if provider_result else "",
            custom_llm_available=provider_result.custom_llm_available if provider_result else False,
            fallback_used=provider_result.fallback_used if provider_result else False,
            attempted_backend=provider_result.attempted_backend if provider_result else None,
            attempted_backend_error=provider_result.attempted_backend_error if provider_result else None,
            request_id=state.get("request_id"),
            message=state.get("error_message"),
        )
        return {"response": response}

    provider_result = state.get("provider_result")
    protocols = state.get("protocols", [])
    pending_exams = state.get("pending_exams_reviewed", [])

    sources_used = [
        {
            "source_id": patient_context.patient_id,
            "title": "Patient Record",
            "type": "patient_record",
            "snippet": "Structured patient record from SQLite",
        }
    ]
    sources_used.extend([protocol.model_dump() for protocol in protocols])

    response = AssistantQueryResponse(
        status="success",
        patient_id=request.patient_id,
        question=request.question,
        assistant_answer=provider_result.answer_text if provider_result else "",
        patient_context_used=patient_context.model_dump() if patient_context else {},
        sources_used=sources_used,
        sources_cited=[source for source in sources_used if provider_result and
                      re.search(r"(?<![\w-])" + re.escape(source["source_id"]) + r"(?![\w-])",
                                provider_result.answer_text, re.I)],
        pending_exams_reviewed=list(pending_exams),
        alerts=list(state.get("alerts", [])),
        recommended_actions=list(state.get("recommended_actions", [])),
        llm_backend_used=provider_result.backend_used if provider_result else "",
        custom_llm_available=provider_result.custom_llm_available if provider_result else False,
        fallback_used=provider_result.fallback_used if provider_result else False,
        requires_human_review=requires_human_validation(patient_context, protocols) if patient_context else False,
        attempted_backend=provider_result.attempted_backend if provider_result else None,
        attempted_backend_error=provider_result.attempted_backend_error if provider_result else None,
        request_id=state.get("request_id"),
    )
    return {"response": response}


def format_blocked_response_node(state: AssistantWorkflowState) -> Dict[str, Any]:
    request = state.get("request")
    patient_context = state.get("patient_context")
    protocols = state.get("protocols", [])
    provider_result = state.get("provider_result")

    sources_used = []
    if patient_context is not None:
        sources_used.append(
            {
                "source_id": patient_context.patient_id,
                "title": "Patient Record",
                "type": "patient_record",
                "snippet": "Structured patient record from SQLite",
            }
        )
        sources_used.extend([protocol.model_dump() for protocol in protocols])

    block_reason = state.get("block_reason") or "Request blocked by assistant safety guardrails."

    response = AssistantQueryResponse(
        status="blocked",
        patient_id=request.patient_id if request else "",
        question=request.question if request else "",
        assistant_answer="Resposta retida pelas validações de segurança ou de fontes. Encaminhe o caso para revisão clínica.",
        patient_context_used=patient_context.model_dump() if patient_context else {},
        sources_used=sources_used,
        pending_exams_reviewed=list(state.get("pending_exams_reviewed", [])),
        alerts=[],
        recommended_actions=[],
        llm_backend_used=provider_result.backend_used if provider_result else "",
        custom_llm_available=provider_result.custom_llm_available if provider_result else False,
        fallback_used=provider_result.fallback_used if provider_result else False,
        requires_human_review=True,
        blocked=True,
        block_reason=block_reason,
        attempted_backend=provider_result.attempted_backend if provider_result else None,
        attempted_backend_error=provider_result.attempted_backend_error if provider_result else None,
        request_id=state.get("request_id"),
        message=block_reason,
    )
    return {"response": response}


def format_error_node(state: AssistantWorkflowState) -> Dict[str, Any]:
    return format_response_node(state)


def log_interaction_node(state: AssistantWorkflowState) -> Dict[str, Any]:
    request = state.get("request")
    response = state.get("response")
    provider_result = state.get("provider_result")
    input_validation = state.get("input_validation")
    output_validation = state.get("output_validation")

    log_interaction(
        {
            "request_id": state.get("request_id"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "patient_id": request.patient_id if request else None,
            "question": request.question if request else None,
            "status": response.status if response else "error",
            "assistant_answer": response.assistant_answer if response else "",
            "sources_used": response.sources_used if response else [],
            "llm_backend_used": response.llm_backend_used if response else "",
            "fallback_used": response.fallback_used if response else False,
            "blocked": response.blocked if response else bool(state.get("blocked")),
            "block_reason": response.block_reason if response else state.get("block_reason"),
            "requires_human_review": response.requires_human_review if response else False,
            "attempted_backend": provider_result.attempted_backend if provider_result else None,
            "attempted_backend_error": provider_result.attempted_backend_error if provider_result else None,
            "raw_answer": provider_result.answer_text if provider_result else None,
            "input_validation": input_validation.model_dump() if input_validation else None,
            "output_validation": output_validation.model_dump() if output_validation else None,
            "error_message": state.get("error_message"),
            "system_prompt": state.get("system_prompt"),
            "user_prompt": state.get("user_prompt"),
        }
    )
    return {}


def build_assistant_graph():
    graph = StateGraph(AssistantWorkflowState)
    graph.add_node("validate_input", validate_input)
    graph.add_node("load_patient_context", load_patient_context_node)
    graph.add_node("retrieve_protocols", retrieve_protocols_node)
    graph.add_node("review_pending_exams", review_pending_exams_node)
    graph.add_node("evaluate_alerts", evaluate_alerts_node)
    graph.add_node("suggest_actions", suggest_actions_node)
    graph.add_node("build_prompt", build_prompt_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("validate_output", validate_output_node)
    graph.add_node("format_response", format_response_node)
    graph.add_node("format_error", format_error_node)
    graph.add_node("format_blocked", format_blocked_response_node)
    graph.add_node("log_interaction", log_interaction_node)

    graph.set_entry_point("validate_input")
    graph.add_conditional_edges(
        "validate_input",
        _route_after_validation,
        {
            "load_patient_context": "load_patient_context",
            "format_error": "format_error",
            "format_blocked": "format_blocked",
        },
    )
    graph.add_edge("load_patient_context", "retrieve_protocols")
    graph.add_edge("retrieve_protocols", "review_pending_exams")
    graph.add_edge("review_pending_exams", "evaluate_alerts")
    graph.add_edge("evaluate_alerts", "suggest_actions")
    graph.add_edge("suggest_actions", "build_prompt")
    graph.add_edge("build_prompt", "generate_answer")
    graph.add_edge("generate_answer", "validate_output")
    graph.add_conditional_edges(
        "validate_output",
        _route_after_output_validation,
        {
            "format_error": "format_error",
            "format_blocked": "format_blocked",
            "format_response": "format_response",
        },
    )
    graph.add_edge("format_response", "log_interaction")
    graph.add_edge("format_error", "log_interaction")
    graph.add_edge("format_blocked", "log_interaction")
    graph.add_edge("log_interaction", END)

    return graph.compile()
