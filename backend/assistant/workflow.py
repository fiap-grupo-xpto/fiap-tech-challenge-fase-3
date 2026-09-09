from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from backend.assistant.llm_adapter import AssistantProviderSelector
from backend.assistant.prompts import ASSISTANT_SYSTEM_PROMPT, build_chat_prompt_template, build_user_prompt
from backend.assistant.retrievers import (
    derive_alerts,
    derive_recommended_actions,
    get_patient_context,
    get_pending_exams_reviewed,
    get_protocols_for_context,
)
from backend.assistant.schemas import AssistantProviderResult, AssistantQueryRequest, AssistantQueryResponse, AssistantWorkflowState


def _route_after_validation(state: AssistantWorkflowState) -> str:
    if state.get("error_message"):
        return "format_error"
    return "load_patient_context"


def validate_input(state: AssistantWorkflowState) -> Dict[str, Any]:
    request = state.get("request")
    if request is None:
        return {"error_message": "Missing request"}

    if not request.patient_id.strip():
        return {"error_message": "patient_id must not be empty"}

    if not request.question.strip():
        return {"error_message": "question must not be empty"}

    return {}


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

    provider_result: AssistantProviderResult = provider.generate_prompt(prompt_text)
    return {"provider_result": provider_result}


def format_response_node(state: AssistantWorkflowState) -> Dict[str, Any]:
    request: AssistantQueryRequest = state["request"]
    patient_context = state.get("patient_context")

    if state.get("error_message"):
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
            llm_backend_used="",
            custom_llm_available=False,
            fallback_used=False,
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
        pending_exams_reviewed=list(pending_exams),
        alerts=list(state.get("alerts", [])),
        recommended_actions=list(state.get("recommended_actions", [])),
        llm_backend_used=provider_result.backend_used if provider_result else "",
        custom_llm_available=provider_result.custom_llm_available if provider_result else False,
        fallback_used=provider_result.fallback_used if provider_result else False,
    )
    return {"response": response}


def format_error_node(state: AssistantWorkflowState) -> Dict[str, Any]:
    return format_response_node(state)


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
    graph.add_node("format_response", format_response_node)
    graph.add_node("format_error", format_error_node)

    graph.set_entry_point("validate_input")
    graph.add_conditional_edges(
        "validate_input",
        _route_after_validation,
        {"load_patient_context": "load_patient_context", "format_error": "format_error"},
    )
    graph.add_edge("load_patient_context", "retrieve_protocols")
    graph.add_edge("retrieve_protocols", "review_pending_exams")
    graph.add_edge("review_pending_exams", "evaluate_alerts")
    graph.add_edge("evaluate_alerts", "suggest_actions")
    graph.add_edge("suggest_actions", "build_prompt")
    graph.add_edge("build_prompt", "generate_answer")
    graph.add_edge("generate_answer", "format_response")
    graph.add_edge("format_response", END)
    graph.add_edge("format_error", END)

    return graph.compile()
