import uuid
from datetime import datetime, timezone

from backend.assistant.audit_log import log_interaction
from backend.assistant.schemas import AssistantQueryRequest, AssistantQueryResponse, AssistantWorkflowState
from backend.assistant.workflow import build_assistant_graph

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_assistant_graph()
    return _graph


def run_assistant_query(request: AssistantQueryRequest) -> AssistantQueryResponse:
    request_id = str(uuid.uuid4())
    initial_state: AssistantWorkflowState = {"request": request, "request_id": request_id}

    try:
        graph = _get_graph()
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        # Um nó do grafo pode falhar de forma inesperada (ex.: banco indisponível) sem
        # nunca chegar ao nó log_interaction — sem este catch, a interação sumiria da
        # auditoria exatamente no cenário em que ela mais importa.
        log_interaction(
            {
                "request_id": request_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "patient_id": request.patient_id,
                "question": request.question,
                "status": "error",
                "assistant_answer": "",
                "sources_used": [],
                "llm_backend_used": "",
                "fallback_used": False,
                "blocked": False,
                "block_reason": None,
                "requires_human_review": False,
                "input_validation": None,
                "output_validation": None,
                "error_message": f"Unhandled exception in assistant workflow: {exc}",
                "system_prompt": None,
                "user_prompt": None,
            }
        )
        return AssistantQueryResponse(
            status="error",
            patient_id=request.patient_id,
            question=request.question,
            assistant_answer="",
            patient_context_used={},
            sources_used=[],
            pending_exams_reviewed=[],
            alerts=[],
            recommended_actions=[],
            llm_backend_used="",
            custom_llm_available=False,
            fallback_used=False,
            request_id=request_id,
            message="An internal error occurred while processing the assistant request.",
        )

    response = final_state.get("response")
    if response is None:
        return AssistantQueryResponse(
            status="error",
            patient_id=request.patient_id,
            question=request.question,
            assistant_answer="",
            patient_context_used={},
            sources_used=[],
            pending_exams_reviewed=[],
            alerts=[],
            recommended_actions=[],
            llm_backend_used="",
            custom_llm_available=False,
            fallback_used=False,
            request_id=request_id,
            message="Assistant workflow did not produce a response.",
        )
    return response

