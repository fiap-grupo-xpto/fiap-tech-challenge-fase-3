from backend.assistant.schemas import AssistantQueryRequest, AssistantQueryResponse, AssistantWorkflowState
from backend.assistant.workflow import build_assistant_graph

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_assistant_graph()
    return _graph


def run_assistant_query(request: AssistantQueryRequest) -> AssistantQueryResponse:
    graph = _get_graph()
    initial_state: AssistantWorkflowState = {"request": request}
    final_state = graph.invoke(initial_state)
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
            message="Assistant workflow did not produce a response.",
        )
    return response

