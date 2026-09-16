from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class AssistantQueryRequest(BaseModel):
    patient_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    include_protocols: bool = True
    include_pending_exams: bool = True
    force_llm_mode: Optional[str] = None


class PatientContext(BaseModel):
    patient_id: str
    name_synthetic: str
    age: int
    sex: str
    smoking_history: str
    last_updated_at: str
    symptoms: List[Dict[str, str]] = Field(default_factory=list)
    pending_exams: List[Dict[str, str]] = Field(default_factory=list)
    latest_findings: List[Dict[str, str]] = Field(default_factory=list)
    recent_notes: List[Dict[str, str]] = Field(default_factory=list)


class SourceReference(BaseModel):
    source_id: str
    title: str
    type: str
    snippet: str


class AssistantProviderResult(BaseModel):
    answer_text: str
    backend_used: str
    custom_llm_available: bool
    fallback_used: bool
    provider_error: Optional[str] = None
    # Preenchidos por AutoFallbackProvider quando a tentativa PRIMÁRIA falha mas o
    # fallback tem sucesso — sem isso, um "success" via Gemini apagava completamente o
    # fato de que o modelo local foi tentado primeiro e por que falhou.
    attempted_backend: Optional[str] = None
    attempted_backend_error: Optional[str] = None


class GuardrailResult(BaseModel):
    blocked: bool
    requires_human_review: bool
    reason: Optional[str] = None
    matched_rules: List[str] = Field(default_factory=list)


class AssistantQueryResponse(BaseModel):
    status: str
    patient_id: str
    question: str
    assistant_answer: str
    patient_context_used: Dict[str, Any]
    sources_used: List[Dict[str, Any]]
    sources_cited: List[Dict[str, Any]] = Field(default_factory=list)
    pending_exams_reviewed: List[str]
    alerts: List[str]
    recommended_actions: List[str]
    llm_backend_used: str
    custom_llm_available: bool
    fallback_used: bool
    requires_human_review: bool = False
    blocked: bool = False
    block_reason: Optional[str] = None
    attempted_backend: Optional[str] = None
    attempted_backend_error: Optional[str] = None
    request_id: Optional[str] = None
    message: Optional[str] = None


class AssistantWorkflowState(TypedDict, total=False):
    request: AssistantQueryRequest
    request_id: str
    patient_context: PatientContext
    protocols: List[SourceReference]
    pending_exams_reviewed: List[str]
    alerts: List[str]
    recommended_actions: List[str]
    system_prompt: str
    user_prompt: str
    provider_result: AssistantProviderResult
    input_validation: GuardrailResult
    output_validation: GuardrailResult
    blocked: bool
    block_reason: str
    response: AssistantQueryResponse
    error_message: str
    provider_failure_message: str
