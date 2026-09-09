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


class AssistantQueryResponse(BaseModel):
    status: str
    patient_id: str
    question: str
    assistant_answer: str
    patient_context_used: Dict[str, Any]
    sources_used: List[Dict[str, Any]]
    pending_exams_reviewed: List[str]
    alerts: List[str]
    recommended_actions: List[str]
    llm_backend_used: str
    custom_llm_available: bool
    fallback_used: bool
    message: Optional[str] = None


class AssistantWorkflowState(TypedDict, total=False):
    request: AssistantQueryRequest
    patient_context: PatientContext
    protocols: List[SourceReference]
    pending_exams_reviewed: List[str]
    alerts: List[str]
    recommended_actions: List[str]
    system_prompt: str
    user_prompt: str
    provider_result: AssistantProviderResult
    response: AssistantQueryResponse
    error_message: str

