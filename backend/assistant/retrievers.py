import re
from typing import List, Optional, Sequence

from backend.assistant.db import get_connection
from backend.assistant.schemas import PatientContext, SourceReference


def _normalize_tokens(text: str) -> List[str]:
    cleaned = re.sub(r"[^0-9A-Za-z]+", " ", (text or "").lower()).strip()
    if not cleaned:
        return []
    return [token for token in cleaned.split() if token]


def get_patient_context(patient_id: str) -> Optional[PatientContext]:
    with get_connection() as conn:
        patient_row = conn.execute(
            "SELECT patient_id, name_synthetic, age, sex, smoking_history, last_updated_at "
            "FROM patients WHERE patient_id = ?;",
            (patient_id,),
        ).fetchone()

        if patient_row is None:
            return None

        symptoms = conn.execute(
            "SELECT symptom, severity FROM symptoms WHERE patient_id = ?;",
            (patient_id,),
        ).fetchall()
        pending_exams = conn.execute(
            "SELECT exam_name, status, requested_at FROM pending_exams WHERE patient_id = ?;",
            (patient_id,),
        ).fetchall()
        latest_findings = conn.execute(
            "SELECT finding, risk_level, created_at FROM latest_findings WHERE patient_id = ? "
            "ORDER BY created_at DESC;",
            (patient_id,),
        ).fetchall()
        recent_notes = conn.execute(
            "SELECT note_type, content, created_at FROM clinical_notes WHERE patient_id = ? "
            "ORDER BY created_at DESC;",
            (patient_id,),
        ).fetchall()

    return PatientContext(
        patient_id=patient_row["patient_id"],
        name_synthetic=patient_row["name_synthetic"],
        age=int(patient_row["age"]),
        sex=patient_row["sex"],
        smoking_history=patient_row["smoking_history"],
        last_updated_at=patient_row["last_updated_at"],
        symptoms=[{"symptom": row["symptom"], "severity": row["severity"]} for row in symptoms],
        pending_exams=[
            {
                "exam_name": row["exam_name"],
                "status": row["status"],
                "requested_at": row["requested_at"],
            }
            for row in pending_exams
        ],
        latest_findings=[
            {
                "finding": row["finding"],
                "risk_level": row["risk_level"],
                "created_at": row["created_at"],
            }
            for row in latest_findings
        ],
        recent_notes=[
            {
                "note_type": row["note_type"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in recent_notes
        ],
    )


def get_pending_exams_reviewed(patient_context: PatientContext) -> List[str]:
    return [
        exam["exam_name"]
        for exam in patient_context.pending_exams
        if exam.get("status", "").lower() == "pending"
    ]


def get_protocols_for_context(question: str, patient_context: PatientContext) -> List[SourceReference]:
    question_tokens = set(_normalize_tokens(question))
    symptom_tokens = set(
        token
        for entry in patient_context.symptoms
        for token in _normalize_tokens(entry.get("symptom", ""))
    )
    exam_tokens = set(
        token
        for entry in patient_context.pending_exams
        for token in _normalize_tokens(f"{entry.get('exam_name', '')} {entry.get('status', '')}")
    )
    finding_tokens = set(
        token
        for entry in patient_context.latest_findings
        for token in _normalize_tokens(entry.get("finding", ""))
    )
    smoking_tokens = set(_normalize_tokens(patient_context.smoking_history))

    context_tokens = question_tokens | symptom_tokens | exam_tokens | finding_tokens | smoking_tokens

    with get_connection() as conn:
        protocol_rows = conn.execute(
            "SELECT protocol_id, title, category, keywords, content FROM protocols;"
        ).fetchall()

    scored = []
    for row in protocol_rows:
        keywords = set(_normalize_tokens(row["keywords"]))
        overlap = len(context_tokens.intersection(keywords))
        if overlap <= 0:
            continue
        scored.append((overlap, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [row for _, row in scored[:5]]

    results: List[SourceReference] = []
    for row in selected:
        content = row["content"]
        snippet = content if len(content) <= 220 else content[:217].rstrip() + "..."
        results.append(
            SourceReference(
                source_id=row["protocol_id"],
                title=row["title"],
                type="protocol",
                snippet=snippet,
            )
        )
    return results


def derive_alerts(patient_context: PatientContext, pending_exams: Sequence[str]) -> List[str]:
    alerts: List[str] = []

    risk_levels = [entry.get("risk_level", "").lower() for entry in patient_context.latest_findings]
    if any(level == "alto" for level in risk_levels):
        alerts.append("High-risk indicators present in latest findings")

    symptom_list = [entry.get("symptom", "").lower() for entry in patient_context.symptoms]
    if any("hemoptysis" in symptom for symptom in symptom_list):
        alerts.append("Hemoptysis reported; consider urgent clinician review")

    if pending_exams:
        alerts.append(f"Pending exams: {', '.join(pending_exams)}")

    if patient_context.smoking_history.lower().startswith("active"):
        alerts.append("Active smoker; increased risk profile")

    return alerts


def derive_recommended_actions(
    patient_context: PatientContext,
    pending_exams: Sequence[str],
    protocols: Sequence[SourceReference],
) -> List[str]:
    actions: List[str] = []

    if pending_exams:
        actions.append("Review pending exams and expected completion status")

    risk_levels = [entry.get("risk_level", "").lower() for entry in patient_context.latest_findings]
    if any(level == "alto" for level in risk_levels):
        actions.append("Escalate for clinician validation and specialized assessment")
    else:
        actions.append("Monitor symptoms and reassess based on clinical evolution")

    protocol_ids = {protocol.source_id for protocol in protocols}
    if "PROTO-NO-RX-001" in protocol_ids:
        actions.append("Avoid direct prescription; recommend clinician review of treatment options")

    if "PROTO-HITL-001" in protocol_ids:
        actions.append("Ensure all recommendations are validated by a clinician")

    seen = set()
    deduped = []
    for action in actions:
        normalized = action.strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(action)
    return deduped
