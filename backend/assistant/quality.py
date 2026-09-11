"""Conservative, deterministic checks; not a clinical factuality evaluator."""
import re
from collections import Counter

from backend.assistant.guardrails import _normalize
from backend.assistant.schemas import GuardrailResult


def validate_answer_quality(answer, question, patient, protocols):
    text = _normalize(answer)
    rules = []
    ids = [patient.patient_id, *(p.source_id for p in protocols)]
    if not any(re.search(r"(?<![\w-])" + re.escape(s.lower()) + r"(?![\w-])", text) for s in ids):
        rules.append("quality:missing_source")
    lines = [line.strip() for line in text.splitlines() if len(line.strip().split()) >= 4]
    if any(count >= 3 for count in Counter(lines).values()):
        rules.append("quality:repetition")
    if re.search(r'"(?:patient_id|pending_exams|symptoms)"\s*:', text):
        rules.append("quality:context_echo")
    # Reject claims of confirmation by an exam: this assistant cannot validate a diagnosis.
    if re.search(r"\b(?:confirmad[oa]s?\s+(?:por|pelo)|confirmed\s+by)\b", text):
        rules.append("quality:unvalidated_confirmation")
    pending = [e['exam_name'] for e in patient.pending_exams if e.get('status', '').lower() == 'pending']
    asks_pending = re.search(r"\b(?:pending|pendentes?|pendente)\b", _normalize(question))
    if asks_pending and pending and not any(e.lower() in text for e in pending):
        rules.append("quality:pending_exam_not_answered")
    return GuardrailResult(blocked=bool(rules), requires_human_review=True,
                           reason="Resposta sem qualidade ou evidência suficiente para liberação automática." if rules else None,
                           matched_rules=rules)
