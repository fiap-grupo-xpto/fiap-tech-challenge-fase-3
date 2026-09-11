"""Synthetic operational examples using the same prompt builder as the API.

No clinical protocols or treatment doses are invented here. Variants of a patient
share a qid so they cannot leak across train and evaluation splits.
"""
from backend.assistant.prompts import build_user_prompt
from backend.assistant.schemas import PatientContext


def build_context_examples():
    rows = []
    for i in range(30):
        patient_id = f"SYN-{i:03d}"
        exam = ["chest_ct", "spirometry", "chest_xray"][i % 3]
        patient = PatientContext(
            patient_id=patient_id, name_synthetic=f"Synthetic {i}", age=30+i,
            sex="F" if i % 2 else "M", smoking_history="unknown",
            last_updated_at="2026-09-01T00:00:00",
            pending_exams=[{"exam_name": exam, "status": "pending", "requested_at": "2026-09-01"}],
        )
        for question in ["Qual exame está pendente? Indique a fonte.",
                         "Which pending exam should the clinician review? Cite its source."]:
            prompt = build_user_prompt(question, patient, [], [exam], [], [])
            answer = (f"Resumo: O exame {exam} está pendente [{patient_id}].\n"
                      f"Contexto do paciente: Não há resultado desse exame no contexto fornecido.\n"
                      f"Conduta sugerida: O clínico deve revisar o status do exame {exam}.\n"
                      f"Justificativa: O registro informa status pending [{patient_id}].\n"
                      f"Fontes utilizadas: [{patient_id}].\n"
                      "Observação: Requer validação humana; não permite concluir diagnóstico.")
            rows.append({"qid": patient_id, "question_type": "synthetic_context",
                         "question_anon": prompt, "answer_anon": answer})
    return rows
