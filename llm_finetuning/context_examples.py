"""Synthetic operational examples using the same prompt builder as the API.

Covers:
1. Operational context & pending exams review
2. Internal report models (laudos estruturados com achados preliminares)
3. Prescription & medication reconciliation models (receitas seguras, sem prescrição autônoma)
4. Internal hospital procedures (procedimentos e preparos conforme protocolos)

No treatment doses are invented autonomously. Variants of a patient share a qid
so they cannot leak across train and evaluation splits.
"""
import json
from pathlib import Path

try:
    from backend.assistant.prompts import build_user_prompt
    from backend.assistant.schemas import PatientContext, SourceReference
    _SCHEMAS_AVAILABLE = True
except ImportError:
    _SCHEMAS_AVAILABLE = False


def build_context_examples():
    jsonl_path = Path(__file__).parent / "data" / "context_examples.jsonl"
    if not _SCHEMAS_AVAILABLE and jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    rows = []

    # 1. Operational pending exams (30 patients x 2 queries = 60 rows)
    for i in range(30):
        patient_id = f"SYN-{i:03d}"
        exam = ["chest_ct", "spirometry", "chest_xray"][i % 3]
        patient = PatientContext(
            patient_id=patient_id,
            name_synthetic=f"Synthetic {i}",
            age=30 + i,
            sex="F" if i % 2 else "M",
            smoking_history="unknown",
            last_updated_at="2026-09-01T00:00:00",
            pending_exams=[{"exam_name": exam, "status": "pending", "requested_at": "2026-09-01"}],
        )
        for question in [
            "Qual exame está pendente? Indique a fonte.",
            "Which pending exam should the clinician review? Cite its source.",
        ]:
            prompt = build_user_prompt(question, patient, [], [exam], [], [])
            answer = (
                f"Resumo: O exame {exam} está pendente [{patient_id}].\n"
                f"Contexto do paciente: Não há resultado desse exame no contexto fornecido.\n"
                f"Conduta sugerida: O clínico deve revisar o status do exame {exam}.\n"
                f"Justificativa: O registro informa status pending [{patient_id}].\n"
                f"Fontes utilizadas: [{patient_id}].\n"
                "Observação: Requer validação humana; não permite concluir diagnóstico."
            )
            rows.append(
                {
                    "qid": patient_id,
                    "question_type": "synthetic_context",
                    "question_anon": prompt,
                    "answer_anon": answer,
                }
            )

    # 2. Internal report models / Laudos estruturados (15 patients x 2 queries = 30 rows)
    for i in range(15):
        patient_id = f"SYN-REP-{i:03d}"
        finding = ["nódulo pulmonar em lobo superior direito de 6mm", "espessamento pleural bilateral", "infiltrado intersticial difuso"][i % 3]
        exam_type = ["Tomografia de Tórax", "Radiografia de Tórax", "Tomografia Computadorizada de Alta Resolução"][i % 3]
        patient = PatientContext(
            patient_id=patient_id,
            name_synthetic=f"Synthetic Report {i}",
            age=45 + i,
            sex="M" if i % 2 else "F",
            smoking_history="active smoker" if i % 2 == 0 else "former smoker",
            last_updated_at="2026-09-05T00:00:00",
            latest_findings=[{"finding": f"{exam_type}: {finding}", "risk_level": "moderate", "created_at": "2026-09-05"}],
            recent_notes=[{"note_type": "laudo_preliminar", "content": f"Achado radiológico: {finding}. Correlação clínica recomendada.", "created_at": "2026-09-05"}],
        )
        protocol = SourceReference(
            source_id="PROTO-LUNG-001",
            title="Manejo de Achados Pulmonares Incidentais",
            type="protocol",
            snippet="Nódulos indeterminados requerem acompanhamento tomográfico seriado e correlação clínica sem diagnóstico imediato.",
        )
        for question in [
            "Qual o modelo de estruturação de laudo e conduta recomendada para este achado?",
            "Como estruturar o resumo clínico deste achado de imagem com base no protocolo?",
        ]:
            prompt = build_user_prompt(question, patient, [protocol], [], ["Achado incidental identificado."], ["Revisar imagem com especialista."])
            answer = (
                f"Resumo: Laudo preliminar indica achado de {finding} em {exam_type} [{patient_id}].\n"
                f"Contexto do paciente: Paciente {patient.age} anos, histórico {patient.smoking_history} [{patient_id}].\n"
                f"Conduta sugerida: Adotar modelo de laudo descritivo estruturado e seguir protocolo [PROTO-LUNG-001] para acompanhamento seriado.\n"
                f"Justificativa: Achados indeterminados necessitam de vigilância radiológica comparativa antes de condutas invasivas [PROTO-LUNG-001].\n"
                f"Fontes utilizadas: [{patient_id}], [PROTO-LUNG-001].\n"
                "Observação: Requer validação humana; não permite concluir diagnóstico definitivo."
            )
            rows.append(
                {
                    "qid": patient_id,
                    "question_type": "synthetic_report",
                    "question_anon": prompt,
                    "answer_anon": answer,
                }
            )

    # 3. Safe prescription & medication reconciliation models (10 patients x 2 queries = 20 rows)
    for i in range(10):
        patient_id = f"SYN-REC-{i:03d}"
        symptom = ["broncoespasmo moderado", "tosse seca persistente", "dor torácica ventilatório-dependente"][i % 3]
        patient = PatientContext(
            patient_id=patient_id,
            name_synthetic=f"Synthetic Rx {i}",
            age=50 + i,
            sex="F" if i % 2 else "M",
            smoking_history="never smoker",
            last_updated_at="2026-09-06T00:00:00",
            symptoms=[{"symptom": symptom, "severity": "moderate"}],
            recent_notes=[{"note_type": "medicacoes_em_uso", "content": "Em uso de anti-hipertensivo regular. Sem alergias conhecidas registradas.", "created_at": "2026-09-06"}],
        )
        protocol = SourceReference(
            source_id="PROTO-MED-002",
            title="Reconciliação Medicamentosa e Segurança na Prescrição",
            type="protocol",
            snippet="Toda recomendação terapêutica deve listar medicações em uso e exige validação e assinatura por médico assistente habilitado.",
        )
        for question in [
            "Qual o procedimento para conferência medicamentosa e elaboração de modelo de receita para este paciente?",
            "Como registrar a conciliação terapêutica para avaliação do médico assistente?",
        ]:
            prompt = build_user_prompt(question, patient, [protocol], [], [], ["Revisar medicações em uso antes de nova prescrição."])
            answer = (
                f"Resumo: Paciente refere {symptom}, em uso de medicação contínua sem alergias relatadas [{patient_id}].\n"
                f"Contexto do paciente: Histórico levantado em prontuário [{patient_id}].\n"
                f"Conduta sugerida: Realizar conciliação medicamentosa conforme [PROTO-MED-002] e submeter minuta de receita para validação e assinatura do médico assistente.\n"
                f"Justificativa: O assistente não prescreve diretamente nem estabelece doses autônomas [PROTO-MED-002].\n"
                f"Fontes utilizadas: [{patient_id}], [PROTO-MED-002].\n"
                "Observação: Requer validação humana; prescrição direta sem revisão médica é vedada."
            )
            rows.append(
                {
                    "qid": patient_id,
                    "question_type": "synthetic_prescription",
                    "question_anon": prompt,
                    "answer_anon": answer,
                }
            )

    # 4. Internal Hospital Procedures (10 patients x 2 queries = 20 rows)
    for i in range(10):
        patient_id = f"SYN-PROC-{i:03d}"
        proc = ["broncoscopia diagnóstica", "biópsia pulmonar percutânea guiada por TC", "espirometria com prova broncodilatadora"][i % 3]
        patient = PatientContext(
            patient_id=patient_id,
            name_synthetic=f"Synthetic Proc {i}",
            age=55 + i,
            sex="M" if i % 2 else "F",
            smoking_history="active smoker",
            last_updated_at="2026-09-07T00:00:00",
            pending_exams=[{"exam_name": proc, "status": "scheduled", "requested_at": "2026-09-07"}],
        )
        protocol = SourceReference(
            source_id="PROTO-PROC-003",
            title="Procedimentos Invasivos e Cuidados Pré-Operatórios Pulmonares",
            type="protocol",
            snippet="Procedimentos diagnósticos exigem verificação de coagulograma, jejum e termo de consentimento livre e esclarecido.",
        )
        for question in [
            f"Quais os preparos e procedimentos internos necessários para a realização de {proc}?",
            f"Qual o fluxo operacional hospitalar para o procedimento {proc} agendado?",
        ]:
            prompt = build_user_prompt(question, patient, [protocol], [proc], ["Verificar jejum e exames prévios."], ["Checar termo de consentimento."])
            answer = (
                f"Resumo: Procedimento de {proc} agendado para o paciente [{patient_id}].\n"
                f"Contexto do paciente: Registro hospitalar [{patient_id}].\n"
                f"Conduta sugerida: Seguir checklist operacional do protocolo interno [PROTO-PROC-003] (jejum, triagem de coagulação e consentimento informado).\n"
                f"Justificativa: Conformidade com os padrões de segurança do paciente para procedimentos pulmonares [PROTO-PROC-003].\n"
                f"Fontes utilizadas: [{patient_id}], [PROTO-PROC-003].\n"
                "Observação: Requer validação humana; orientações sujeitas à liberação da equipe médica."
            )
            rows.append(
                {
                    "qid": patient_id,
                    "question_type": "synthetic_procedure",
                    "question_anon": prompt,
                    "answer_anon": answer,
                }
            )

    return rows
