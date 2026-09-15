"""Export enriched synthetic dataset for Phase 3 fine-tuning.

Produces jsonl data including:
- Operational pending exams (SYN-000 to SYN-029)
- Structured diagnostic reports (SYN-REP-000 to SYN-REP-014)
- Medication reconciliation & prescription models (SYN-REC-000 to SYN-REC-009)
- Internal hospital procedures (SYN-PROC-000 to SYN-PROC-009)

Uses standard library only so it can run in any environment.
"""
import json
from pathlib import Path

OUTPUT_FILE = Path(__file__).parent.parent / "llm_finetuning" / "data" / "context_examples.jsonl"

FORMAT_INSTRUCTIONS = (
    "Responda obrigatoriamente neste formato, em texto simples:\n"
    "Resumo:\n"
    "Contexto do paciente:\n"
    "Conduta sugerida:\n"
    "Justificativa:\n"
    "Fontes utilizadas:\n"
    "Observação:\n\n"
    "Regras:\n"
    "- Não diga que há diagnóstico confirmado.\n"
    "- Não prescreva diretamente.\n"
    "- Cite as fontes utilizadas (registro do paciente e protocolos) de forma explícita.\n"
    "- Cite IDs exatos entre colchetes, por exemplo [P001] ou [PROTO-EXAMS-001].\n"
    "- Use apenas os fatos fornecidos. Se faltar informação, diga explicitamente.\n"
    "- Ao responder sobre exames pendentes, inclua o exam_name exato.\n"
    "- Não copie o JSON nem invente resultados, duração de sintomas ou confirmações.\n"
    "- Seja objetivo.\n"
    "- Não use markdown.\n"
)


def format_user_prompt(question, patient_payload, protocols_snippet):
    return (
        "Pergunta clínica:\n"
        f"{question}\n\n"
        "Informações atualizadas do paciente (base estruturada):\n"
        f"{json.dumps(patient_payload, ensure_ascii=False, indent=2)}\n\n"
        "Protocolos e referências recuperadas:\n"
        f"{protocols_snippet}\n\n"
        f"{FORMAT_INSTRUCTIONS}"
    )


def generate_all():
    rows = []

    # 1. Operational pending exams (30 patients x 2 queries = 60 rows)
    for i in range(30):
        patient_id = f"SYN-{i:03d}"
        exam = ["chest_ct", "spirometry", "chest_xray"][i % 3]
        payload = {
            "patient_id": patient_id,
            "name_synthetic": f"Synthetic {i}",
            "age": 30 + i,
            "sex": "F" if i % 2 else "M",
            "smoking_history": "unknown",
            "last_updated_at": "2026-09-01T00:00:00",
            "symptoms": [],
            "pending_exams": [{"exam_name": exam, "status": "pending", "requested_at": "2026-09-01"}],
            "latest_findings": [],
            "recent_notes": [],
            "pending_exams_reviewed": [exam],
            "alerts": [],
            "recommended_actions": [],
        }
        for q in [
            "Qual exame está pendente? Indique a fonte.",
            "Which pending exam should the clinician review? Cite its source.",
        ]:
            prompt = format_user_prompt(q, payload, "Nenhum protocolo relevante foi recuperado.")
            answer = (
                f"Resumo: O exame {exam} está pendente [{patient_id}].\n"
                f"Contexto do paciente: Não há resultado desse exame no contexto fornecido.\n"
                f"Conduta sugerida: O clínico deve revisar o status do exame {exam}.\n"
                f"Justificativa: O registro informa status pending [{patient_id}].\n"
                f"Fontes utilizadas: [{patient_id}].\n"
                "Observação: Requer validação humana; não permite concluir diagnóstico."
            )
            rows.append({
                "qid": patient_id,
                "question_type": "synthetic_context",
                "question_anon": prompt,
                "answer_anon": answer,
            })

    # 2. Internal report models / Laudos estruturados (15 patients x 2 queries = 30 rows)
    for i in range(15):
        patient_id = f"SYN-REP-{i:03d}"
        finding = ["nódulo pulmonar em lobo superior direito de 6mm", "espessamento pleural bilateral", "infiltrado intersticial difuso"][i % 3]
        exam_type = ["Tomografia de Tórax", "Radiografia de Tórax", "Tomografia Computadorizada de Alta Resolução"][i % 3]
        payload = {
            "patient_id": patient_id,
            "name_synthetic": f"Synthetic Report {i}",
            "age": 45 + i,
            "sex": "M" if i % 2 else "F",
            "smoking_history": "active smoker" if i % 2 == 0 else "former smoker",
            "last_updated_at": "2026-09-05T00:00:00",
            "symptoms": [],
            "pending_exams": [],
            "latest_findings": [{"finding": f"{exam_type}: {finding}", "risk_level": "moderate", "created_at": "2026-09-05"}],
            "recent_notes": [{"note_type": "laudo_preliminar", "content": f"Achado radiológico: {finding}. Correlação clínica recomendada.", "created_at": "2026-09-05"}],
            "pending_exams_reviewed": [],
            "alerts": ["Achado incidental identificado."],
            "recommended_actions": ["Revisar imagem com especialista."],
        }
        proto_snippet = json.dumps([{
            "protocol_id": "PROTO-LUNG-001",
            "title": "Manejo de Achados Pulmonares Incidentais",
            "snippet": "Nódulos indeterminados requerem acompanhamento tomográfico seriado e correlação clínica sem diagnóstico imediato."
        }], ensure_ascii=False, indent=2)

        for q in [
            "Qual o modelo de estruturação de laudo e conduta recomendada para este achado?",
            "Como estruturar o resumo clínico deste achado de imagem com base no protocolo?",
        ]:
            prompt = format_user_prompt(q, payload, proto_snippet)
            answer = (
                f"Resumo: Laudo preliminar indica achado de {finding} em {exam_type} [{patient_id}].\n"
                f"Contexto do paciente: Paciente {payload['age']} anos, histórico {payload['smoking_history']} [{patient_id}].\n"
                f"Conduta sugerida: Adotar modelo de laudo descritivo estruturado e seguir protocolo [PROTO-LUNG-001] para acompanhamento seriado.\n"
                f"Justificativa: Achados indeterminados necessitam de vigilância radiológica comparativa antes de condutas invasivas [PROTO-LUNG-001].\n"
                f"Fontes utilizadas: [{patient_id}], [PROTO-LUNG-001].\n"
                "Observação: Requer validação humana; não permite concluir diagnóstico definitivo."
            )
            rows.append({
                "qid": patient_id,
                "question_type": "synthetic_report",
                "question_anon": prompt,
                "answer_anon": answer,
            })

    # 3. Safe prescription & medication reconciliation models (10 patients x 2 queries = 20 rows)
    for i in range(10):
        patient_id = f"SYN-REC-{i:03d}"
        symptom = ["broncoespasmo moderado", "tosse seca persistente", "dor torácica ventilatório-dependente"][i % 3]
        payload = {
            "patient_id": patient_id,
            "name_synthetic": f"Synthetic Rx {i}",
            "age": 50 + i,
            "sex": "F" if i % 2 else "M",
            "smoking_history": "never smoker",
            "last_updated_at": "2026-09-06T00:00:00",
            "symptoms": [{"symptom": symptom, "severity": "moderate"}],
            "pending_exams": [],
            "latest_findings": [],
            "recent_notes": [{"note_type": "medicacoes_em_uso", "content": "Em uso de anti-hipertensivo regular. Sem alergias conhecidas registradas.", "created_at": "2026-09-06"}],
            "pending_exams_reviewed": [],
            "alerts": [],
            "recommended_actions": ["Revisar medicações em uso antes de nova prescrição."],
        }
        proto_snippet = json.dumps([{
            "protocol_id": "PROTO-MED-002",
            "title": "Reconciliação Medicamentosa e Segurança na Prescrição",
            "snippet": "Toda recomendação terapêutica deve listar medicações em uso e exige validação e assinatura por médico assistente habilitado."
        }], ensure_ascii=False, indent=2)

        for q in [
            "Qual o procedimento para conferência medicamentosa e elaboração de modelo de receita para este paciente?",
            "Como registrar a conciliação terapêutica para avaliação do médico assistente?",
        ]:
            prompt = format_user_prompt(q, payload, proto_snippet)
            answer = (
                f"Resumo: Paciente refere {symptom}, em uso de medicação contínua sem alergias relatadas [{patient_id}].\n"
                f"Contexto do paciente: Histórico levantado em prontuário [{patient_id}].\n"
                f"Conduta sugerida: Realizar conciliação medicamentosa conforme [PROTO-MED-002] e submeter minuta de receita para validação e assinatura do médico assistente.\n"
                f"Justificativa: O assistente não prescreve diretamente nem estabelece doses autônomas [PROTO-MED-002].\n"
                f"Fontes utilizadas: [{patient_id}], [PROTO-MED-002].\n"
                "Observação: Requer validação humana; prescrição direta sem revisão médica é vedada."
            )
            rows.append({
                "qid": patient_id,
                "question_type": "synthetic_prescription",
                "question_anon": prompt,
                "answer_anon": answer,
            })

    # 4. Internal Hospital Procedures (10 patients x 2 queries = 20 rows)
    for i in range(10):
        patient_id = f"SYN-PROC-{i:03d}"
        proc = ["broncoscopia diagnóstica", "biópsia pulmonar percutânea guiada por TC", "espirometria com prova broncodilatadora"][i % 3]
        payload = {
            "patient_id": patient_id,
            "name_synthetic": f"Synthetic Proc {i}",
            "age": 55 + i,
            "sex": "M" if i % 2 else "F",
            "smoking_history": "active smoker",
            "last_updated_at": "2026-09-07T00:00:00",
            "symptoms": [],
            "pending_exams": [{"exam_name": proc, "status": "scheduled", "requested_at": "2026-09-07"}],
            "latest_findings": [],
            "recent_notes": [],
            "pending_exams_reviewed": [proc],
            "alerts": ["Verificar jejum e exames prévios."],
            "recommended_actions": ["Checar termo de consentimento."],
        }
        proto_snippet = json.dumps([{
            "protocol_id": "PROTO-PROC-003",
            "title": "Procedimentos Invasivos e Cuidados Pré-Operatórios Pulmonares",
            "snippet": "Procedimentos diagnósticos exigem verificação de coagulograma, jejum e termo de consentimento livre e esclarecido."
        }], ensure_ascii=False, indent=2)

        for q in [
            f"Quais os preparos e procedimentos internos necessários para a realização de {proc}?",
            f"Qual o fluxo operacional hospitalar para o procedimento {proc} agendado?",
        ]:
            prompt = format_user_prompt(q, payload, proto_snippet)
            answer = (
                f"Resumo: Procedimento de {proc} agendado para o paciente [{patient_id}].\n"
                f"Contexto do paciente: Registro hospitalar [{patient_id}].\n"
                f"Conduta sugerida: Seguir checklist operacional do protocolo interno [PROTO-PROC-003] (jejum, triagem de coagulação e consentimento informado).\n"
                f"Justificativa: Conformidade com os padrões de segurança do paciente para procedimentos pulmonares [PROTO-PROC-003].\n"
                f"Fontes utilizadas: [{patient_id}], [PROTO-PROC-003].\n"
                "Observação: Requer validação humana; orientações sujeitas à liberação da equipe médica."
            )
            rows.append({
                "qid": patient_id,
                "question_type": "synthetic_procedure",
                "question_anon": prompt,
                "answer_anon": answer,
            })

    return rows


if __name__ == "__main__":
    rows = generate_all()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Sucesso! {len(rows)} exemplos sintéticos gerados e salvos em {OUTPUT_FILE}")
