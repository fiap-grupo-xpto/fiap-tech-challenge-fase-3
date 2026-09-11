from __future__ import annotations

import json
from typing import List, Sequence

from langchain_core.prompts import ChatPromptTemplate

from backend.assistant.schemas import PatientContext, SourceReference


ASSISTANT_SYSTEM_PROMPT = (
    "Você é um assistente médico especialista responsável por apoiar decisões clínicas "
    "com clareza, rigor e alinhamento com protocolos hospitalares. "
    "Você NÃO confirma diagnósticos, NÃO prescreve diretamente e NÃO substitui avaliação médica. "
    "Todas as recomendações devem ser validadas por um clínico responsável."
)


def build_output_instructions() -> str:
    return (
        "Responda obrigatoriamente neste formato, em texto simples:\n"
        "Resumo:\n"
        "Contexto do paciente:\n"
        "Conduta sugerida:\n"
        "Justificativa:\n"
        "Fontes utilizadas:\n"
        "Observação:\n"
        "\n"
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


def _format_protocol_snippets(protocols: Sequence[SourceReference]) -> str:
    if not protocols:
        return "Nenhum protocolo relevante foi recuperado."

    payload = [
        {
            "protocol_id": protocol.source_id,
            "title": protocol.title,
            "snippet": protocol.snippet,
        }
        for protocol in protocols
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_user_prompt(
    question: str,
    patient_context: PatientContext,
    protocols: Sequence[SourceReference],
    pending_exams_reviewed: Sequence[str],
    alerts: Sequence[str],
    recommended_actions: Sequence[str],
) -> str:
    patient_payload = patient_context.model_dump()
    patient_payload["pending_exams_reviewed"] = list(pending_exams_reviewed)
    patient_payload["alerts"] = list(alerts)
    patient_payload["recommended_actions"] = list(recommended_actions)

    return (
        "Pergunta clínica:\n"
        f"{question}\n\n"
        "Informações atualizadas do paciente (base estruturada):\n"
        f"{json.dumps(patient_payload, ensure_ascii=False, indent=2)}\n\n"
        "Protocolos e referências recuperadas:\n"
        f"{_format_protocol_snippets(protocols)}\n\n"
        f"{build_output_instructions()}"
    )


def build_chat_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", "{system_prompt}"),
            ("user", "{user_prompt}"),
        ]
    )

