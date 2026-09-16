from __future__ import annotations

import re
import unicodedata
from typing import List, Sequence

from backend.assistant.schemas import GuardrailResult

# IDs de protocolo no formato usado em backend/data/bootstrap_hospital_db.py
# (ex.: PROTO-TRIAGE-001).
_PROTOCOL_ID_PATTERN = re.compile(r"\bPROTO-(?:[A-Z0-9]+-)+\d+\b", re.IGNORECASE)

# Termos de rota/forma de administração que, combinados com um verbo de instrução de
# tratamento, indicam prescrição mesmo sem dosagem numérica explícita.
_ADMINISTRATION_TERMS_PT = (
    r"via\s+oral|via\s+intravenosa|via\s+intramuscular|via\s+subcutanea|comprimido|"
    r"capsula|ampola|xarope|pomada|gotas?|injecao"
)
_ADMINISTRATION_TERMS_EN = r"orally|intravenously|by\s+mouth|tablet|capsule|injection|ointment|drops?"

_TREATMENT_VERBS_PT = (
    r"inicie|iniciar|administre|administrar|aplique|aplicar|utilize|utilizar|"
    r"use|usar|tome|tomar"
)
_TREATMENT_VERBS_EN = r"start|administer|apply|give|use|take"

# Doenças/condições cujo uso em "o paciente tem X" / "the patient has X" caracteriza
# diagnóstico fechado. Lista não exaustiva por natureza (abordagem por palavra-chave) —
# cobre as condições mais citadas/críticas; ver limitações no README do módulo.
_DIAGNOSIS_NOUNS_PT = (
    r"cancer|tumor|tuberculose|pneumonia|covid|diabetes|hipertensao|dpoc|asma|"
    r"infarto|avc|derrame|metastase"
)
_DIAGNOSIS_NOUNS_EN = (
    r"cancer|tumor|tuberculosis|pneumonia|covid|diabetes|hypertension|copd|asthma|"
    r"heart\s+attack|stroke|metastasis"
)

# Pedidos explícitos de prescrição/posologia feitos pelo usuário (PT/EN).
_INPUT_PRESCRIPTION_PATTERNS = [
    r"\bprescreva\b",
    r"\breceite\b",
    r"\bme\s+receit\w*\b",
    r"\bpasse?\s+(me\s+)?(uma\s+)?receita\b",
    r"\bindique\s+um\s+(remedio|medicamento)\b",
    r"\bqual\s+(remedio|medicamento|dose|posologia)\b.{0,30}\b(devo|deveria|preciso)\s+tomar\b",
    r"\bqual\s+(remedio|medicamento|dose|posologia)\b.{0,30}\btomo\b",
    r"\bquanto\s+devo\s+tomar\b",
    r"\bprescribe\b",
    r"\bwrite\s+(me\s+)?a\s+prescription\b",
    r"\bwhat\s+(medication|drug|dose)\b.{0,30}\bshould\s+i\s+take\b",
    r"\bhow\s+much\s+should\s+i\s+take\b",
]

# Pedidos explícitos de confirmação fechada de diagnóstico feitos pelo usuário (PT/EN).
_INPUT_DEFINITIVE_DIAGNOSIS_PATTERNS = [
    r"\bconfirme\s+(que|se)\s+(eu\s+)?tenho\b",
    r"\bme\s+diga\s+com\s+certeza\s+se\s+(eu\s+)?tenho\b",
    r"\bconfirm\s+(that\s+)?i\s+have\b",
    r"\btell\s+me\s+for\s+sure\s+if\s+i\s+have\b",
]

# Linguagem de prescrição/posologia na resposta gerada pela LLM (PT/EN).
_OUTPUT_PRESCRIPTION_PATTERNS = [
    r"\b(?:inicie|use|tome|administre|start|take)\s+(?:amoxicilina|amoxicillin|ibuprofeno|ibuprofen|paracetamol|insulina|insulin)\b",
    r"\b(?:tome|tomar|take)\s+(?:um|uma|dois|duas|tres|one|two|three|\d+)\s+(?:comprimidos?|capsulas?|tablets?|pills?)\b",

    r"\btome\s+\d",
    r"\btomar\s+\d",
    r"\badministr\w*\s+\d",
    r"\bprescrevo\b",
    r"\breceito\b",
    r"\bposologia\s*:",
    r"\bdose\s+recomendada\b",
    r"\bdose\s+de\s+\d",
    r"\bmg/kg\b",
    r"\d+\s*(mg|mcg|ml|g|ui)\b",
    r"\bcomprimido(s)?\s+de\b",
    r"\bcapsula(s)?\s+de\b",
    r"\binjete\b",
    rf"\b({_TREATMENT_VERBS_PT})\b.{{0,40}}\b({_ADMINISTRATION_TERMS_PT})\b",
    r"\btake\s+\d",
    r"\btake\s+one\s+(tablet|pill|capsule|dose)\b",
    r"\bprescrib(e|ing)\b",
    rf"\b({_TREATMENT_VERBS_EN})\b.{{0,40}}\b({_ADMINISTRATION_TERMS_EN})\b",
]

# Linguagem de diagnóstico fechado/definitivo na resposta gerada pela LLM (PT/EN).
_OUTPUT_DEFINITIVE_DIAGNOSIS_PATTERNS = [
    r"\bdiagnostico\s+confirmado\b",
    r"\bdiagnostico\s+e\s+definitivo\b",
    r"\bcom\s+certeza\s+(e|tem)\b",
    r"\be\s+definitivamente\b",
    r"\bconfirmo\s+que\s+o\s+paciente\s+tem\b",
    r"\bo\s+paciente\s+definitivamente\s+tem\b",
    # No máximo UMA palavra entre o verbo e a doença (ex.: "tem CÂNCER de pulmão", "has
    # LUNG cancer") — não um trecho arbitrário de até 20 caracteres. Isso é o que evita
    # bloquear "o paciente tem histórico de asma" / "tem sintomas de pneumonia": entre
    # "tem" e a doença há duas palavras ("histórico de" / "sintomas de"), não uma só,
    # então o padrão não casa. Menções hedged (histórico/sintomas/suspeita/risco de X)
    # não são diagnóstico fechado e não devem ser bloqueadas.
    rf"\bo\s+paciente\s+tem\s+(?:\w+\s+)?({_DIAGNOSIS_NOUNS_PT})\b",
    rf"\bvoce\s+tem\s+(?:\w+\s+)?({_DIAGNOSIS_NOUNS_PT})\b",
    r"\bdiagnosis\s+confirmed\b",
    r"\bdefinitely\s+has\b",
    rf"\bthe\s+patient\s+has\s+(?:\w+\s+)?({_DIAGNOSIS_NOUNS_EN})\b",
    rf"\byou\s+have\s+(?:\w+\s+)?({_DIAGNOSIS_NOUNS_EN})\b",
]


def _normalize(text: str) -> str:
    text = (text or "").lower()
    return "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )


def _match_patterns(normalized_text: str, patterns: List[str], label: str) -> List[str]:
    return [f"{label}:{pattern}" for pattern in patterns if re.search(pattern, normalized_text)]


def check_input_safety(question: str) -> GuardrailResult:
    """Bloqueia perguntas que pedem prescrição direta ou diagnóstico fechado (PT/EN)."""
    normalized = _normalize(question)

    matched = _match_patterns(normalized, _INPUT_PRESCRIPTION_PATTERNS, "input_prescription")
    matched += _match_patterns(
        normalized, _INPUT_DEFINITIVE_DIAGNOSIS_PATTERNS, "input_definitive_diagnosis"
    )

    if matched:
        return GuardrailResult(
            blocked=True,
            requires_human_review=True,
            reason=(
                "A pergunta solicita diretamente prescrição de medicamento ou confirmação "
                "definitiva de diagnóstico, fora dos limites de atuação do assistente "
                "(protocolos PROTO-NO-RX-001 e PROTO-HITL-001)."
            ),
            matched_rules=matched,
        )

    return GuardrailResult(blocked=False, requires_human_review=False)


def check_citation_consistency(answer_text: str, known_source_ids: Sequence[str]) -> GuardrailResult:
    """Bloqueia IDs desconhecidos e referências externas não recuperadas.

    O catálogo atual é SQLite: URLs/PDFs não são fontes fornecidas ao modelo.
    Esta checagem não comprova que cada afirmação é sustentada pelo documento.
    """
    mentioned = {m.upper() for m in _PROTOCOL_ID_PATTERN.findall(answer_text or "")}
    known = {source_id.upper() for source_id in known_source_ids}
    uncited = sorted(mentioned - known)

    uncited += re.findall(r"https?://[^\s]+|\b[\w-]+\.pdf\b", answer_text or "", re.I)
    if uncited:
        return GuardrailResult(
            blocked=True,
            requires_human_review=True,
            reason=(
                f"A resposta cita protocolo(s) que não estavam entre as fontes recuperadas "
                f"para esta pergunta: {', '.join(uncited)}. Possível citação inventada."
            ),
            matched_rules=[f"uncited_protocol_reference:{ref}" for ref in uncited],
        )

    return GuardrailResult(blocked=False, requires_human_review=False)


def check_output_safety(answer_text: str) -> GuardrailResult:
    """Bloqueia respostas da LLM com prescrição direta ou diagnóstico fechado (PT/EN).

    Baseado em regras de palavra-chave/regex — cobre os padrões mais comuns de violação,
    mas não substitui um classificador de linguagem natural. Limitação conhecida: uma
    paráfrase fora das listas acima pode não ser detectada.
    """
    # Match each clause independently so a refusal does not hide a later instruction.
    # Também removemos negações explícitas de diagnóstico: "não há diagnóstico
    # confirmado" é uma ressalva segura, não uma afirmação diagnóstica.
    clauses = re.split(r"[.!?;\n]+|\bmas\b|\bbut\b", _normalize(answer_text))
    normalized = "\n".join(
        clause for clause in clauses
        if not re.match(
            r"^\s*(?:"
            r"nao\s+(?:tome|use|inicie|administre)|"
            r"do\s+not\s+(?:take|use|start)|"
            r")\b",
            clause,
        )
    )
    normalized = re.sub(r"\b(?:provavel|possivel|suspeita\s+de|historico\s+de|sintomas\s+de)\b[^.!?;\n]*", "", normalized)

    matched = _match_patterns(normalized, _OUTPUT_PRESCRIPTION_PATTERNS, "output_prescription")
    matched += _match_patterns(
        normalized, _OUTPUT_DEFINITIVE_DIAGNOSIS_PATTERNS, "output_definitive_diagnosis"
    )

    if matched:
        return GuardrailResult(
            blocked=True,
            requires_human_review=True,
            reason=(
                "A resposta gerada continha linguagem de prescrição direta ou diagnóstico "
                "definitivo, violando os limites de atuação do assistente (protocolos "
                "PROTO-NO-RX-001 e PROTO-HITL-001). A resposta foi bloqueada."
            ),
            matched_rules=matched,
        )

    return GuardrailResult(blocked=False, requires_human_review=False)
