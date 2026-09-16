from backend.assistant.guardrails import (
    check_citation_consistency,
    check_input_safety,
    check_output_safety,
)


def test_check_input_safety_blocks_prescription_request():
    result = check_input_safety("Prescreva um remédio para a tosse do paciente.")
    assert result.blocked is True
    assert result.requires_human_review is True
    assert result.matched_rules


def test_check_input_safety_blocks_dosage_request():
    result = check_input_safety("Quanto devo tomar de amoxicilina?")
    assert result.blocked is True


def test_check_input_safety_allows_normal_clinical_question():
    result = check_input_safety(
        "O paciente apresenta tosse persistente e dispneia. O que revisar em seguida?"
    )
    assert result.blocked is False
    assert result.matched_rules == []


def test_check_output_safety_blocks_dosage_instruction():
    result = check_output_safety(
        "Conduta sugerida: tome 500 mg de amoxicilina a cada 8 horas."
    )
    assert result.blocked is True
    assert result.requires_human_review is True


def test_check_output_safety_blocks_definitive_diagnosis():
    result = check_output_safety("Diagnóstico confirmado: o paciente tem câncer de pulmão.")
    assert result.blocked is True


def test_check_output_safety_allows_negated_confirmed_diagnosis():
    result = check_output_safety(
        "Não há diagnóstico confirmado. O caso requer revisão clínica."
    )
    assert result.blocked is False
    assert result.matched_rules == []


def test_check_output_safety_allows_supportive_language():
    result = check_output_safety(
        "Resumo: revisar exames pendentes\n"
        "Conduta sugerida: encaminhar para avaliação clínica\n"
        "Fontes utilizadas: protocolo de triagem pulmonar\n"
        "Observação: apoio à decisão clínica, sem substituir avaliação médica"
    )
    assert result.blocked is False
    assert result.matched_rules == []


# Casos que vazavam pelo filtro anterior (reportados em revisão adversarial) — cobertos
# depois do endurecimento dos padrões de entrada/saída.


def test_check_input_safety_blocks_bare_imperative_prescription_request():
    result = check_input_safety("Receite um medicamento para o paciente.")
    assert result.blocked is True


def test_check_output_safety_blocks_treatment_instruction_without_dosage():
    result = check_output_safety("Conduta sugerida: inicie amoxicilina por via oral.")
    assert result.blocked is True


def test_check_output_safety_blocks_bare_definitive_diagnosis():
    result = check_output_safety("O paciente tem câncer de pulmão.")
    assert result.blocked is True


def test_check_output_safety_blocks_english_dosage_instruction():
    result = check_output_safety("Suggested plan: take one tablet daily.")
    assert result.blocked is True


def test_check_output_safety_blocks_english_prescription_verb():
    result = check_output_safety("Recommendation: prescribe amoxicillin for the patient.")
    assert result.blocked is True


def test_check_output_safety_blocks_english_definitive_diagnosis():
    result = check_output_safety("The patient has lung cancer.")
    assert result.blocked is True


def test_check_input_safety_still_allows_normal_question_in_english():
    result = check_input_safety(
        "The patient reports persistent cough and dyspnea. What should be reviewed next?"
    )
    assert result.blocked is False


# Segunda rodada de revisão adversarial: 3 bypasses novos + 2 falsos positivos
# introduzidos pelo endurecimento anterior (o gap de até 20 caracteres entre "tem" e a
# doença capturava também menções hedged como "tem histórico de X" / "tem sintomas de X").


def test_check_input_safety_blocks_present_tense_dosage_question():
    result = check_input_safety("Qual remédio eu tomo?")
    assert result.blocked is True


def test_check_output_safety_blocks_use_verb_with_administration_route():
    result = check_output_safety("Use amoxicilina por via oral.")
    assert result.blocked is True


def test_check_output_safety_blocks_tome_without_explicit_dosage():
    result = check_output_safety("Tome um comprimido por dia.")
    assert result.blocked is True


def test_check_output_safety_allows_history_mention_without_blocking():
    result = check_output_safety("O paciente tem histórico de asma.")
    assert result.blocked is False
    assert result.matched_rules == []


def test_check_output_safety_allows_symptom_mention_without_blocking():
    result = check_output_safety("O paciente tem sintomas de pneumonia.")
    assert result.blocked is False
    assert result.matched_rules == []


def test_check_citation_consistency_flags_uncited_protocol_reference():
    result = check_citation_consistency(
        "Fontes utilizadas: PROTO-HITL-001.", known_source_ids=["PROTO-TRIAGE-001"]
    )
    assert result.blocked is True
    assert result.requires_human_review is True
    assert "uncited_protocol_reference:PROTO-HITL-001" in result.matched_rules


def test_check_citation_consistency_allows_cited_protocol_reference():
    result = check_citation_consistency(
        "Fontes utilizadas: PROTO-TRIAGE-001.", known_source_ids=["PROTO-TRIAGE-001"]
    )
    assert result.blocked is False
    assert result.requires_human_review is False
    assert result.matched_rules == []


def test_check_citation_consistency_allows_answer_without_protocol_mentions():
    result = check_citation_consistency(
        "Resumo: apoio à decisão clínica.", known_source_ids=["PROTO-TRIAGE-001"]
    )
    assert result.blocked is False
    assert result.matched_rules == []


def test_new_prescriptions_and_negation():
    for text in ["Inicie amoxicilina.", "Tome dois comprimidos por dia."]:
        assert check_output_safety(text).blocked
    for text in ["Não tome 500 mg de amoxicilina.", "O paciente tem provável pneumonia."]:
        assert not check_output_safety(text).blocked
    assert check_output_safety("Não tome amoxicilina; tome dois comprimidos por dia.").blocked


def test_unknown_external_and_multisegment_sources_block():
    for text in ["PROTO-DOES-NOT-EXIST-999", "Fonte: inventado.pdf", "Fonte: https://example.org/x"]:
        assert check_citation_consistency(text, []).blocked
