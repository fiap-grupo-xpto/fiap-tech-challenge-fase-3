import pytest
from backend.assistant.quality import validate_answer_quality
from backend.assistant.schemas import PatientContext
from llm_finetuning.context_examples import build_context_examples


@pytest.fixture
def patient():
    return PatientContext(patient_id="P001", name_synthetic="Synthetic", age=58, sex="M",
        smoking_history="unknown", last_updated_at="2026-09-01",
        pending_exams=[{"exam_name": "chest_ct", "status": "pending"}])


def test_grounded_exam_answer(patient):
    result = validate_answer_quality("O exame chest_ct está pendente [P001]. Requer revisão clínica.",
                                     "Qual exame está pendente?", patient, [])
    assert not result.blocked


@pytest.mark.parametrize("answer,rule", [
    ("Revisar exames.", "quality:missing_source"),
    ('{"patient_id": "P001"}', "quality:context_echo"),
    ("Revisar histórico [P001].", "quality:pending_exam_not_answered"),
    ("Dispneia confirmada por exame de pulmão [P001].", "quality:unvalidated_confirmation"),
    (("Pesquisa clínica com achados repetidos [P001]\n" * 3), "quality:repetition"),
])
def test_unacceptable_answers(patient, answer, rule):
    result = validate_answer_quality(answer, "Qual exame está pendente?", patient, [])
    assert result.blocked
    assert rule in result.matched_rules


def test_context_examples_have_grouped_variants_and_real_prompt():
    examples = build_context_examples()
    assert len(examples) == 60
    assert len({e['qid'] for e in examples}) == 30
    for row in examples:
        assert row['question_anon'].startswith('Pergunta clínica:')
        assert f"[{row['qid']}]" in row['answer_anon']
        assert 'pending_exams' in row['question_anon']


def test_observed_real_model_failure_is_blocked(patient):
    answer = ("Resumo: Paciente sintético 001.\n"
              "Persistência de cachorro: confirmado por exame de pulmão\n" * 3)
    result = validate_answer_quality(answer,
        "Which pending exam should the clinician review? Cite its source.", patient, [])
    assert result.blocked
    assert {"quality:missing_source", "quality:repetition", "quality:unvalidated_confirmation",
            "quality:pending_exam_not_answered"}.issubset(result.matched_rules)
