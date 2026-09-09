# Item 2 - Assistente Médico (LangChain + LangGraph)

Este módulo implementa o Item 2 do Tech Challenge Fase 3:
- integração com uma LLM customizada (Item 1) quando disponível;
- fallback para Gemini quando os artefatos locais não estão disponíveis;
- consulta a base estruturada (SQLite) para prontuários e protocolos;
- orquestração com LangChain e fluxo de decisão com LangGraph;
- endpoint dedicado `POST /assistant/query`.

## Banco de Dados

O banco SQLite padrão é criado em:
- [backend/data/hospital.db](file:///C:/Coding/fiap-tech-challenge-fase-3/backend/data/hospital.db)

Se o arquivo não existir, ele é bootstrapado automaticamente via:
- [bootstrap_hospital_db.py](file:///C:/Coding/fiap-tech-challenge-fase-3/backend/data/bootstrap_hospital_db.py)

## Endpoint

`POST /assistant/query`

Payload (exemplo):

```json
{
  "patient_id": "P001",
  "question": "O paciente apresenta tosse persistente e dispneia. O que revisar em seguida?",
  "include_protocols": true,
  "include_pending_exams": true,
  "force_llm_mode": "auto"
}
```

Campos relevantes de resposta:
- `patient_context_used` (contexto do paciente consultado via SQLite)
- `sources_used` (protocolos e registro do paciente)
- `alerts` e `recommended_actions`
- `llm_backend_used`, `custom_llm_available`, `fallback_used` (transparência do backend LLM)

## Modos de LLM

Controlado por request (`force_llm_mode`) ou pela variável de ambiente `ASSISTANT_LLM_MODE`:
- `auto` (padrão): tenta Item 1, fallback para Gemini
- `item1_only`: falha se Item 1 não estiver disponível
- `gemini_only`: usa Gemini diretamente

Artefatos esperados do Item 1 (frozen contract):
- [llama_medical_lora_model](file:///C:/Coding/fiap-tech-challenge-fase-3/llm_finetuning/artifacts/llama_medical_lora_model)
- [tokenizer](file:///C:/Coding/fiap-tech-challenge-fase-3/llm_finetuning/artifacts/tokenizer)

