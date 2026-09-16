# Item 2 e Item 3 - Assistente Médico (LangChain + LangGraph) e Segurança/Validação

Este módulo implementa o Item 2 e o Item 3 do Tech Challenge Fase 3:
- integração com uma LLM customizada (Item 1) quando disponível;
- cadeia de recuperação Item 1 → Gemini → síntese determinística segura quando os dois provedores falham;
- consulta a base estruturada (SQLite) para prontuários e protocolos;
- orquestração com LangChain e fluxo de decisão com LangGraph;
- guardrails de entrada/saída que bloqueiam prescrição direta e diagnóstico fechado;
- logging técnico e auditoria persistente de cada interação (tabela `audit_log`);
- explainability: toda resposta traz as fontes usadas (`sources_used`) de forma estruturada;
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
- `sources_used` (protocolos e registro do paciente — explainability)
- `alerts` e `recommended_actions`
- `llm_backend_used`, `custom_llm_available`, `fallback_used` (transparência do backend LLM)
- `status` (`success`, `error` ou `blocked`)
- `requires_human_review` (`true` quando a resposta exige validação de um clínico)
- `blocked` / `block_reason` (`true` quando um guardrail interceptou a pergunta ou a resposta)
- `attempted_backend` / `attempted_backend_error` (preenchidos quando o backend primário falhou e
  o sistema caiu para o fallback — preserva a causa da falha mesmo quando a resposta final é
  `"success"`; sem isso, um fallback bem-sucedido apagava completamente o rastro de que o modelo
  local foi tentado primeiro e por que falhou)
- `request_id` (identificador único da interação, usado para correlacionar com o log de auditoria)

## Segurança e validação (Item 3)

### Limites de atuação / guardrails

Implementados em [backend/assistant/guardrails.py](file:///C:/Coding/fiap-tech-challenge-fase-3/backend/assistant/guardrails.py):
- `check_input_safety`: bloqueia perguntas que pedem prescrição direta, posologia ou confirmação
  fechada de diagnóstico **antes** de consultar o paciente ou chamar a LLM. Cobre PT e EN.
- `check_output_safety`: analisa a resposta gerada pela LLM e bloqueia (substituindo por uma
  mensagem segura) qualquer conteúdo com linguagem de prescrição/posologia (mesmo sem dosagem
  numérica explícita, ex.: "inicie amoxicilina por via oral") ou diagnóstico definitivo (inclusive
  a forma direta "o paciente tem X"), mesmo que o modelo não tenha seguido as instruções do prompt.
  Cobre PT e EN.
- `check_citation_consistency`: **bloqueia** quando a resposta cita um ID de protocolo
  (`PROTO-XXX-000`, inclusive com vários hífens) que não estava entre as fontes de fato
  recuperadas para aquela pergunta, ou quando cita uma URL/nome de PDF — a recuperação
  atual é exclusivamente SQLite, então essas referências são necessariamente inventadas.
- `validate_output` também roda `check_output_safety` sobre `recommended_actions`/`alerts`, não só
  sobre o texto da LLM — hoje esses campos são determinísticos, mas ficam cobertos pela mesma regra
  caso um dia passem a ser influenciados pela LLM.
- **Limitação conhecida:** a detecção é baseada em regras de palavra-chave/regex, não em um
  classificador de linguagem natural — uma paráfrase fora dos padrões cobertos (ver
  `tests/assistant/test_guardrails.py`, que inclui uma bateria adversarial mantida por revisões
  externas) pode não ser detectada, e o inverso também já aconteceu (um alargamento de padrão para
  fechar um bypass bloqueou por engano menções legítimas como "tem histórico de X"). Não deve ser
  tratado como garantia absoluta, e sim como uma camada de defesa complementar às instruções do
  prompt — sujeita a rodadas contínuas de correção.

Esses dois nós (`validate_input` e `validate_output`) estão integrados ao grafo LangGraph em
[backend/assistant/workflow.py](file:///C:/Coding/fiap-tech-challenge-fase-3/backend/assistant/workflow.py).
Quando um guardrail de **entrada** bloqueia, o fluxo desvia (edge condicional) para `format_blocked`, que retorna
`status: "blocked"`, `requires_human_review: true` e o motivo do bloqueio, sem nunca prescrever
diretamente ao usuário.

`requires_human_review` é sempre `true` para qualquer resposta com sugestão do assistente
(`status: "success"` ou `"blocked"`), refletindo a política de `PROTO-HITL-001` ("All assistant
suggestions require clinician validation") — não é condicionado a achado de risco ou a qual
protocolo foi casado por palavra-chave na recuperação (ver `requires_human_validation` em
`backend/assistant/retrievers.py`).

### Logging e auditoria

Implementado em [backend/assistant/audit_log.py](file:///C:/Coding/fiap-tech-challenge-fase-3/backend/assistant/audit_log.py)
e acionado pelo nó `log_interaction` (último nó do grafo, antes de `END`):
- log técnico via `logging` padrão (nível configurável por `ASSISTANT_LOG_LEVEL`);
- persistência estruturada na tabela `audit_log` do SQLite (`request_id`, timestamp, `patient_id`,
  pergunta, resposta, fontes usadas, `system_prompt`/`user_prompt` efetivamente enviados à LLM,
  backend usado, resultado das validações de entrada/saída, se foi bloqueado e o motivo,
  `requires_human_review`, erros).
- `get_interaction(request_id)` permite consultar de volta uma interação já registrada.
- `backend/assistant/service.py` envolve a execução do grafo em um `try/except`: se qualquer nó
  falhar de forma não tratada (ex.: banco de dados indisponível), a interação ainda é registrada
  na auditoria com `status: "error"` antes de devolver uma resposta de erro genérica ao cliente —
  sem esse cuidado, uma falha catastrófica desapareceria da auditoria justamente quando ela mais
  importa.

### Explainability

`sources_used` já é retornado de forma estruturada e determinística (registro do paciente +
protocolos recuperados via `retrieve_protocols`), independente do texto gerado pela LLM — o prompt
também pede explicitamente uma seção "Fontes utilizadas" na resposta em texto livre.

## Modos de LLM

Controlado por request (`force_llm_mode`) ou pela variável de ambiente `ASSISTANT_LLM_MODE`:
- `auto` (padrão) e `item1_only`: ambos iniciam pelo Item 1. Se ele não estiver disponível, não
  carregar ou não gerar resposta, tentam Gemini. A diferença é apenas de intenção na interface;
  ambos preservam a recuperação segura para não interromper uma consulta clínica válida.
- `gemini_only`: chama Gemini diretamente. Se ele falhar, também usa a síntese determinística segura.

Quando qualquer cadeia de provider esgota as tentativas, `generate_answer` não expõe uma mensagem
técnica como resposta clínica: o LangGraph direciona para `format_safe_fallback`. A resposta contém
somente contexto estruturado, exige revisão humana e registra a falha final em `error_message`. A
falha do Item 1 permanece em `attempted_backend_error`; portanto, uma resposta final de sucesso não
apaga a evidência da indisponibilidade dos provedores.

Artefatos esperados do Item 1 (frozen contract):
- [llama_medical_lora_model](file:///C:/Coding/fiap-tech-challenge-fase-3/llm_finetuning/artifacts/llama_medical_lora_model)
- [tokenizer](file:///C:/Coding/fiap-tech-challenge-fase-3/llm_finetuning/artifacts/tokenizer)



## Correções verificadas em 11/09/2026

- Respostas bloqueadas não devolvem `recommended_actions` nem `alerts`.
- Referências desconhecidas no formato PROTO (inclusive IDs com vários hífens),
  URLs e nomes de PDF são bloqueados: a recuperação atual usa exclusivamente SQLite.
- `sources_used` mantém compatibilidade e significa fontes recuperadas; `sources_cited`
  lista somente as referências mencionadas explicitamente pelo ID na resposta liberada.
  Títulos livres e sustentação factual de cada afirmação ainda não são verificados.
- A auditoria preserva `raw_answer` e a falha inicial do fallback em sucesso, erro e bloqueio.
- Se SQLite falhar, a interação é gravada e sincronizada em JSONL, configurável por
  `ASSISTANT_AUDIT_FALLBACK` (padrão `backend/data/audit_fallback.jsonl`). Se ambos os
  destinos falharem, a exceção é propagada: não se anuncia persistência inexistente.
  Esse arquivo contém o mesmo conteúdo sensível da auditoria e deve ter acesso restrito.
- A flag de revisão humana indica necessidade de revisão; não implementa aprovação
  autenticada por um clínico. Regex permanece uma defesa parcial, com limitações de linguagem.

Verificação: `.venv-review/Scripts/python.exe -m pytest tests/assistant -q`.
A suíte de `tests/assistant` roda sem bibliotecas de treinamento (todos os providers de
LLM são mockados). `torch`/`transformers`/`peft`/`accelerate`/`datasets` foram instalados
depois nesse mesmo ambiente, à parte, para os testes manuais com o modelo real (`item1_only`
via API, sem mock) e para o fine-tuning com `context_examples.py` — não são dependência da
suíte de testes automatizada.


## Validação de qualidade

`quality.validate_answer_quality` bloqueia ausência de referência recuperada, linhas
repetidas três vezes, cópia de campos JSON e linguagem de confirmação por exames.
Perguntas explícitas sobre exames pendentes devem mencionar o identificador do exame.
Essas regras não comprovam sustentação semântica de toda afirmação e podem rejeitar
sinônimos: o prompt solicita os IDs exatos. Falhas são registradas em output_validation.
Não são um substituto de avaliação clínica nem garantia de detecção de toda alucinação.
