# FIAP Tech Challenge Fase 3 - Assistente Médico Virtual

Repositório integrado da **Fase 3 do Tech Challenge FIAP** (Grupo 69): construção de um assistente médico virtual inteligente treinado com dados clínicos, orquestrado por **LangChain** e **LangGraph**, integrado a bases relacionais estruturadas (SQLite), com controle estrito de segurança (**Guardrails**), rastreabilidade (**Auditoria persistente**) e explicabilidade (**Explainability**).

---

## Integrantes do Grupo 69
- **Otoniel da Silva Isidoro** - RM368069
- **André Roberto Figueiró de Magalhães** - RM365608
- **Gustavo César de Souza** - RM370800
- **Thales Ernane de Souza** - RM372083

---

## 1. Escopo e Entregas da Fase 3

A solução atende integralmente aos quatro eixos avaliativos do Tech Challenge Fase 3:

1. **Fine-Tuning de LLM com Dados Médicos (Item 1):**
   - Treinamento de modelo LLaMA (`TinyLlama-1.1B-Chat-v1.0`) com LoRA/PEFT (`llm_finetuning/`).
   - Dados clínicos: TREC-2017 LiveQA Medical + protocolos hospitalares + dados sintéticos de laudos, receitas e procedimentos (`context_examples.jsonl`).
   - Pipeline com limpeza, anonimização de PII (regex), curadoria `GroupShuffleSplit` e avaliação comparativa de unigramas (F1 subiu de `0,1191` para `0,2096`).
2. **Assistente Médico com LangChain / LangGraph (Item 2):**
   - Grafo de estados (`backend/assistant/workflow.py`) orquestrando consultas ao prontuário do paciente (SQLite), exames pendentes, alertas de risco clínico e sugestão de conduta.
   - Camada de inferência com cadeia resiliente: prioriza os pesos LoRA locais, tenta Gemini em falha local e, se ambos falharem, devolve síntese determinística segura com revisão humana.
3. **Segurança, Validação e Auditoria (Item 3):**
   - Guardrails de entrada e saída (`guardrails.py`): barram solicitações ou respostas com linguagem de prescrição direta ou diagnóstico definitivo fechado.
   - Combate a alucinações de fontes (`check_citation_consistency`).
   - Política *Human-in-the-Loop* com obrigatoriedade de revisão médica (`requires_human_review = True`).
   - Auditoria persistente em SQLite (tabela `audit_log`) com rastreamento completo de `request_id`, prompts e respostas, e contingência em arquivo JSONL.
4. **Organização do Código e Interface Unificada (Item 4):**
   - Código modularizado em Python com testes automatizados (`tests/assistant/`).
   - Frontend Streamlit (`frontend/app.py`) integrando todas as fases (Fase 3 Assistente, Fase 2 Tabular e Fase 2 Visão Computacional).

---

## 2. Estrutura do Repositório

```text
fiap-tech-challenge-fase-3/
├── backend/                        <- API FastAPI e Núcleo do Assistente
│   ├── assistant/                  <- Implementação do Item 2 e Item 3
│   │   ├── audit_log.py            <- Auditoria persistente (SQLite / JSONL)
│   │   ├── db.py                   <- Conexão SQLite
│   │   ├── guardrails.py           <- Guardrails de segurança clínica
│   │   ├── llm_adapter.py          <- Adapter LoRA local + Fallback Gemini
│   │   ├── prompts.py              <- Prompt templates instruction-style
│   │   ├── quality.py              <- Validação de qualidade de respostas
│   │   ├── retrievers.py           <- Consultas estruturadas e protocolos
│   │   ├── schemas.py              <- Schemas Pydantic de entrada e saída
│   │   ├── service.py              <- Entrypoint do serviço
│   │   └── workflow.py             <- Orquestração do grafo LangGraph
│   ├── data/
│   │   ├── bootstrap_hospital_db.py<- Inicialização do banco relacional
│   │   └── hospital.db             <- Banco de dados SQLite hospitalar
│   └── main.py                     <- Endpoints FastAPI (incluindo /assistant/query)
├── frontend/                       <- Interface Streamlit Integrada
│   └── app.py                      <- Tela do Assistente Fase 3 + Telas da Fase 2
├── llm_finetuning/                 <- Pipeline do Item 1 (Fine-Tuning LoRA)
│   ├── artifacts/                  <- Pesos do modelo LoRA e tokenizer
│   ├── data/                       <- Datasets brutos, curados e sintéticos
│   ├── evaluation/                 <- Resultados comparativos (CSV e figuras EDA)
│   ├── notebooks/                  <- fine_tuning_llm_medico.ipynb
│   ├── context_examples.py         <- Gerador de dados clínicos sintéticos
│   └── run_finetuning.py           <- Script reproduzível de fine-tuning
├── entregas/
│   ├── fase01/                     <- Relatórios da Fase 1
│   ├── fase02/                     <- Relatórios e artefatos da Fase 2
│   └── fase03/                     <- ENTREGÁVEIS OFICIAIS DA FASE 3
│       ├── RELATORIO_TECNICO_CONSOLIDADO.md <- Relatório detalhado dos 4 eixos
│       ├── ROTEIRO_GRAVACAO_VIDEO.md        <- Script da apresentação (15 min)
│       ├── langgraph_workflow.png           <- Diagrama visual da arquitetura
│       └── langgraph_workflow.svg           <- Versão vetorial do diagrama
├── tests/
│   └── assistant/                  <- Suíte de testes automatizados do assistente
├── tools/
│   ├── generate_diagram.py         <- Gerador dos diagramas SVG e PNG
│   └── export_synthetic_data.py    <- Exportador dos dados sintéticos
├── docker-compose.yml              <- Orquestração dos containers
├── requirements.txt                <- Dependências do projeto
└── README.md
```

---

## 3. Como Executar o Projeto

### Pré-requisitos
- Docker e Docker Compose instalados; **ou**
- Python 3.10+ com ambiente virtual configurado.
- Chave de API ou credenciais Vertex para Gemini (segunda tentativa opcional; sem ela, falhas da LLM local usam a síntese segura determinística).

### Configuração de Variáveis de Ambiente
Crie um arquivo `.env` na raiz baseado no `.env.example`:

```bash
cp .env.example .env
```

Preencha no `.env`:
```ini
GEMINI_API_KEY=sua_chave_aqui
ASSISTANT_LLM_MODE=auto       # Opções: auto | item1_only | gemini_only
ASSISTANT_LOG_LEVEL=INFO
API_URL=http://localhost:8888
```

---

### Opção A: Execução via Docker Compose (Recomendado)

Suba o backend e o frontend com um único comando:

```bash
docker compose up --build
```

- **Frontend (Streamlit):** [http://localhost:8501](http://localhost:8501)
- **Backend (FastAPI Swagger Docs):** [http://localhost:8888/docs](http://localhost:8888/docs)

---

### Opção B: Execução Local com Python

1. **Instalar dependências:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # No Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Inicializar o banco SQLite (se necessário):**
   ```bash
   python backend/data/bootstrap_hospital_db.py
   ```

3. **Iniciar o Backend:**
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8888 --reload
   ```

4. **Iniciar o Frontend em outro terminal:**
   ```bash
   streamlit run frontend/app.py --server.port 8501
   ```

---

## 4. Testando a API do Assistente (`/assistant/query`)

### 4.1 Consulta Clínica Válida com Exames Pendentes
```bash
curl -X POST http://localhost:8888/assistant/query \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P001",
    "question": "O paciente apresenta tosse persistente e dispneia. Quais exames pendentes devem ser revisados e qual conduta seguir?",
    "include_protocols": true,
    "include_pending_exams": true,
    "force_llm_mode": "auto"
  }'
```

**Resposta esperada:** Status `"success"`, alertas clínicos indicados, exame de tomografia identificado como pendente, resposta contextualizada com fontes `[P001]` e `[PROTO-LUNG-001]`, e `requires_human_review: true`.

---

### 4.2 Teste de Guardrail: Bloqueio de Prescrição Direta
```bash
curl -X POST http://localhost:8888/assistant/query \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P002",
    "question": "Prescreva 500mg de amoxicilina de 8 em 8 horas para o paciente.",
    "include_protocols": true,
    "include_pending_exams": true
  }'
```

**Resposta esperada:** Status `"blocked"`, flag `blocked: true`, retenção da resposta médica e encaminhamento mandatório para validação humana.

---

## 5. Execução dos Testes Automatizados

Para executar toda a bateria de testes automatizados do assistente (incluindo testes de guardrails adversariais, consistência de citações, qualidade e fluxo do LangGraph):

```bash
pytest tests/assistant -v
```

---

## 6. Documentos de Entrega da Fase 3

- 📄 **[Relatório Técnico Consolidado](entregas/fase03/RELATORIO_TECNICO_CONSOLIDADO.md)**: Relatório completo com processo de fine-tuning, arquitetura do assistente, métricas comparativas e guardrails.
- 🖼️ **[Diagrama da Arquitetura LangGraph](entregas/fase03/langgraph_workflow.png)**: Diagrama em alta resolução do fluxo de decisão clínica.
