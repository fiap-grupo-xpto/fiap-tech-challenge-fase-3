# FIAP Tech Challenge Fase 3 - Assistente Médico Virtual

Repositório integrado do Tech Challenge FIAP Fase 3: construção de um assistente médico virtual treinado com dados clínicos, orquestrado por LangChain e LangGraph, com controle de segurança, auditoria e logging detalhado.

## Escopo da Fase 3

O desafio é entregar uma solução que permita ao hospital:
- treinar uma LLM com dados clínicos;
- construir um assistente médico capaz de consultar bases de dados e contextualizar respostas;
- coordenar fluxos de decisão clínica automatizados e seguros;
- garantir limites de atuação, rastreabilidade e explicabilidade das respostas.

## Mapeamento dos Itens de Entrega

### 1. Fine-Tuning de LLM com Dados Médicos

**Local:** [llm_finetuning](file:///C:/Coding/fiap-tech-challenge-fase-3/llm_finetuning)

Conteúdo coberto:
- parsing e preprocessing de dataset médico;
- anonimização de PII;
- curadoria de pares pergunta/resposta;
- formatação instruction-style para LLaMA;
- fine-tuning com LoRA / PEFT;
- avaliação comparativa no conjunto de teste.

Arquivos principais:
- Notebook principal: [fine_tuning_llm_medico.ipynb](file:///C:/Coding/fiap-tech-challenge-fase-3/llm_finetuning/notebooks/fine_tuning_llm_medico.ipynb)
- Dataset bruto: [TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml](file:///C:/Coding/fiap-tech-challenge-fase-3/llm_finetuning/data/raw/TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml)
- Documentação do módulo: [llm_finetuning/README.md](file:///C:/Coding/fiap-tech-challenge-fase-3/llm_finetuning/README.md)

### 2. Assistente Médico com LangChain

**Local planejado:** backend (integrações LLM + RAG + LangChain + LangGraph)

Entregáveis esperados:
- integração entre a LLM customizada do Item 1 e o pipeline de assistente;
- consultas a fontes estruturadas (prontuários, registros, exames);
- contextualização das respostas com dados atualizados do paciente;
- coordenação de etapas por LangGraph (ex.: verificar exames pendentes, sugerir conduta, emitir alertas).

Referências atuais:
- Código existente da camada LLM: [backend/llm/interpreter.py](file:///C:/Coding/fiap-tech-challenge-fase-3/backend/llm/interpreter.py)
- Backend existente: [backend/main.py](file:///C:/Coding/fiap-tech-challenge-fase-3/backend/main.py)
- Documentação de API da fase anterior: [backend/API_DOCS.md](file:///C:/Coding/fiap-tech-challenge-fase-3/backend/API_DOCS.md)

### 3. Segurança e Validação

**Local previsto:** módulos de guardrails, logging, auditoria e explainability no backend e em camada dedicada de validação.

Entregáveis esperados:
- limites de atuação do assistente;
- proibição de prescrever ou concluir diagnóstico sem validação humana;
- logging estruturado para rastreio e auditoria;
- explicabilidade das respostas (fontes usadas, contexto recuperado, referências).

### 4. Organização do Código e README

Estrutura modular atualizada para a Fase 3:

```text
fiap-tech-challenge-fase-3/
├── llm_finetuning/          <- Item 1 congelado
│   ├── notebooks/
│   ├── data/raw/
│   ├── data/processed/
│   ├── artifacts/
│   └── evaluation/figures/
├── backend/                 <- Item 2 + Item 3 + API
│   └── llm/
├── frontend/                <- Interface
├── train_model/             <- Experimentos tabular / imagem de fases anteriores
├── avaliation/              <- Avaliação de qualidade de LLMs
├── terraform/               <- Infraestrutura em nuvem
├── tests/                   <- Testes automatizados
├── entregas/
│   ├── fase01/
│   ├── fase02/
│   └── fase03/
│       └── assets/
├── docs/                    <- Diagramas e documentação extra (a criar se necessário)
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Como Navegar por Fase

- **Fase 1:** `entregas/fase01/`
- **Fase 2:** `entregas/fase02/`
- **Fase 3:** `entregas/fase03/` + `llm_finetuning/` + evoluções em `backend/`

## Próximos Passos Recomendados

1. Consumir o contrato de integração do Item 1 descrito em [llm_finetuning/README.md](file:///C:/Coding/fiap-tech-challenge-fase-3/llm_finetuning/README.md).
2. Implementar o Item 2 em `backend/` usando LangChain/LangGraph, consumindo os artefatos/documentação de `llm_finetuning/`.
3. Implementar o Item 3 com guardrails, logging e explicação de fontes.
4. Consolidar relatório técnico, diagrama LangChain e resultados em `entregas/fase03/`.
