# Módulo de Fine-Tuning de LLM (Item 1 - Fase 3)

Este diretório centraliza o Item 1 do Tech Challenge FIAP Fase 3: o pipeline de preparação de dados e fine-tuning de um modelo LLM (LLaMA) para suportar o assistente médico virtual do hospital.

## Objetivo

Aplicar fine-tuning em um LLM da família LLaMA utilizando dados médicos com as seguintes etapas obrigatórias:
- preprocessing;
- anonimização;
- curadoria;
- formatação no template instruction-style do LLaMA;
- fine-tuning com LoRA / PEFT;
- avaliação comparativa entre modelo base e modelo ajustado.

## Estrutura

- `notebooks/fine_tuning_llm_medico.ipynb`
  - Notebook principal com o pipeline completo de fine-tuning de LLM.
- `data/raw/TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml`
  - Dataset médico de perguntas e respostas clínicas utilizado como fonte proxy para dados internos do hospital.
- `evaluation/figures/`
  - Gráficos de EDA gerados durante a análise exploratória dos dados curados.

## Fluxo Implementado

1. Parsing do dataset XML e extração de perguntas, resumos e respostas de referência.
2. Limpeza e normalização textual (HTML, espaços, caracteres de controle).
3. Anonimização de PII / dados sensíveis (URLs, e-mails, telefones, códigos de medicamentos, datas e menções a médicos).
4. Curadoria (remoção de respostas curtas, duplicatas e amostras não informativas).
5. Análise exploratória (distribuição de tamanho de textos e categorias clínicas).
6. Formatação dos pares no template LLaMA ChatML com `system`, `user` e `assistant`.
7. Divisão em conjuntos de treino, validação e teste.
8. Configuração de LoRA + PEFT sobre um modelo base LLaMA.
9. Execução do fine-tuning via Hugging Face `Trainer`.
10. Avaliação comparativa no conjunto de teste com métricas de overlap textuais (ROUGE-like / unigramas).

## Contrato de Integração com o Item 2

O Item 2 (assistente médico com LangChain) consome os artefatos e as premissas definidas aqui através do seguinte contrato:

- Fonte de conhecimento inicial: dataset médico question-answering processado no notebook.
- Style das respostas: template instrucional clínico com system prompt médico.
- Limites:
  - não substitui avaliação médica;
  - não emite diagnóstico definitivo;
  - não prescreve sem validação humana.
- Local esperado dos artefatos produtivos (quando exportados):
  - `artifacts/llama_medical_lora_model/`
  - `artifacts/tokenizer/`
- Formato de entrada esperado pelo LLM:
  - `system` -> persona médica e restrições;
  - `user` -> pergunta médica + contexto do paciente + fontes recuperadas.
- Formato de saída esperado pelo LLM:
  - texto clínico objetivo, contendo justificativa e referências utilizadas quando disponível.

## Premissas de Entrega

Como dados reais do hospital não estão disponíveis neste repositório, o dataset médico público citado acima atua como:
- proxy estilizado de FAQs médicas;
- amostra proxy para protocolos e condutas clínicas textualizadas;
- base curada para validação do pipeline de fine-tuning e integração posterior com LangChain.

## Como Reproduzir

1. Abra o notebook `notebooks/fine_tuning_llm_medico.ipynb`.
2. Instale as dependências necessárias de fine-tuning (`transformers`, `peft`, `datasets`, `accelerate`, `torch`).
3. Ajuste o caminho do dataset caso rode o notebook a partir de um `cwd` diferente:
   - caminho relativo esperado: `../data/raw/TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml`
   - caminho absoluto: `llm_finetuning/data/raw/TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml`
4. Execute as células de parsing, pré-processamento, anonimização, curadoria, EDA e formatação instruction.
5. Execute a célula de fine-tuning LoRA.
6. Rode a célula de avaliação comparativa para gerar respostas e métricas no conjunto de teste.

## Artefatos Esperados Após Execução Completa

- adaptador LoRA salvo em `artifacts/llama_medical_lora_model/`
- tokenizer salvo em `artifacts/tokenizer/`
- dataset formatado exportado em `data/processed/`
- gráficos / relatórios auxiliares em `evaluation/figures/`

## Observações Operacionais

- O notebook foi projetado para rodar preferencialmente com GPU.
- Em CPU, o fine-tuning é demonstrativo e visa documentar o pipeline, não garantir performance produtiva.
- Este módulo é mantido congelado durante a construção do Item 2, salvo ajustes pontuais de path/documentação.
