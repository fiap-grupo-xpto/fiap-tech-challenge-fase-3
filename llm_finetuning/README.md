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
  - Notebook de referência/exploração com o pipeline completo de fine-tuning de LLM.
- `run_finetuning.py`
  - **Script que efetivamente gera os artefatos de produção.** Existe porque a execução
    headless do notebook via `jupyter nbconvert --execute` falha no Windows com "Kernel
    died" já na primeira célula (incompatibilidade conhecida entre o event loop Proactor
    do asyncio no Windows e o pyzmq usado pelo kernel do Jupyter — não é um problema do
    código de fine-tuning). Este script roda exatamente a mesma lógica das células do
    notebook, fora do kernel Jupyter, e é a forma confiável de reproduzir o treino no
    Windows. Em Linux/macOS ou no Google Colab, o notebook também deve funcionar
    normalmente célula a célula.
- `data/raw/TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml`
  - Dataset médico de perguntas e respostas clínicas utilizado como fonte proxy para dados internos do hospital.
- `evaluation/figures/`
  - Gráficos de EDA gerados durante a análise exploratória dos dados curados.
- `evaluation/comparative_evaluation_results.csv`
  - Resultado da avaliação comparativa real (modelo base vs. modelo fine-tuned) no
    conjunto de teste, gerado por `run_finetuning.py`.

## Fluxo Implementado

1. Parsing do dataset XML e extração de perguntas, resumos e respostas de referência.
2. Limpeza e normalização textual (HTML, espaços, caracteres de controle).
3. Anonimização de PII / dados sensíveis (URLs, e-mails, telefones, códigos de medicamentos, datas e menções a médicos).
4. Curadoria (remoção de respostas curtas, duplicatas e amostras não informativas).
5. Análise exploratória (distribuição de tamanho de textos e categorias clínicas).
6. Formatação dos pares usando o **chat template nativo do tokenizer do modelo base**
   (`tokenizer.apply_chat_template`) — não tags manuais fixas. Isso garante que o texto
   de treino usa exatamente o mesmo formato que a inferência em produção usa
   (`backend/assistant/llm_adapter.py`); uma versão anterior formatava com tags estilo
   Llama-3 que não existem no vocabulário do tokenizer do TinyLlama, criando um
   descompasso entre treino e inferência.
7. Divisão em treino/validação/teste **agrupada por `qid`** (`GroupShuffleSplit`): várias
   linhas do XML compartilham o mesmo `qid` (múltiplas respostas de referência para a
   mesma pergunta original); um split aleatório simples podia colocar a mesma pergunta
   em treino e em teste ao mesmo tempo, inflando artificialmente as métricas.
8. Configuração de LoRA + PEFT sobre o modelo base (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`,
   arquitetura Llama).
9. Execução real do fine-tuning via Hugging Face `Trainer` (`trainer.train()`), com
   padding dinâmico por batch via `DataCollatorForLanguageModeling`.
10. Avaliação comparativa real no conjunto de teste — modelo **base** (adapter LoRA
    desativado via `model.disable_adapter()`) vs. modelo **fine-tuned** (mesmo objeto de
    modelo), com métricas de overlap textual (precisão/revocação/F1 de unigramas).

O dataset de treino agora inclui três fontes:
- 166 exemplos curados do TREC-2017-LiveQA (QA médico público, proxy);
- **10 exemplos sintéticos baseados nos 5 protocolos hospitalares internos**
  (os mesmos usados em runtime por `backend/data/bootstrap_hospital_db.py`), fraseados
  como dúvidas de médicos sobre a conduta de cada protocolo;
- **60 exemplos sintéticos de `context_examples.py`**, gerados com a MESMA função
  `build_user_prompt` que a API usa em produção (30 pacientes fictícios × 2 perguntas
  PT/EN cada), ensinando o modelo a responder no formato real de produção — citando o
  paciente pelo ID, sem inventar resultado de exame.

Isso endereça parcialmente o requisito do desafio de treinar também sobre "protocolos
médicos do hospital" e "dúvidas de médicos", não só QA público de pacientes, e ataca
diretamente o problema (descrito abaixo) de o modelo se confundir com o prompt real de
produção. O dataset processado final (train/validation/test, já no formato de chat
template) é exportado em `data/processed/`.

### Execução mais recente (`run_finetuning.py`, CPU, 1 época, dataset ampliado com contexto de produção)

| Métrica | Modelo base | Modelo fine-tuned |
|---|---|---|
| F1 médio (unigramas vs. referência) | 0,1191 | **0,2096** |
| Precisão média | 0,1501 | **0,2963** |
| Revocação média | 0,1126 | **0,1813** |

Loss final de treino: 1,4950 (1 época, 181 amostras de treino — 233 curadas/ampliadas no
total, 3 excluídas do TREC por exceder `MAX_SEQ_LENGTH=1024` tokens sem truncar a
resposta; `TinyLlama-1.1B-Chat-v1.0` com LoRA — 4.505.600 parâmetros treináveis, 0,41% do
total). Resultado completo, por pergunta, em `evaluation/comparative_evaluation_results.csv`.

`MAX_SEQ_LENGTH=1024` (configurável via `FINETUNE_MAX_SEQ_LENGTH`) foi escolhido medindo
a distribuição real de tokens: os 60 exemplos de `context_examples.py` ficam todos entre
760-772 tokens (nenhum excluído nesse limite), os 10 de protocolo entre 177-220 (nenhum
excluído), e só 3 dos 166 do TREC excedem 1024 — ou seja, preserva quase todo o dataset
pela metade do custo de memória/tempo de um limite de 2048.

**Resultado do teste end-to-end via API real, em `item1_only`, sem nenhum mock** (a mesma
pergunta que expôs o problema original de o modelo se confundir com o prompt de
produção):
- O modelo **não alucina mais o JSON do paciente** (o achado original: "Persistência de
  cachorro... confirmado por exame de pulmão" repetido) — mas passou a exibir uma forma
  diferente do mesmo problema de fundo: ele às vezes ecoa o **cabeçalho de instruções de
  formato** do prompt ("Resumo:\nContexto do paciente:\nConduta sugerida:\n...\nRegras:\n-
  Não diga que há diagnóstico confirmado...") sem preencher nenhum campo com conteúdo real.
- Essa resposta vazia/eco é corretamente **bloqueada** por `quality.py`
  (`quality:pending_exam_not_answered`) antes de chegar ao usuário — o sistema de
  guardrails do Ponto 3 funciona exatamente como projetado mesmo quando o modelo do
  Ponto 1 ainda não é bom o suficiente.
- **Conclusão honesta:** o pipeline de ponta a ponta (template, treino real, avaliação
  comparativa real) está correto e a métrica melhorou, mas 1 época com ~180 exemplos de
  treino em um modelo de 1,1B parâmetros ainda não é suficiente para seguir de forma
  confiável um prompt de produção longo e estruturado. Mais épocas (`FINETUNE_EPOCHS`),
  mais exemplos de contexto, ou um modelo base maior tendem a fechar essa lacuna — mas
  isso está fora do orçamento de tempo/hardware (CPU, 16 GB RAM) desta execução. Na
  prática, o Gemini de fallback continua sendo o caminho que produz respostas úteis hoje.

**Limitações honestas que permanecem:**
- Rodada em CPU (sem GPU disponível no ambiente), com apenas **1 época** — reduzida de 3
  para viabilizar o tempo de execução (~55 min mesmo já otimizado). Um treino mais longo
  em GPU (Colab) tende a melhorar os resultados.
- O grosso do dataset (166 de 233 exemplos) ainda é o TREC-2017-LiveQA, QA público de
  pacientes em inglês — os 70 exemplos sintéticos (protocolo + contexto) não substituem
  dados reais de laudos/receitas/procedimentos internos, que o desafio também pede e não
  estão disponíveis neste repositório.

**Notas de ambiente (Windows/CPU), para quem for reproduzir:**
- Em uma execução anterior, o treino sofreu um *segmentation fault* nativo (sinal 139) no
  primeiro passo do `Trainer` — conflito comum de threads entre BLAS/OMP e o PyTorch em
  CPU no Windows, não um problema do código. Limitar `OMP_NUM_THREADS`/`MKL_NUM_THREADS`
  evita o crash, mas **`=1` deixa o treino extremamente lento** (~300s/passo, quase 2h no
  total) por desperdiçar os outros núcleos da CPU — um valor intermediário (`=4`, testado
  nesta máquina de 6 núcleos/12 threads) evita o crash e mantém o treino em ~55min.
- Processos de treino anteriores que travam ou são interrompidos **não liberam a memória
  sozinhos no Windows** — um processo de uma tentativa anterior ficou "zumbi" segurando
  ~4GB de RAM por horas depois de eu achar que tinha sido interrompido, o que causou uma
  segunda falha de memória em uma tentativa seguinte. Confira `tasklist`/Gerenciador de
  Tarefas por processos `python.exe` órfãos antes de assumir que falta memória de verdade.

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

**Opção recomendada (Windows, ou qualquer SO, via terminal):**

```bash
pip install transformers peft datasets accelerate torch scikit-learn pandas numpy matplotlib seaborn tqdm
# No Windows, evita um segfault conhecido por conflito de threads BLAS/OMP em CPU.
# Use um valor > 1 (ex.: metade dos núcleos físicos) — "=1" evita o crash mas deixa o
# treino ~2x mais lento por não usar o resto da CPU.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python llm_finetuning/run_finetuning.py
```

Variável de ambiente opcional: `FINETUNE_EPOCHS` (padrão `1`; aumente para um treino
mais completo, especialmente com GPU disponível).

**Opção alternativa (notebook, Linux/macOS/Colab):**

1. Abra o notebook `notebooks/fine_tuning_llm_medico.ipynb`.
2. Instale as mesmas dependências acima.
3. Ajuste o caminho do dataset caso rode o notebook a partir de um `cwd` diferente:
   - caminho relativo esperado: `../data/raw/TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml`
   - caminho absoluto: `llm_finetuning/data/raw/TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml`
4. Execute as células em ordem, célula a célula (execução headless via
   `jupyter nbconvert --execute` é conhecida por falhar no Windows — ver nota abaixo).

## Artefatos Esperados Após Execução Completa

- adaptador LoRA salvo em `artifacts/llama_medical_lora_model/` (junto com
  `training_metadata.json`: época(s), tamanho dos splits e loss final de treino)
- tokenizer salvo em `artifacts/tokenizer/`
- resultados da avaliação comparativa em `evaluation/comparative_evaluation_results.csv`
- gráficos / relatórios auxiliares em `evaluation/figures/`

## Observações Operacionais

- O notebook/script funciona tanto em CPU quanto em GPU; com GPU, aumente
  `FINETUNE_EPOCHS` para um treino mais completo (o valor padrão foi reduzido para
  viabilizar execução em CPU).
- **Execução headless do notebook no Windows:** `jupyter nbconvert --execute` falha com
  "Kernel died" logo na primeira célula, por uma incompatibilidade conhecida entre o
  event loop Proactor do asyncio no Windows e o pyzmq do kernel Jupyter — não é um
  problema do código deste módulo. Use `run_finetuning.py` no Windows; o notebook
  continua funcionando normalmente quando executado célula a célula dentro do Jupyter
  (não via nbconvert) ou em Linux/macOS/Colab.
- Este módulo não está mais "congelado" no sentido de nunca ter sido executado: já foi
  rodado de ponta a ponta com resultados reais (ver seção acima); pode ser executado
  novamente a qualquer momento para gerar um novo adapter (ex.: com mais épocas, em GPU,
  ou com dataset ampliado).


## Contexto de produção no treino (`context_examples.py`)

`context_examples.py` gera 60 exemplos sintéticos operacionais, com 30 pacientes e
variantes PT/EN agrupadas pelo mesmo qid. Usa `build_user_prompt` da API, inclui o
exame pendente e ensina a citar o paciente sem inventar resultado. Os exemplos estão
em `data/context_examples.jsonl`. Não são protocolos clínicos validados.
`run_finetuning.py` incorpora esses exemplos e usa o system prompt da API.
O notebook antigo é referência histórica; execute o script para essa versão.

**Status: já incorporado, treinado e avaliado** (ver "Execução mais recente" acima) — o
adaptador e as métricas em `artifacts/`/`evaluation/` já refletem esse treino, incluindo
os 60 exemplos de contexto. Duas tentativas anteriores de rodar esse treino falharam
(erro nativo de memória, depois um treino tão lento que parecia travado) por usar
`MAX_SEQ_LENGTH=2048` sem necessidade — medir a distribuição real de tokens mostrou que
1024 é suficiente (ver acima) e resolveu ambos os problemas.
