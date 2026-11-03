# Relatorio Tecnico - Fase 4
## Analise de Videos de Reabilitacao com Estimativa de Pose

**FIAP - Inteligencia Artificial para Desenvolvedores**  
**Grupo 69**

> O classificador e um prototipo experimental para pesquisa/ensino, nao uma
> ferramenta clinica. Resultados devem ser interpretados dentro das limitacoes
> do dataset e da separacao por participantes descritas abaixo.

## 1. Objetivo

Implementar um fluxo para analisar videos de exercicios de reabilitacao,
estimar postura/movimento e classificar repeticoes como corretas ou incorretas.
A API e a interface apresentam o resultado como experimental, sem emitir
diagnosticos ou substituir a avaliacao de um fisioterapeuta.

## 2. Dataset selecionado

O REHAB24-6, publicado no [Zenodo](https://zenodo.org/records/13305826), e
diretamente relacionado ao objetivo: disponibiliza videos RGB de duas cameras,
sequencias de articulacoes 2D/3D e marcadores de captura de movimento. Suas
anotacoes incluem 1.072 repeticoes, intervalos por frame, rotulos binarios de
execucao correta/incorreta, direcao do exercicio e condicoes de iluminacao.

O conjunto cobre seis exercicios: abducao de braco, movimento de bracos em V/W,
flexoes apoiadas em mesa, abducao de perna, afundo e agachamento. Os videos
foram gravados a 30 FPS; as anotacoes de segmentacao sao baseadas nessa taxa.
Participaram dez voluntarios saudaveis, com 65 gravacoes no total.

### Adequacao e limites

- **Videos de exercicios:** adequado para prototipar o fluxo de analise de
  movimento em fisioterapia; os participantes sao voluntarios saudaveis.
- **Pose e movimento:** as articulacoes 2D/3D permitem avaliar estimadores como
  OpenPose e comparar resultados com uma referencia de movimento.
- **Repeticoes incorretas:** ha rotulo correto/incorreto por repeticao, mas a
  descricao do dataset nao garante anotacao detalhada da categoria ou causa de
  cada erro.
- **YOLOv8:** o dataset pode servir como entrada visual, mas a documentacao nao
  informa caixas delimitadoras de objetos/areas criticas; nao e suficiente, por
  si so, para treinar e avaliar detector supervisionado de objetos.
- **Generalizacao clinica:** participantes saudaveis e ambiente controlado nao
  representam pacientes em tratamento; os resultados nao devem ser extrapolados
  para pacientes em recuperacao.
- **Identidade e licenca:** uso restrito a pesquisa academica ou nao comercial,
  conforme o registro Zenodo; nao redistribuir os arquivos e seguir a citacao
  indicada.

## 3. Organizacao no repositorio

```text
entregas/fase04/
├── data/
│   ├── README.md
│   └── raw/rehab24-6/       <- arquivos Zenodo, excluidos do Git
├── src/train_classifier.py  <- treinamento e avaliacao
├── results/                 <- relatorio de metricas gerado pelo treino
├── download_dataset.py      <- download e verificacao de checksums
├── README.md
└── RELATORIO_TECNICO_CONSOLIDADO.md
```

O pipeline reutilizavel esta em `backend/rehab_video/`; a rota de upload esta
em `backend/main.py` e a tela Streamlit em `frontend/app.py`. O modelo treinado
fica em `backend/model/rehab_pose_classifier.joblib`; videos e pesos YOLO sao
artefatos locais excluidos do Git.

## 4. Fluxo experimental implementado

1. Validar `Segmentation.csv` e os videos Camera17 a 30 FPS. O CSV usa intervalos
   inclusivos e numeracao de frames a partir de 1; a leitura converte para
   indices zero-based do OpenCV. Uma anotacao termina um frame apos o video e e
   limitada ao ultimo frame disponivel.
2. Separar participantes em treino, validacao e teste, sem sobreposicao entre
   particoes.
3. Amostrar 12 frames de cada repeticao, estimar 17 pontos de pose COCO com
   YOLOv8n-pose pre-treinado e derivar caracteristicas temporais normalizadas.
4. Treinar regressao logistica balanceada e comparar no conjunto de teste com
   um baseline de classe majoritaria (`DummyClassifier`).
5. Salvar metricas agregadas, por classe e por exercicio em
   `results/training_report.json`.
6. Expor inferencia via `POST /analyze-rehabilitation-video` e pela tela
   Streamlit. Upload limitado a 250 MiB por padrao; a API aceita MP4 e frame
   range zero-based.

Como os ZIPs de articulacoes 2D/3D nao estao disponiveis localmente, nao foi
possivel comparar a pose estimada com ground truth. O baseline e somente a
classe majoritaria, nao uma referencia de pose.

## 5. Estado da entrega

- [x] Definicao do escopo de video de reabilitacao.
- [x] Selecao e avaliacao de adequacao do REHAB24-6.
- [x] Dados: anotacoes e 130 videos extraidos em `Ex1` a `Ex6`.
- [x] Pipeline de pose, treinamento, avaliacao e testes automatizados.
- [x] Integracao com API FastAPI e interface Streamlit.
- [x] Treinamento integral e consolidacao das metricas no conjunto de teste.
- [x] Smoke test de inferencia real pela rota FastAPI.
- [ ] Comparacao com ground truth de pose: pendente dos ZIPs de articulacoes.

Os 11 testes automatizados da Fase 4 passaram. O experimento utilizou os 65
videos da Camera17, com 1.072 repeticoes anotadas. A divisao por participante
resultou em 600 repeticoes para treino (participantes 3, 4, 5, 7, 8 e 10), 250
para validacao (1 e 9) e 222 para teste (2 e 6), sem participantes compartilhados.
O ambiente nao tinha CUDA; a extracao completa foi executada na CPU.

## 6. Resultados experimentais

| Conjunto | Acuracia | F1 macro | Repeticoes |
|---|---:|---:|---:|
| Validacao | 63,20% | 62,23% | 250 |
| Teste | 59,46% | 57,80% | 222 |
| Baseline majoritario (teste) | 43,69% | 30,41% | 222 |

No teste, a matriz de confusao (linhas reais e colunas previstas na ordem
incorreta/correta) foi `[[44, 53], [37, 88]]`. Para repeticoes incorretas, a
precisao foi 54,32%, o recall 45,36% e o F1 49,44%; para corretas, 62,41%,
70,40% e 66,17%, respectivamente. O resultado supera o baseline simples, mas
nao e suficiente para uso clinico.

| Exercicio | Repeticoes de teste | Acuracia | F1 macro |
|---|---:|---:|---:|
| 1 - abducao de braco | 30 | 70,00% | 69,97% |
| 2 - movimentos V/W | 46 | 52,17% | 52,08% |
| 3 - flexao apoiada | 18 | 55,56% | 55,56% |
| 4 - abducao de perna | 45 | 62,22% | 47,20% |
| 5 - afundo | 40 | 42,50% | 39,43% |
| 6 - agachamento | 43 | 74,42% | 71,39% |

Os resultados variam por exercicio; em particular, o recall de repeticoes
incorretas e baixo no exercicio 4 e o desempenho geral e fraco no exercicio 5.
As metricas completas estao em
[`results/training_report.json`](results/training_report.json).

## 7. Referencia

Černek, A., Sedmidubsky, J., Budikova, P. *REHAB24-6: Physical Therapy Dataset
for Analyzing Pose Estimation Methods*. SISAP, Springer, 2024.
[Registro e arquivos do dataset](https://zenodo.org/records/13305826).
