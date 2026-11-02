# Fase 4 - Analise de Videos de Reabilitacao

Esta pasta contem o fluxo experimental para estimar pose em videos de exercicios
de reabilitacao, classificar repeticoes como corretas/incorretas e produzir
metricas e relatorios. O dataset REHAB24-6 contem voluntarios saudaveis; o modelo
nao deve ser usado para decisao clinica.

## Estrutura

```text
entregas/fase04/
├── data/
│   ├── README.md
│   └── raw/
│       └── rehab24-6/   <- arquivos originais do Zenodo (nao versionados)
├── src/                 <- treinamento e avaliacao
├── results/             <- metricas do experimento
├── download_dataset.py  <- download com verificacao de checksum
├── README.md
└── RELATORIO_TECNICO_CONSOLIDADO.md
```

O codigo integrado da aplicacao esta em `backend/rehab_video/`, com a rota em
`backend/main.py` e a tela no `frontend/app.py`. O modelo treinado e o peso do
YOLO sao artefatos locais ignorados pelo Git. Os testes automatizados ficam em
`tests/fase04/` na raiz do repositorio.

## Dataset e uso

Veja [data/README.md](data/README.md) para a fonte, os arquivos previstos, a
licenca e o procedimento de verificacao. Os arquivos brutos sao grandes e estao
excluidos do Git.

## Treinar

Com Python 3.11, dependencias instaladas e os videos em
`data/raw/rehab24-6/Ex1` a `Ex6`, execute a partir da raiz do repositorio:

```powershell
py -3.11 entregas\fase04\src\train_classifier.py
```

O comando valida anotacoes e videos, estima pose com YOLOv8n-pose, separa os
participantes em treino/validacao/teste e salva o classificador e as metricas
em `backend/model/rehab_pose_classifier.joblib` e
`entregas/fase04/results/training_report.json`. A inferencia na API exige o
artefato treinado. O resultado medido e o baseline estao documentados no
[relatorio tecnico](RELATORIO_TECNICO_CONSOLIDADO.md). Para executar os testes:

```powershell
py -3.11 -m pytest tests\fase04 -q
```
