# Dados da Fase 4

## Fonte

- Dataset: [REHAB24-6: A multi-modal dataset of physical rehabilitation exercises](https://zenodo.org/records/13305826)
- DOI: [10.5281/zenodo.13305826](https://doi.org/10.5281/zenodo.13305826)
- Versao: v1
- Uso permitido pelo registro: pesquisa academica ou de organizacoes sem fins
  lucrativos, para fins nao comerciais. Consulte o registro e seus termos antes
  de reutilizar ou redistribuir os dados.

## Arquivos

O pacote completo publicado no Zenodo tem aproximadamente 5,7 GB:

| Arquivo | Uso previsto | Tamanho aproximado |
|---|---|---:|
| `videos.zip` | Videos RGB de duas cameras para o pipeline visual | 2,7 GB |
| `Segmentation.csv` | Limites das repeticoes, rotulos de correcao e condicoes | 53 kB |
| `Segmentation.txt` | Definicao e orientacao das anotacoes | 1,7 kB |
| `2d_joints.zip`, `3d_joints.zip` | Articulacoes de referencia para pose e movimento | 1,2 GB |
| `2d_markers.zip`, `3d_markers.zip` | Marcadores de captura de movimento | 1,8 GB |
| `joints_names.txt`, `marker_names.txt` | Nomes das articulacoes e marcadores | menos de 1 kB |

O download local fica em `entregas/fase04/data/raw/rehab24-6/`. Os arquivos sao
ignorados pelo Git devido ao tamanho e aos termos de distribuicao.

## Downloads

Com Python 3.9 ou superior, baixe todos os arquivos e valide seus checksums MD5
contra os metadados oficiais do Zenodo:

```powershell
py -3.11 entregas\fase04\download_dataset.py
```

O comando baixa aproximadamente 5,7 GB para
`entregas\fase04\data\raw\rehab24-6`. Arquivos ja existentes so sao reutilizados
quando tamanho e checksum coincidem. Nao e necessario colocar credenciais ou
tokens no repositorio.

Para baixar somente as anotacoes pequenas antes dos videos:

```powershell
py -3.11 entregas\fase04\download_dataset.py --only Segmentation.csv --only Segmentation.txt --only joints_names.txt --only marker_names.txt
```

### Estado neste workspace

`Segmentation.csv`, `Segmentation.txt`, `joints_names.txt` e
`marker_names.txt` estao presentes. Os videos estao extraidos em `Ex1` a `Ex6`:
130 MP4s, correspondentes a 65 gravacoes nas cameras 17 e 18. O pipeline de
treinamento usa os 65 videos da Camera17. Os arquivos ZIP nao precisam estar
presentes para treinar se os videos ja estiverem extraidos. Os ZIPs de
articulacoes e marcadores nao estao disponiveis localmente; por isso, a
comparacao com uma referencia de pose nao faz parte do experimento atual.

## Citacao

Černek, A., Sedmidubsky, J., Budikova, P. *REHAB24-6: Physical Therapy Dataset
for Analyzing Pose Estimation Methods*. 17th International Conference on
Similarity Search and Applications (SISAP), Springer, 2024. Cite tambem o
registro Zenodo ao publicar resultados que usem os dados.
