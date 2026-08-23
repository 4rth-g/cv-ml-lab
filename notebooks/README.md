# Notebooks

Exploração livre em R, sobre os artefatos que o pipeline exporta.

**Todo notebook aqui é consumidor.** Lê `results/` e não produz artefato nenhum.
Quem produz é sempre o `just` (`just train`, `just eda`). Se esta pasta inteira
desaparecer, nenhum número do relatório se perde. É essa regra, e não o formato
do arquivo, que resolve o problema de notebooks que não se comunicam entre si.

## Organização

Por **pergunta**, não por dataset nem por run. O dataset e o run são parâmetros
no topo de cada notebook. Com 14 datasets e 56 combinações possíveis, um notebook
por coisa viraria duplicação, que é o problema que essa estrutura evita.

| Notebook | Pergunta | Parâmetro |
|---|---|---|
| `eda/01-dataset` | O que há neste dataset, e acurácia é métrica honesta nele? | `DATASET` |
| `eda/02-dermamnist` | O mesmo, num caso de desbalanço extremo (58x) | `DATASET` |
| `eda/03-chestmnist` | Multi-label: 14 achados que coexistem por radiografia | `DATASET` |
| `eda/04-organmnist3d` | Volume 3D, com as fatias tratadas como canais | `DATASET` |
| `pos-hoc/01-metricas` | Quão bom é este run, por métrica que não seja acurácia? | `RUN` |
| `pos-hoc/02-erros` | Onde e por que este run erra? | `RUN` |

Para trocar de dataset ou de run, edite as variáveis da primeira célula. Não
crie um notebook novo por dataset.

## Uso

```bash
just nb          # JupyterLab com o kernel R
just nb-kernel   # registra o kernel para o VS Code (uma vez, por clone)
just nb-run      # reexecuta todos, regenerando as saídas
```

Formato: `.ipynb` apenas, versionado com as saídas. O GitHub renderiza a análise
para quem só abre o link. O pareamento com `.Rmd` foi removido: o VS Code lê mal
esse formato.

## Fronteira com o relatório

`analysis/report.qmd` é o entregável estável, gerado por comando e com formato
fixo. Estes notebooks são exploração: podem quebrar, podem ficar desatualizados,
e nada depende deles. Quando uma análise daqui provar valor, ela migra para o
pacote `cvlabstats` e para o relatório.
