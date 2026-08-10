# cv-ml-lab

Framework para classificação de imagens com PyTorch Lightning, Hydra, Optuna e Weights & Biases. Aplica um mesmo procedimento de busca de hiperparâmetros e comparação estatística a qualquer dataset de imagem, definido por configuração em vez de código.

## Objetivo

Reunir, num framework único e dataset-agnóstico, os datasets e arquiteturas explorados nos repositórios [`mnist-study`](https://github.com/4rth-g/mnist-study) e [`fashion-mnist-fundamentos-ia`](https://github.com/4rth-g/fashion-mnist-fundamentos-ia), aplicando o mesmo protocolo de tuning e avaliação a todos. Serve também como ambiente de experimentação com PyTorch Lightning.

O fluxo de trabalho (exploração em notebooks → formalização neste repositório) está descrito em [`docs/WORKFLOW.md`](docs/WORKFLOW.md).

## Stack

- **PyTorch Lightning** — organização do modelo e do loop de treino.
- **Hydra** — configuração hierárquica e reproduzível.
- **Optuna** — busca de hiperparâmetros (TPESampler + MedianPruner).
- **Weights & Biases** — registro de experimentos, com fallback local em SQLite.

## Estrutura

```
cv-ml-lab/
├── configs/            # Hydra: dataset / model / tuning / logger / trainer
├── src/cvlab/
│   ├── data/           # DataModules (registry, base, mnist, fashion_mnist)
│   ├── models/         # CNN, MLP e Perceptron + fábrica (factory) + LitClassifier
│   ├── tuning/         # busca Optuna (search.py) + comparação rigorosa (rigorous.py)
│   ├── tracking.py     # W&B, SQLite e mcnemar_test
│   ├── xpu.py          # accelerator Intel Arc (XPU)
│   └── train.py        # entrypoint (cvlab-train)
├── tests/              # pytest (forward, mcnemar, datamodule)
└── docs/WORKFLOW.md
```

## Método

1. **Busca (Optuna)** em um subconjunto do treino, com a configuração baseline enfileirada como primeiro trial (`study.enqueue_trial`), garantindo que o melhor da busca não fique abaixo da baseline na validação.
2. **Retreino multi-seed** da baseline e da melhor configuração no split de treino completo, com restauração dos pesos da melhor época segundo a validação.
3. **Comparação estatística pareada**: teste t pareado (`scipy.stats.ttest_rel`), tamanho de efeito de Cohen ($d_z$), intervalo de confiança de 95% e teste de McNemar (`binom_exact` / `chi2_continuity`).
4. **Seleção na validação**: a configuração tunada só substitui a baseline se `best_val_mean >= baseline_val_mean`. O conjunto de teste é usado uma única vez, para reportar.

## Instalação

```bash
uv sync
```

Para GPU Intel Arc (XPU), o PyTorch precisa ser o build `+xpu` (o padrão do PyPI é CUDA). O `uv` não troca um build já instalado que satisfaça a versão, então desinstale antes:

```bash
uv pip uninstall torch torchvision
uv pip install --index-url https://download.pytorch.org/whl/xpu torch torchvision
# Verificar: deve terminar em "+xpu True 1"
uv run --no-sync python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())"
```

## Uso

```bash
# Executar o método completo em um dataset (arquitetura padrão: CNN)
uv run cvlab-train dataset=mnist
uv run cvlab-train dataset=fashion_mnist

# Escolher a arquitetura por preset (casa modelo + espaço de busca)
uv run cvlab-train +experiment=perceptron dataset=mnist
uv run cvlab-train +experiment=mlp dataset=fashion_mnist
uv run cvlab-train +experiment=cnn dataset=mnist

# Escolher o orçamento da busca (n_trials, épocas, seeds) — independente da arquitetura
uv run cvlab-train +experiment=cnn dataset=cifar10 tuning/budget=long

# Sobrescrever parâmetros pela linha de comando (Hydra)
uv run cvlab-train dataset=mnist tuning.n_trials=20 tuning.final_epochs=15

# Execução rápida de verificação
uv run cvlab-train dataset=mnist tuning/budget=smoke logger.mode=disabled
```

Para atalhos que já embutem `uv run --no-sync` e os env vars da Arc, use o `Makefile` (ver `make help`):

```bash
make train-mlp DATASET=mnist   # treino oficial (offline)
make train-cifar               # CIFAR-10 com o orçamento longo
make smoke EXP=cnn             # verificação rápida
make xpu                       # (re)instala o torch build +xpu
```

## Configuração

Toda a configuração fica em `configs/` (Hydra). O arquivo `configs/train.yaml` compõe as seções `dataset`, `model`, `tuning`, `logger` e `trainer`; qualquer campo pode ser sobrescrito na CLI (ex.: `dataset=fashion_mnist model.dropout_rate=0.3`).

A arquitetura é escolhida por configuração, sem código específico no tuning: uma fábrica de modelos (`models/factory.py`) instancia o `_target_` do `configs/model/<arq>.yaml`, e a busca lê um `search_space` tipado (`categorical`/`float`/`int`) do `configs/tuning/<arq>.yaml`. Os presets em `configs/experiment/` casam modelo e espaço de busca (ex.: `+experiment=mlp`), evitando combinar um modelo com o espaço de busca de outro. Adicionar um dataset ou uma arquitetura segue os checklists em [`docs/WORKFLOW.md`](docs/WORKFLOW.md).

O **espaço de busca** e o **orçamento** da busca são eixos separados: o primeiro é propriedade da arquitetura, o segundo do dataset e do hardware. Por isso `n_trials`, `epochs_gs`, `gs_subset_size`, `n_seeds` e `final_epochs` vivem no grupo `configs/tuning/budget/` (`smoke`, `default`, `long`) e se combinam com qualquer arquitetura: `tuning/budget=long`. Assim um dataset mais pesado como o CIFAR-10 não exige duplicar o `search_space` num arquivo novo.

Na camada de dados vale o mesmo princípio: `BaseImageClfDataModule` implementa download, splits e o pipeline de transformações uma única vez, dirigido pelo atributo de classe `dataset_cls`. Um novo dataset do torchvision é uma subclasse de ~15 linhas — a classe do dataset, `mean`/`std`/`class_names` e a política de augmentation do domínio em `_augment_ops()` — mais o yaml correspondente.

## Documentação da API

A documentação (narrativa em `docs/` + referência da API gerada das docstrings) usa
[MkDocs](https://www.mkdocs.org/) com [Material](https://squidfunk.github.io/mkdocs-material/)
e [mkdocstrings](https://mkdocstrings.github.io/):

```bash
uv run mkdocs serve   # servidor local em http://127.0.0.1:8000
uv run mkdocs build   # site estático em site/
```

A referência é extraída por análise estática (griffe), sem importar o pacote. As
docstrings seguem o estilo Google.

## Resultados

Relatório público no Weights & Biases: [Resultados — cv-ml-lab](https://api.wandb.ai/links/grazziaarthur-universidade-federal-de-sergipe/trq7urpk). Cada run contém a configuração completa, as curvas de treino por época e as métricas do método.

| Dataset | Baseline | Tunado (Optuna) | Δ | t pareado (p) | McNemar (p) |
|---|---|---|---|---|---|
| MNIST | 99,22% ± 0,09 | 99,40% ± 0,07 | +0,18 pp | t=5,19 (0,007) | 0,18 |
| Fashion-MNIST | 90,36% ± 0,13 | 91,44% ± 0,15 | +1,08 pp | t=10,87 (0,0004) | 7×10⁻⁵ |

Em ambos os casos a busca selecionou uma CNN de 3 blocos convolucionais, com diferença significativa no teste t pareado. No MNIST, próximo do teto de acurácia (~99%), o teste de McNemar não indica diferença — ilustrando a utilidade de reportar mais de um teste.

## Hardware

O trainer usa `accelerator: auto`, detectando XPU (Intel Arc), CUDA, MPS ou CPU. O suporte a Intel Arc é feito por um accelerator dedicado (`src/cvlab/xpu.py`), já que o PyTorch Lightning não trata XPU nativamente. Sem GPU, a execução recai em CPU.

Na Arc, use `uv run --no-sync` — como o build `+xpu` não está fixado no `pyproject`, um `uv run` normal reverteria o torch para o build CUDA. As variáveis `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` e `MPLBACKEND=Agg` são recomendadas para estabilidade:

```bash
UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 MPLBACKEND=Agg uv run --no-sync cvlab-train +experiment=mlp dataset=mnist logger.mode=offline
```

Quando a máquina de treino não alcança o W&B, registre offline (`logger.mode=offline`) e sincronize de outra máquina — ver [`docs/WORKFLOW.md`](docs/WORKFLOW.md).

## Estado atual e próximos passos

Implementado: arquiteturas CNN configurável (profundidade, filtros, dropout, FC), MLP e perceptron linear, selecionáveis por configuração via fábrica de modelos e objective genérico; datasets MNIST, Fashion-MNIST e CIFAR-10; orçamento de busca como grupo reutilizável (`tuning/budget`); método rigoroso completo; tracking; testes e CI.

Planejado:
- Novas arquiteturas (LeNet, ResNet) pelo mesmo mecanismo de configuração.
- Backend scikit-learn (Random Forest, XGBoost) como segundo eixo de treino.

## Licença

[MIT](LICENSE) © 2026 Arthur Grazzia.
