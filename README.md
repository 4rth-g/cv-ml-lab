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
├── configs/            # Hydra: dataset / model / tuning (+ budget) / logger / trainer
├── src/cvlab/
│   ├── data/           # DataModules (base + dataset_cls: mnist, fashion_mnist, cifar10)
│   ├── models/         # CNN, ResNet, MLP e Perceptron + factory + LitClassifier
│   ├── tuning/         # busca Optuna (search.py) + comparação rigorosa (rigorous.py)
│   ├── tracking.py     # W&B, SQLite, run_name e mcnemar_test
│   ├── xpu.py          # accelerator Intel Arc (XPU)
│   └── train.py        # entrypoint (cvlab-train)
├── tests/              # pytest (forward, mcnemar, datamodule, lit_module)
├── justfile            # atalhos de treino e manutenção (`just`)
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

Para atalhos que já embutem `uv run --no-sync` e os env vars da Arc, use o [`justfile`](justfile) — `just` sem argumentos lista tudo, com os parâmetros de cada recipe:

```bash
just train mlp mnist           # treino oficial (offline)
just train cnn cifar10 long    # arquitetura, dataset e orçamento, posicionais
just train-cifar               # atalho: CIFAR-10 já com o orçamento longo
just smoke cnn                 # verificação rápida do pipeline
just test                      # pytest
just lint                      # os mesmos checks do CI
just xpu                       # (re)instala o torch build +xpu (Arc)
just cpu                       # torch CPU-only, para máquinas sem GPU Intel
```

Overrides do Hydra podem ser passados ao final de qualquer recipe de treino: `just train cnn cifar10 long model.width=32`.

## Configuração

Toda a configuração fica em `configs/` (Hydra). O arquivo `configs/train.yaml` compõe as seções `dataset`, `model`, `tuning`, `logger` e `trainer`; qualquer campo pode ser sobrescrito na CLI (ex.: `dataset=fashion_mnist model.dropout_rate=0.3`).

A arquitetura é escolhida por configuração, sem código específico no tuning: uma fábrica de modelos (`models/factory.py`) instancia o `_target_` do `configs/model/<arq>.yaml`, e a busca lê um `search_space` tipado (`categorical`/`float`/`int`) do `configs/tuning/<arq>.yaml`. Os presets em `configs/experiment/` casam modelo e espaço de busca (ex.: `+experiment=mlp`), evitando combinar um modelo com o espaço de busca de outro. Adicionar um dataset ou uma arquitetura segue os checklists em [`docs/WORKFLOW.md`](docs/WORKFLOW.md).

O **espaço de busca** e o **orçamento** da busca são eixos separados: o primeiro é propriedade da arquitetura, o segundo do dataset e do hardware. Por isso `n_trials`, `epochs_gs`, `gs_subset_size`, `n_seeds` e `final_epochs` vivem no grupo `configs/tuning/budget/` (`smoke`, `default`, `long`) e se combinam com qualquer arquitetura: `tuning/budget=long`. Assim um dataset mais pesado como o CIFAR-10 não exige duplicar o `search_space` num arquivo novo.

A **receita de otimização também é configuração**, não código. Otimizador e scheduler de learning rate são escolhidos por nome, a partir das tabelas `OPTIMIZERS` e `SCHEDULERS` em `models/lit_module.py`, e podem entrar no `search_space` como qualquer outro hiperparâmetro. A consequência metodológica importa mais que a comodidade: em vez de o autor decidir no fonte que "ResNet usa cosine annealing", a decisão passa pelo mesmo crivo de todo o resto — busca, retreino multi-seed e teste pareado. O default é `scheduler: none`, então arquiteturas que não declaram o campo treinam exatamente como antes.

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
| CIFAR-10 | 74,33% ± 0,38 | 85,39% ± 0,23 | +11,06 pp | t=54,92 (7×10⁻⁷) | 5×10⁻¹⁶⁴ |

Nos três casos a busca selecionou uma CNN de 3 blocos convolucionais, com diferença significativa no teste t pareado. No MNIST, próximo do teto de acurácia (~99%), o teste de McNemar não indica diferença — ilustrando a utilidade de reportar mais de um teste.

A magnitude do ganho acompanha a dificuldade do dataset. Em MNIST quase tudo satura perto de 99% e a busca tem pouco a recuperar; em CIFAR-10 a baseline (2 blocos, 32 filtros, 21 K parâmetros) é claramente subdimensionada, e a configuração escolhida (3 blocos, 128 filtros, `fc_units=512`, AdamW, 1,5 M parâmetros) rende +11 pp. É o caso em que a busca de arquitetura efetivamente paga — e onde o `d_z` de 24,6 diz que a diferença não é só significativa, é grande.

Vale contrastar as duas etapas do método: a busca reportou 76,8% de val_acc no subconjunto de 20 k com 15 épocas, enquanto o retreino multi-seed no split completo com 30 épocas chegou a 86,0%. A métrica da busca é um proxy barato para *ordenar* configurações, não uma estimativa de desempenho — por isso a decisão final vem do multi-seed. O CIFAR-10 usou `tuning/budget=long` (~53 min numa Intel Arc B580).

### Arquiteturas no CIFAR-10

Mesmo dataset, mesmo orçamento (`tuning/budget=long`), trocando só a arquitetura por configuração:

| Arquitetura | Baseline | Tunado (Optuna) | Δ | t pareado (p) | McNemar (p) |
|---|---|---|---|---|---|
| Perceptron linear | 34,36% ± 0,06 | 37,30% ± 0,30 | +2,94 pp | t=20,15 (4×10⁻⁵) | 1×10⁻⁹ |
| MLP | 43,92% ± 0,54 | 54,55% ± 0,21 | +10,63 pp | t=58,46 (5×10⁻⁷) | 1×10⁻⁸⁹ |
| CNN | 74,33% ± 0,38 | 85,39% ± 0,23 | +11,06 pp | t=54,92 (7×10⁻⁷) | 5×10⁻¹⁶⁴ |

A escada 37% → 55% → 85% é o argumento central a favor da convolução em imagem natural: achatar a imagem, como fazem o perceptron e o MLP, descarta a estrutura espacial que a CNN explora. O contraste com o MNIST é instrutivo — lá um classificador linear já passa de 90%, porque dígitos centralizados em fundo preto são quase linearmente separáveis nos próprios pixels.

O ganho da busca também cresce com a capacidade do modelo (+2,94, +10,63, +11,06 pp). No perceptron linear não há arquitetura para buscar — só `lr`, `optimizer`, `weight_decay` e `batch_size` —, então o teto do tuning é baixo por construção. É o comportamento esperado, e serve de sanidade: se a busca "melhorasse" muito um modelo sem hiperparâmetros de arquitetura, algo estaria errado.

## Hardware

O trainer usa `accelerator: auto`, detectando XPU (Intel Arc), CUDA, MPS ou CPU. O suporte a Intel Arc é feito por um accelerator dedicado (`src/cvlab/xpu.py`), já que o PyTorch Lightning não trata XPU nativamente. Sem GPU, a execução recai em CPU.

Na Arc, use `uv run --no-sync` — como o build `+xpu` não está fixado no `pyproject`, um `uv run` normal reverteria o torch para o build CUDA. As variáveis `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` e `MPLBACKEND=Agg` são recomendadas para estabilidade:

```bash
UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 MPLBACKEND=Agg uv run --no-sync cvlab-train +experiment=mlp dataset=mnist logger.mode=offline
```

Quando a máquina de treino não alcança o W&B, registre offline (`logger.mode=offline`) e sincronize de outra máquina — ver [`docs/WORKFLOW.md`](docs/WORKFLOW.md).

## Estado atual e próximos passos

Implementado: arquiteturas CNN configurável (profundidade, filtros, dropout, FC), ResNet (profundidade e largura configuráveis, com conexões residuais), MLP e perceptron linear, selecionáveis por configuração via fábrica de modelos e objective genérico; datasets MNIST, Fashion-MNIST e CIFAR-10; orçamento de busca como grupo reutilizável (`tuning/budget`); otimizador e scheduler de LR como hiperparâmetros de busca; método rigoroso completo; tracking; testes e CI.

Planejado:
- Novas arquiteturas (LeNet, DenseNet) pelo mesmo mecanismo de configuração.
- Backend scikit-learn (Random Forest, XGBoost) como segundo eixo de treino.

## Licença

[MIT](LICENSE) © 2026 Arthur Grazzia.
