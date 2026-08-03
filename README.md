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
│   ├── models/         # ConfigurableCNN + LitClassifier
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
# GPU Intel Arc (XPU): instalar o build XPU do PyTorch separadamente
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
```

## Uso

```bash
# Executar o método completo em um dataset
uv run cvlab-train dataset=mnist
uv run cvlab-train dataset=fashion_mnist

# Sobrescrever parâmetros pela linha de comando (Hydra)
uv run cvlab-train dataset=mnist tuning.n_trials=20 trainer.max_epochs=15

# Execução rápida de verificação
uv run cvlab-train dataset=mnist tuning.n_trials=2 tuning.final_epochs=1 logger.mode=disabled
```

## Configuração

Toda a configuração fica em `configs/` (Hydra). O arquivo `configs/train.yaml` compõe as seções `dataset`, `model`, `tuning`, `logger` e `trainer`; qualquer campo pode ser sobrescrito na CLI (ex.: `dataset=fashion_mnist model.dropout_rate=0.3`). Adicionar um dataset ou uma arquitetura segue os checklists em [`docs/WORKFLOW.md`](docs/WORKFLOW.md).

## Documentação da API

A referência da API é gerada a partir das docstrings do pacote com [pdoc](https://pdoc.dev):

```bash
uv run pdoc cvlab              # servidor local em http://localhost:8080
uv run pdoc cvlab -o docs/api  # HTML estático em docs/api/
```

O pdoc importa o pacote para introspecção, portanto requer as dependências instaladas.

## Resultados

Relatório público no Weights & Biases: [Resultados — cv-ml-lab](https://api.wandb.ai/links/grazziaarthur-universidade-federal-de-sergipe/trq7urpk). Cada run contém a configuração completa, as curvas de treino por época e as métricas do método.

| Dataset | Baseline | Tunado (Optuna) | Δ | t pareado (p) | McNemar (p) |
|---|---|---|---|---|---|
| MNIST | 99,22% ± 0,09 | 99,40% ± 0,07 | +0,18 pp | t=5,19 (0,007) | 0,18 |
| Fashion-MNIST | 90,36% ± 0,13 | 91,44% ± 0,15 | +1,08 pp | t=10,87 (0,0004) | 7×10⁻⁵ |

Em ambos os casos a busca selecionou uma CNN de 3 blocos convolucionais, com diferença significativa no teste t pareado. No MNIST, próximo do teto de acurácia (~99%), o teste de McNemar não indica diferença — ilustrando a utilidade de reportar mais de um teste.

## Hardware

O trainer usa `accelerator: auto`, detectando XPU (Intel Arc), CUDA, MPS ou CPU. O suporte a Intel Arc é feito por um accelerator dedicado (`src/cvlab/xpu.py`), já que o PyTorch Lightning não trata XPU nativamente. Sem GPU, a execução recai em CPU.

## Estado atual e próximos passos

Implementado: CNN configurável (profundidade, filtros, dropout, FC); datasets MNIST e Fashion-MNIST; método rigoroso completo; tracking; testes e CI.

Planejado:
- Fábrica de modelos + objective genérico, para MLP, Perceptron, LeNet e ResNet apenas por configuração.
- Backend scikit-learn (Random Forest, XGBoost) como segundo eixo de treino.
- Dataset CIFAR-10.

## Licença

[MIT](LICENSE) © 2026 Arthur Grazzia.
