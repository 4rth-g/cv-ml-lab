# Makefile — atalhos de treino e manutenção do cv-ml-lab.
#
# A GPU Intel Arc (XPU) é detectada automaticamente por train.py. Os alvos de
# treino usam `uv run --no-sync` (para o uv não reverter o torch +xpu para o
# build CUDA) e os env vars de estabilidade da Arc. Em CPU esses env vars são
# inofensivos, então o mesmo Makefile serve nas duas máquinas.
#
# Uso: make train-mlp DATASET=mnist   |   make train EXP=cnn DATASET=fashion_mnist

DATASET ?= mnist
EXP     ?= cnn
BUDGET  ?= default
ARGS    ?=

XPU_ENV := UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 MPLBACKEND=Agg
RUN     := $(XPU_ENV) uv run --no-sync cvlab-train

.PHONY: help train train-perceptron train-mlp train-cnn train-cifar smoke xpu xpu-check test lint format typecheck docs docs-serve sync clean

help:
	@echo "cv-ml-lab — alvos:"
	@echo "  make train EXP=<perceptron|mlp|cnn> DATASET=<mnist|fashion_mnist|cifar10>  treino oficial (offline)"
	@echo "  make train-perceptron | train-mlp | train-cnn [DATASET=...]        atalhos por arquitetura"
	@echo "  make train-cifar [EXP=...]                                         CIFAR-10 com o budget longo"
	@echo "  make smoke [EXP=... DATASET=...]                                   verificacao rapida (logger off)"
	@echo "  make xpu                                                           (re)instala o torch build +xpu"
	@echo "  make xpu-check                                                     mostra torch / xpu disponivel"
	@echo "  make test | lint | format                                         pytest / ruff check / ruff format"
	@echo "  make typecheck                                                     ty (Astral) - advisory, fora do CI"
	@echo "  make docs | docs-serve                                            build do site / servidor local (mkdocs)"
	@echo "  make sync                                                          uvx wandb sync dos offline-runs locais"
	@echo "  make clean                                                         remove results/ e caches"
	@echo "  Variaveis: DATASET(=$(DATASET))  EXP(=$(EXP))  BUDGET(=$(BUDGET))  ARGS='overrides extras do Hydra'"
	@echo "  BUDGET: smoke | default | long (orcamento da busca, em configs/tuning/budget/)"

train:
	$(RUN) +experiment=$(EXP) dataset=$(DATASET) tuning/budget=$(BUDGET) logger.mode=offline $(ARGS)

train-perceptron:
	$(MAKE) train EXP=perceptron

train-mlp:
	$(MAKE) train EXP=mlp

train-cnn:
	$(MAKE) train EXP=cnn

# CIFAR-10 nao converge em 10 epocas: usa o budget longo por padrao.
train-cifar:
	$(MAKE) train DATASET=cifar10 BUDGET=long

smoke:
	$(RUN) +experiment=$(EXP) dataset=$(DATASET) tuning/budget=smoke logger.mode=disabled $(ARGS)

xpu:
	uv pip uninstall torch torchvision
	uv pip install --index-url https://download.pytorch.org/whl/xpu torch torchvision
	$(MAKE) xpu-check

xpu-check:
	uv run --no-sync python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())"

test:
	uv run --no-sync pytest -q

lint:
	uv run --no-sync ruff check src tests
	uv run --no-sync ruff format --check .

format:
	uv run --no-sync ruff format .

# Type checking com o ty (Astral). Deliberadamente FORA do CI: o ty ainda e
# preview e boa parte dos diagnosticos vem da tipagem do proprio torch
# (Dataset nao declara __len__), nao do codigo daqui. Use como orientacao.
# Sai com codigo != 0 enquanto houver diagnostico - e esperado, nao e falha do alvo.
typecheck:
	uvx ty check --python .venv/bin/python src tests

docs:
	uv run --no-sync mkdocs build

docs-serve:
	uv run --no-sync mkdocs serve

sync:
	uvx wandb sync wandb/offline-run-*

clean:
	rm -rf results outputs .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
