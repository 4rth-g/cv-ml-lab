# Atalhos de treino e manutenção do cv-ml-lab.
#
# A GPU Intel Arc (XPU) é detectada automaticamente por train.py. As recipes de
# treino usam `uv run --no-sync` (para o uv não reverter o torch +xpu para o
# build CUDA) e os env vars de estabilidade da Arc. Em CPU esses env vars são
# inofensivos, então o mesmo justfile serve nas duas máquinas.
#
# `just` sem argumentos lista tudo. Ex.: just train mlp mnist
#                                       just train cnn cifar10 long

xpu_env := "UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1 MPLBACKEND=Agg"
run := xpu_env + " uv run --no-sync cvlab-train"

[private]
default:
    @just --list

# Treino oficial (offline). budget: smoke|default|long. Aceita overrides do Hydra ao final.
train exp="cnn" dataset="mnist" budget="default" *overrides:
    {{run}} +experiment={{exp}} dataset={{dataset}} tuning/budget={{budget}} logger.mode=offline {{overrides}}

# CIFAR-10 com o orçamento longo: 10 épocas não bastam nesse dataset.
train-cifar exp="cnn" *overrides:
    @just train {{exp}} cifar10 long {{overrides}}

# O output_dir separado NÃO é detalhe: optuna_search retoma estudos com
# load_if_exists=True, então um smoke gravando em results/ deixaria trials de
# orçamento smoke no estudo que o run real depois herdaria — contaminando a
# busca e anulando a baseline enfileirada como piso.

# Verificação rápida do pipeline (sem logger, resultados descartáveis)
smoke exp="cnn" dataset="mnist" *overrides:
    {{run}} +experiment={{exp}} dataset={{dataset}} tuning/budget=smoke logger.mode=disabled output_dir=results/_smoke {{overrides}}

# (Re)instala o torch build +xpu (Intel Arc)
xpu:
    uv pip uninstall torch torchvision
    uv pip install --index-url https://download.pytorch.org/whl/xpu torch torchvision
    @just xpu-check

# Instala o torch CPU-only (~200 MB, sem o stack CUDA). Suficiente para lint e testes.
cpu:
    uv pip uninstall torch torchvision
    uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Mostra a versão do torch e se a XPU está disponível
xpu-check:
    uv run --no-sync python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())"

# Roda a suíte de testes
test *args:
    uv run --no-sync pytest -q {{args}}

# ruff check + ruff format --check (mesmos comandos do CI)
lint:
    uv run --no-sync ruff check .
    uv run --no-sync ruff format --check .

# Aplica o ruff format
format:
    uv run --no-sync ruff format .

# Deliberadamente FORA do CI: o ty ainda é preview e boa parte dos diagnósticos
# vem da tipagem do próprio torch (Dataset não declara __len__), não do código
# daqui. Sai com código != 0 enquanto houver diagnóstico — é esperado, não é
# falha da recipe.

# Type checking com o ty (Astral) — advisory, fora do CI
typecheck:
    uvx ty check --python .venv/bin/python src tests

# Build do site de documentação
docs:
    uv run --no-sync mkdocs build

# Servidor local da documentação
docs-serve:
    uv run --no-sync mkdocs serve

# Envia ao W&B os runs registrados offline
sync:
    uvx wandb sync wandb/offline-run-*

# Remove caches e os resultados descartáveis do smoke
clean:
    rm -rf results/_smoke outputs .pytest_cache .ruff_cache
    find . -type d -name __pycache__ -prune -exec rm -rf {} +

# CUIDADO: apaga results/ inteiro — checkpoints, estudos do Optuna e o tracker.
clean-results:
    rm -rf results
