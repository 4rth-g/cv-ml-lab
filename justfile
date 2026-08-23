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
# `python -m` em vez do console script `cvlab-eda`: o entry point só existe após
# um `uv sync`, e sync aqui reverteria o torch +xpu para o build CUDA.
run_eda := xpu_env + " uv run --no-sync python -m cvlab.eda"

[private]
default:
    @just --list

# Treino oficial (offline). budget: smoke|default|long. Aceita overrides do Hydra ao final.
train exp="cnn" dataset="mnist" budget="default" *overrides:
    {{run}} +experiment={{exp}} dataset={{dataset}} tuning/budget={{budget}} logger.mode=offline {{overrides}}

# CIFAR-10 com o orçamento longo: 10 épocas não bastam nesse dataset.
train-cifar exp="cnn" *overrides:
    @just train {{exp}} cifar10 long {{overrides}}

# Ordem deliberada: dataset por FORA, arquitetura por dentro. Em
# `ranks.R::performance_matrix` as linhas são datasets (blocos) e as colunas são
# modelos; terminar um dataset fecha uma LINHA inteira, e uma linha completa já
# compara as 4 arquiteturas entre si. Na ordem inversa uma interrupção deixaria
# colunas parciais, que não comparam nada.
#
# `objective=balanced_accuracy` no derma (razão 58,7) e no breast (2,7): lá a
# acurácia simples é maximizada ignorando as classes raras, e a escolha de
# config sairia guiada pela majoritária. É critério de SELEÇÃO; a coluna `acc`
# do export continua sendo acurácia simples em todo run. Ver `cvlab.metrics`.
#
# Screening NÃO é resultado reportável (n_seeds=3, busca em subconjunto). Serve
# para achar onde vale gastar `default`. Ao promover um par, APAGUE o
# `results/optuna_study_<dataset>_<modelo>.db` correspondente: o estudo é
# retomado com load_if_exists e herdaria os trials de screening.

# Grid de varredura: 4 arquiteturas x 4 datasets do EDA, orçamento screening
grid:
    #!/usr/bin/env bash
    set -uo pipefail
    falhas=()
    for spec in "mnist|" \
                "medmnist|dataset.subset=breastmnist tuning.objective=balanced_accuracy" \
                "medmnist|dataset.subset=dermamnist tuning.objective=balanced_accuracy" \
                "medmnist|dataset.subset=organmnist3d"; do
        ds="${spec%%|*}"; extra="${spec#*|}"
        for exp in perceptron mlp cnn resnet; do
            echo "=== $exp / $ds $extra ==="
            if ! just train "$exp" "$ds" screening $extra; then
                falhas+=("$exp/$ds $extra")
            fi
        done
    done
    echo
    if [ ${#falhas[@]} -eq 0 ]; then
        echo "grid completo: $(wc -l < results/runs/index.csv) linhas no índice"
    else
        printf 'FALHARAM (%d):\n' "${#falhas[@]}"; printf '  %s\n' "${falhas[@]}"
    fi
    echo "Próximo passo: just report-all"

# O output_dir separado NÃO é detalhe: optuna_search retoma estudos com
# load_if_exists=True, então um smoke gravando em results/ deixaria trials de
# orçamento smoke no estudo que o run real depois herdaria — contaminando a
# busca e anulando a baseline enfileirada como piso.

# Verificação rápida do pipeline (sem logger, resultados descartáveis)
smoke exp="cnn" dataset="mnist" *overrides:
    {{run}} +experiment={{exp}} dataset={{dataset}} tuning/budget=smoke logger.mode=disabled output_dir=results/_smoke {{overrides}}

# A chave (split, example_idx) casa com as predições exportadas pelo run, que é
# o que permite cruzar erro do modelo com propriedade da imagem no relatório.

# EDA do dataset -> results/datasets/<dataset_id>/ (desacoplado do treino)
eda dataset="mnist" *overrides:
    {{run_eda}} dataset={{dataset}} {{overrides}}

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

# --- Análise estatística (R) ---------------------------------------------
# Nenhuma destas usa `uv`: o lado R depende SÓ dos artefatos em results/, nunca
# de código Python.

# Prefixo que garante R disponível. Vazio quando o Rscript já está no PATH (ou
# seja, dentro de `nix develop`); senão entra no devShell automaticamente.
#
# O R deste projeto vive só no flake, então sem isto `just report` falha fora do
# devShell com "Unable to locate an installed version of R" — uma mensagem do
# Quarto que não diz o que fazer. Preferimos resolver a lembrar.
r_prefix := `command -v Rscript >/dev/null 2>&1 && echo "" || echo "nix develop --command"`

# Pré-condições das recipes de análise: existir run exportado e existir um R
# alcançável. As duas falhas são comuns e ambas geram erros ilegíveis quando
# aparecem lá dentro do Quarto.
[private]
preflight results_dir="results" recipe="report" run="latest":
    #!/usr/bin/env bash
    if [ ! -f "{{results_dir}}/runs/index.csv" ]; then
      echo "Nenhum run exportado em {{results_dir}}/runs/." >&2
      echo "" >&2

      # Procura runs em OUTRAS raízes antes de mandar treinar. O caso comum é o
      # usuário ter acabado de rodar `just smoke`, que grava em results/_smoke:
      # mandá-lo treinar de novo aqui seria dizer para refazer o que ele já fez.
      found=""
      for candidate in results results/_smoke; do
        if [ "$candidate" != "{{results_dir}}" ] && [ -f "$candidate/runs/index.csv" ]; then
          found="$candidate"
          break
        fi
      done

      if [ -n "$found" ]; then
        n=$(( $(wc -l < "$found/runs/index.csv") - 1 ))
        echo "Encontrei $n run(s) em $found/. Para analisar de lá:" >&2
        echo "" >&2
        if [ "{{recipe}}" = "report-all" ]; then
          echo "    just report-all $found" >&2
        else
          echo "    just {{recipe}} {{run}} $found" >&2
        fi
        if [ "$found" = "results/_smoke" ]; then
          echo "" >&2
          echo "Lembrando: orçamento smoke valida o pipeline, não gera resultado reportável." >&2
        fi
      else
        echo "A análise em R lê artefatos, não treina. Gere um primeiro:" >&2
        echo "    just train cnn mnist     # run de verdade, grava em results/" >&2
        echo "    just smoke cnn mnist     # rápido, grava em results/_smoke" >&2
      fi
      exit 1
    fi
    if ! command -v Rscript >/dev/null 2>&1 && ! command -v nix >/dev/null 2>&1; then
      echo "R não está no PATH e o nix também não." >&2
      echo "Instale o R com os pacotes de analysis/DESCRIPTION, ou use o devShell:" >&2
      echo "    nix develop" >&2
      exit 1
    fi
    if ! command -v Rscript >/dev/null 2>&1; then
      echo "R fora do PATH — usando o devShell do flake (nix develop)." >&2
    fi

# Análise de um run (default: o mais recente)
report run="latest" results_dir="results": (preflight results_dir "report" run)
    {{r_prefix}} Rscript analysis/run_report.R --run {{run}} --root {{results_dir}}

# Análise agregada entre runs: p corrigido, Friedman + Wilcoxon-Holm
report-all results_dir="results": (preflight results_dir "report-all")
    {{r_prefix}} Rscript analysis/run_report.R --all --root {{results_dir}}

# `results_dir` é relativo à RAIZ DO REPO (não a analysis/).

# Relatório HTML (Quarto) -> analysis/output/report.html
report-html run="latest" results_dir="results": (preflight results_dir "report-html" run)
    {{r_prefix}} quarto render analysis/report.qmd -P run_id:{{run}} -P root:../{{results_dir}} --output-dir output

# Sobe um servidor local e re-renderiza a cada save. Ctrl-C para sair.

# Preview do relatório com recarga automática
report-preview run="latest" results_dir="results": (preflight results_dir "report-preview" run)
    {{r_prefix}} quarto preview analysis/report.qmd -P run_id:{{run}} -P root:../{{results_dir}}

# --- Notebooks (.ipynb com kernel R) --------------------------------------
# Exploração livre, que CONSOME artefatos e nunca produz nenhum. Quem produz é
# sempre o `just`. Os .ipynb são versionados COM saídas, porque é isso que faz o
# GitHub renderizar a análise para quem só abre o link. O custo é diff ilegível:
# o pareamento com .Rmd via jupytext, que resolveria isso, foi removido porque o
# VS Code lê mal o formato.

# JupyterLab com o kernel R (cvlab)
nb:
    {{r_prefix}} jupyter lab notebooks/

# O flake registra o kernel via JUPYTER_PATH, o que só vale DENTRO do devShell.
# Editores que rodam fora dele (VS Code, RStudio) não enxergam o kernel e a
# lista aparece sem o R. Esta recipe grava um kernelspec no nível do usuário
# cujo argv chama `nix develop` no próprio repositório: resolve o R atual a cada
# execução, então não quebra quando o flake é atualizado.

# Registra o kernel R para editores fora do devShell (VS Code, RStudio)
nb-kernel:
    #!/usr/bin/env bash
    set -euo pipefail
    repo="$(git rev-parse --show-toplevel)"
    dir="${XDG_DATA_HOME:-$HOME/.local/share}/jupyter/kernels/cvlab-r"
    mkdir -p "$dir"
    cat > "$dir/kernel.json" <<EOF
    {
      "argv": [
        "nix", "develop", "$repo", "--command",
        "R", "--slave", "-e", "IRkernel::main()", "--args", "{connection_file}"
      ],
      "display_name": "R (cvlab)",
      "language": "R"
    }
    EOF
    echo "kernel gravado em $dir/kernel.json"
    echo "Reabra o VS Code e escolha o kernel 'R (cvlab)'."

# Reexecuta todos os notebooks, regenerando as saídas.
# Cada um roda no PRÓPRIO diretório: os caminhos de RAIZ e do pacote R são
# relativos à posição do notebook.
nb-run:
    #!/usr/bin/env bash
    set -euo pipefail
    {{r_prefix}} bash -c '
      for f in $(find notebooks -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*" | sort); do
        echo "--- $f"
        (cd "$(dirname "$f")" && jupyter nbconvert --to notebook --execute --inplace "$(basename "$f")")
      done'

# Suíte testthat do pacote cvlabstats
report-test:
    {{r_prefix}} Rscript -e 'devtools::test("analysis")'

# Shell Nix com R + Quarto
r-shell:
    nix develop

# Remove só a saída do relatório (não toca em results/)
clean-report:
    rm -rf analysis/output

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
