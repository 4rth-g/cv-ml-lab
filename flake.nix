{
  description = "cv-ml-lab: ambiente R para a camada de análise estatística (analysis/)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Os pacotes que `analysis/` importa. effectsize dá o Cohen's d COM
        # intervalo (que o Python não calculava) e PMCMRplus cobre Friedman e os
        # post-hoc pareados. scmamp ficou de fora de propósito: está arquivado no
        # CRAN.
        rPkgs =
          with pkgs.rPackages;
          [
            # leitura e manipulação
            readr
            dplyr
            tidyr
            purrr
            jsonlite
            glue
            # estatística
            effectsize
            PMCMRplus
            TOSTER
            # gráficos e relatório
            ggplot2
            scales
            knitr
            rmarkdown
            # imagens: o Python exporta PNG, o R só lê. É o que permite EDA
            # visual sem quebrar a regra de o R nunca decodificar dataset.
            magick
            png
            patchwork
            # notebooks (.ipynb com kernel R)
            IRkernel
            repr
            IRdisplay
            # desenvolvimento do pacote cvlabstats
            devtools
            roxygen2
            testthat
            # tooling de editor: a extensão R do VS Code depende do
            # `languageserver` para autocomplete, hover e diagnóstico, e do
            # `httpgd` para mostrar gráfico no painel em vez de abrir janela.
            languageserver
            httpgd
          ];

        rEnv = pkgs.rWrapper.override { packages = rPkgs; };

        # REPL recomendado pela extensão R do VS Code. Precisa dos MESMOS pacotes
        # que o rEnv, senão o console enxerga uma biblioteca diferente do resto.
        radianEnv = pkgs.radianWrapper.override { packages = rPkgs; };

        # Kernel R para o Jupyter, declarado em vez de instalado.
        #
        # O caminho usual é `IRkernel::installspec()`, que escreve em
        # ~/.local/share/jupyter/kernels e deixa apontando para um R que pode
        # sumir numa troca de geração do Nix. Construir o kernelspec aqui e
        # expor via JUPYTER_PATH mantém tudo reprodutível e sem tocar no $HOME.
        irKernel = pkgs.runCommand "cvlab-irkernel" { } ''
          mkdir -p $out/kernels/cvlab-r
          cat > $out/kernels/cvlab-r/kernel.json <<'EOF'
          {
            "argv": [
              "@R@/bin/R", "--slave", "-e", "IRkernel::main()",
              "--args", "{connection_file}"
            ],
            "display_name": "R (cvlab)",
            "language": "R"
          }
          EOF
          substituteInPlace $out/kernels/cvlab-r/kernel.json --replace '@R@' '${rEnv}'
        '';

        pyTools = pkgs.python3.withPackages (ps: [
          ps.jupyterlab
          ps.notebook
          # Pareia cada .ipynb com um .md legível: o notebook guarda as saídas
          # (é o que faz o GitHub renderizar) e o par de texto dá diff decente.
          ps.jupytext
        ]);
      in
      {
        # Shell padrão: SÓ R e Quarto.
        #
        # O Python fica deliberadamente de fora. O torch deste projeto vem de
        # wheel do índice +xpu e, em NixOS, wheels binárias dependem de nix-ld e
        # do stack Level Zero da Arc; misturar isso aqui é a via mais rápida de
        # quebrar um treino que hoje funciona. O `.venv` do uv e este shell
        # coexistem sem se ver, que é justamente o ponto da separação:
        # `analysis/` nunca importa código Python.
        devShells.default = pkgs.mkShell {
          # O Quarto NÃO entra aqui de propósito. O `quarto` do nixpkgs (1.10.18)
          # empacota um pandoc 3.7 mais antigo do que as opções que ele próprio
          # gera, e `quarto render` morre com `Unknown option
          # "syntax-highlighting"`. Usamos o Quarto instalado no sistema, que o
          # shellHook confere. Reavaliar quando o nixpkgs corrigir o par
          # quarto/pandoc.
          packages = [
            rEnv
            radianEnv
            pyTools
            pkgs.just
          ];

          shellHook = ''
            # Kernel R visível ao Jupyter sem escrever nada no $HOME.
            export JUPYTER_PATH="${irKernel}''${JUPYTER_PATH:+:$JUPYTER_PATH}"

            # O rWrapper injeta R_LIBS_SITE no momento do exec, então processos
            # FILHOS que chamam R não o herdam. É o caso do Quarto, cujo motor
            # knitr abre um R próprio: sem isto, `quarto render` falha com
            # "não há nenhum pacote chamado 'devtools'" mesmo dentro do shell.
            # Perguntar ao próprio R quais são os libPaths e exportar cobre a
            # árvore de dependências inteira, sem duplicar a lista aqui.
            export R_LIBS_SITE="$(R --no-echo -e 'cat(paste(.libPaths(), collapse=":"))')"

            # Só fala quando há terminal. O kernelspec de usuário invoca este
            # shell para subir o R, e aí qualquer echo iria parar no stdout do
            # kernel, poluindo o que o Jupyter lê.
            if [ -t 1 ]; then
              echo "cv-ml-lab · shell de análise (R $(R --version | head -1 | cut -d' ' -f3))"
              if command -v quarto >/dev/null; then
                echo "  quarto $(quarto --version) (do sistema)"
              else
                echo "  ⚠ quarto ausente do PATH — 'just report-html' não vai funcionar."
              fi
              echo "  just report       relatório de um run"
              echo "  just report-all   análise agregada entre runs"
              echo "  just nb           JupyterLab com o kernel R (cvlab)"
              echo "  just report-test  suíte testthat de analysis/"
            fi
          '';
        };

        formatter = pkgs.nixfmt-tree;
      }
    );
}
