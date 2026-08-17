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
        rEnv = pkgs.rWrapper.override {
          packages = with pkgs.rPackages; [
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
            # desenvolvimento do pacote cvlabstats
            devtools
            roxygen2
            testthat
          ];
        };
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
            pkgs.just
          ];

          shellHook = ''
            # O rWrapper injeta R_LIBS_SITE no momento do exec, então processos
            # FILHOS que chamam R não o herdam. É o caso do Quarto, cujo motor
            # knitr abre um R próprio: sem isto, `quarto render` falha com
            # "não há nenhum pacote chamado 'devtools'" mesmo dentro do shell.
            # Perguntar ao próprio R quais são os libPaths e exportar cobre a
            # árvore de dependências inteira, sem duplicar a lista aqui.
            export R_LIBS_SITE="$(R --no-echo -e 'cat(paste(.libPaths(), collapse=":"))')"

            echo "cv-ml-lab · shell de análise (R $(R --version | head -1 | cut -d' ' -f3))"
            if command -v quarto >/dev/null; then
              echo "  quarto $(quarto --version) (do sistema)"
            else
              echo "  ⚠ quarto ausente do PATH — 'just report-html' não vai funcionar."
            fi
            echo "  just report       relatório de um run"
            echo "  just report-all   análise agregada entre runs"
            echo "  just report-test  suíte testthat de analysis/"
          '';
        };

        formatter = pkgs.nixfmt-tree;
      }
    );
}
