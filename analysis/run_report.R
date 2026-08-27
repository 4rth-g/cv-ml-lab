#!/usr/bin/env Rscript
# Driver da análise estatística do cv-ml-lab.
#
#   Rscript analysis/run_report.R --run latest
#   Rscript analysis/run_report.R --all
#
# Consome apenas os artefatos de results/. Não importa código Python, não abre
# imagem, não treina nada.

suppressWarnings(suppressMessages({
  ok <- requireNamespace("devtools", quietly = TRUE)
}))
if (!ok) {
  stop(
    "Pacote 'devtools' ausente. Rode dentro do ambiente: `nix develop` (ou `just r-shell`).",
    call. = FALSE
  )
}

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[i + 1]
}

pkg_dir <- file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
if (!dir.exists(file.path(pkg_dir, "R"))) pkg_dir <- "analysis"
suppressMessages(devtools::load_all(pkg_dir, quiet = TRUE))

fmt_pct <- function(x) sprintf("%.2f%%", 100 * x)
fmt_pp <- function(x) sprintf("%+.2f pp", 100 * x)
rule <- function(title) cat("\n", title, "\n", strrep("-", nchar(title)), "\n", sep = "")

report_run <- function(run_id, root = NULL) {
  run <- read_run(run_id, root)
  m <- run$manifest

  cat("=== cv-ml-lab · análise estatística ===\n")
  cat(sprintf("Run:      %s\n", run$run_id))
  cat(sprintf("Modelo:   %s | Dataset: %s%s\n", m$model, m$dataset,
              if (nzchar(m$subset %||% "")) paste0("/", m$subset) else ""))
  cat(sprintf("Commit:   %s | seeds: %d\n", m$commit, m$n_seeds))

  # Dizer QUAL variância foi medida evita ler o IC como se cobrisse tudo.
  v <- m$variance
  varied <- c(
    if (isTRUE(v$vary_init)) "inicialização",
    if (isTRUE(v$vary_data_order)) "ordem dos dados",
    if (isTRUE(v$vary_split)) "split"
  )
  cat(sprintf("Variância medida sobre: %s\n", paste(varied, collapse = ", ")))
  if (length(varied) == 1) {
    cat("  nota: só uma fonte varia, então o IC abaixo SUBESTIMA a variância real.\n")
  }

  paired <- paired_accuracies(run$seed_metrics, "test")

  rule("Teste pareado (acurácia de teste)")
  tt <- paired_t(paired)
  cat(sprintf("baseline: %s | tunada: %s | Δ = %s\n",
              fmt_pct(mean(paired$baseline)), fmt_pct(mean(paired$tuned)), fmt_pp(tt$diff_mean)))
  if (is.na(tt$p_value)) {
    cat("t pareado: indefinido (diferenças constantes entre as seeds)\n")
  } else {
    cat(sprintf("t(%d) = %.3f, p = %.4f\n", tt$df, tt$t, tt$p_value))
    cat(sprintf("IC95%% da diferença: [%s, %s]\n", fmt_pp(tt$ci_low), fmt_pp(tt$ci_high)))
    cat(sprintf("Cohen's d_z = %.3f  IC95%% [%.3f, %.3f]\n",
                tt$cohens_dz, tt$dz_ci_low, tt$dz_ci_high))
    w <- paired_wilcoxon(paired)
    cat(sprintf("Wilcoxon pareado (robustez): p = %.4f\n", w$p_value))
  }

  rule("Estimativas robustas")
  rs <- robust_summary(run, "test")
  cat(sprintf("IQM baseline: %s  IC95%% [%s, %s]\n",
              fmt_pct(rs$baseline_iqm), fmt_pct(rs$baseline_ci_low), fmt_pct(rs$baseline_ci_high)))
  cat(sprintf("IQM tunada:   %s  IC95%% [%s, %s]  (%s)\n",
              fmt_pct(rs$tuned_iqm), fmt_pct(rs$tuned_ci_low), fmt_pct(rs$tuned_ci_high), rs$ci_type))
  cat(sprintf("P(tunada > baseline) = %.2f\n", rs$prob_improvement))

  rule("McNemar (nível de exemplo)")
  mc <- mcnemar_from_run(run)
  cat(sprintf("n01 = %d, n10 = %d, %s, p = %.4e\n", mc$n01, mc$n10, mc$method, mc$p_value))

  eq <- equivalence_tost(paired)
  if (!is.null(eq)) {
    rule("Equivalência (TOST, margem de 1 p.p.)")
    cat(sprintf("p = %.4f — %s\n", eq$p_tost, eq$decision))
  }

  rule("Métricas pós-hoc (sem retreinar)")
  met <- run_metrics(run) |>
    dplyr::filter(.data$split == "test") |>
    dplyr::group_by(.data$arm) |>
    dplyr::summarise(
      accuracy = mean(.data$accuracy),
      balanced_accuracy = mean(.data$balanced_accuracy),
      f1_macro = mean(.data$f1_macro),
      .groups = "drop"
    )
  print(as.data.frame(met), row.names = FALSE, digits = 4)

  eda_dir <- file.path(results_root(root), "datasets", m$dataset_id)
  if (dir.exists(eda_dir)) {
    rule("Erro cruzado com o EDA do dataset")
    joined <- join_predictions_eda(run, root = root)
    by_class <- joined |>
      dplyr::filter(.data$split == "test") |>
      dplyr::group_by(.data$class_name) |>
      dplyr::summarise(
        n = dplyr::n(),
        erro = 1 - mean(.data$correct),
        intensidade_media = mean(.data$mean_intensity),
        .groups = "drop"
      ) |>
      dplyr::arrange(dplyr::desc(.data$erro))
    print(as.data.frame(by_class), row.names = FALSE, digits = 4)

    cw <- consistently_wrong(run, "test")
    cat(sprintf(
      "\nExemplos que TODAS as seeds erram: %d | que só algumas erram: %d\n",
      sum(cw$always_wrong), sum(cw$sometimes_wrong)
    ))
    cat("  (os primeiros são dificuldade do dado; os segundos, instabilidade de treino)\n")
  } else {
    cat(sprintf("\n(EDA ausente para '%s' — rode `just eda` para a análise de erro por classe)\n",
                m$dataset_id))
  }
}

report_all <- function(root = NULL) {
  cat("=== cv-ml-lab · análise agregada ===\n")
  tbl <- all_runs_table(root)

  rule("Todos os runs, com p corrigido para múltiplas comparações")
  cat(sprintf("Família de %d comparações. p bruto NÃO deve ser reportado sozinho.\n\n", nrow(tbl)))
  print(
    as.data.frame(tbl[, c(
      "model", "dataset", "n_seeds", "diff_pp", "ci_low_pp", "ci_high_pp",
      "cohens_dz", "p_value", "p_holm", "p_BH", "prob_improvement"
    )]),
    row.names = FALSE, digits = 3
  )

  rule("Friedman + post-hoc Wilcoxon-Holm")
  # Falta de runs é o estado NORMAL no início de um projeto: o agregado precisa
  # degradar com uma explicação, não abortar o relatório inteiro.
  fa <- tryCatch(friedman_analysis(performance_matrix(root)), error = function(e) e)

  if (inherits(fa, "error")) {
    cat("indisponível: ", conditionMessage(fa), "\n", sep = "")
    cat("Cada arquitetura precisa ter rodado nos MESMOS datasets para formar os blocos.\n")
  } else {
    cat(sprintf("Friedman: chi2(%d) = %.3f, p = %.4f  (%d datasets)\n",
                fa$friedman$df, fa$friedman$statistic, fa$friedman$p_value, fa$n_datasets))
    cat("\nRank médio (1 = melhor):\n")
    print(as.data.frame(fa$ranks), row.names = FALSE, digits = 3)
    cat("\nComparações par a par (Wilcoxon + Holm; Nemenyi é evitado por ser não-transitivo):\n")
    print(as.data.frame(fa$pairwise), row.names = FALSE, digits = 3)
  }
}

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0 || is.na(a)) b else a

root <- get_arg("--root", NULL)

if ("--all" %in% args) {
  report_all(root)
} else {
  report_run(get_arg("--run", "latest"), root)
}
