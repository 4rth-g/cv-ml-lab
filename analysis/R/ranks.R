#' Comparação de várias arquiteturas em vários datasets
#'
#' O arcabouço é o de Demšar (JMLR 2006): Friedman para detectar diferença
#' global entre arquiteturas usando cada dataset como bloco.
#'
#' O post-hoc, porém, NÃO é o Nemenyi que aquele artigo popularizou. Benavoli et
#' al. (JMLR 2016) mostrou que testes baseados em rank médio são
#' não-transitivos e dependem de quais outros algoritmos estão na comparação:
#' incluir um método ruim pode mudar a conclusão sobre dois bons. O padrão atual
#' é Wilcoxon pareado entre cada par, com correção de Holm.
#'
#' @name ranks
NULL

#' Matriz de desempenho: linhas = datasets (blocos), colunas = arquiteturas
#'
#' @param root Raiz de resultados.
#' @param split Split a usar.
#' @param arm Qual arm reportar (`"selected"` usa o que o pipeline escolheu).
#' @export
performance_matrix <- function(root = NULL, split = "test", arm = "selected") {
  idx <- read_index(root)

  rows <- purrr::pmap_dfr(
    list(idx$run_id, idx$model, idx$dataset, idx$subset, idx$selected_arm),
    function(run_id, model, dataset, subset, selected_arm) {
      run <- read_run(run_id, root)
      want <- if (identical(arm, "selected")) selected_arm else arm
      acc <- run$seed_metrics |>
        dplyr::filter(.data$split == !!split, .data$arm == want) |>
        dplyr::pull("acc")
      data.frame(
        dataset_id = if (is.na(subset) || subset == "") dataset else paste0(dataset, "/", subset),
        model = model,
        acc = mean(acc)
      )
    }
  )

  rows |>
    dplyr::group_by(.data$dataset_id, .data$model) |>
    dplyr::summarise(acc = mean(.data$acc), .groups = "drop") |>
    tidyr::pivot_wider(names_from = "model", values_from = "acc")
}

#' Friedman + post-hoc Wilcoxon-Holm
#'
#' @param mat Saída de [performance_matrix()].
#' @return Lista com `friedman`, `ranks` e `pairwise`.
#' @export
friedman_analysis <- function(mat) {
  values <- as.matrix(mat[, -1, drop = FALSE])
  rownames(values) <- mat[[1]]

  complete <- values[stats::complete.cases(values), , drop = FALSE]
  if (nrow(complete) < 2 || ncol(complete) < 2) {
    stop(
      "Friedman precisa de ao menos 2 datasets e 2 arquiteturas com resultado completo. ",
      "Rode mais combinações antes.",
      call. = FALSE
    )
  }

  fr <- stats::friedman.test(complete)

  # Rank médio: 1 = melhor. Descritivo apenas — a inferência vem do pairwise.
  ranks <- t(apply(-complete, 1, rank))
  mean_ranks <- data.frame(
    model = colnames(complete),
    mean_rank = colMeans(ranks),
    row.names = NULL
  )
  mean_ranks <- mean_ranks[order(mean_ranks$mean_rank), ]

  models <- colnames(complete)
  pairs <- utils::combn(models, 2, simplify = FALSE)
  pairwise <- purrr::map_dfr(pairs, function(p) {
    w <- suppressWarnings(stats::wilcox.test(complete[, p[1]], complete[, p[2]], paired = TRUE))
    data.frame(model_a = p[1], model_b = p[2], V = unname(w$statistic), p_value = w$p.value)
  })
  pairwise <- adjust_p(pairwise, "p_value", methods = c("holm"))

  list(
    n_datasets = nrow(complete),
    friedman = data.frame(
      statistic = unname(fr$statistic),
      df = unname(fr$parameter),
      p_value = fr$p.value
    ),
    ranks = mean_ranks,
    pairwise = pairwise[order(pairwise$p_holm), ]
  )
}

#' Tabela agregada de todos os runs, com p corrigido
#'
#' Uma linha por run, com o teste pareado e o p ajustado sobre a FAMÍLIA inteira
#' de comparações. É o número que deve ir para o README: reportar p bruto de
#' dezenas de comparações é o erro que esta camada existe para corrigir.
#'
#' @param root Raiz de resultados.
#' @param split Split a usar.
#' @export
all_runs_table <- function(root = NULL, split = "test") {
  idx <- read_index(root)

  tbl <- purrr::pmap_dfr(
    list(idx$run_id, idx$model, idx$dataset, idx$subset),
    function(run_id, model, dataset, subset) {
      run <- read_run(run_id, root)
      paired <- paired_accuracies(run$seed_metrics, split)
      tt <- paired_t(paired)
      data.frame(
        run_id = run_id,
        model = model,
        dataset = if (is.na(subset) || subset == "") dataset else paste0(dataset, "/", subset),
        n_seeds = tt$n_seeds,
        baseline = mean(paired$baseline),
        tuned = mean(paired$tuned),
        diff_pp = 100 * tt$diff_mean,
        ci_low_pp = 100 * tt$ci_low,
        ci_high_pp = 100 * tt$ci_high,
        cohens_dz = tt$cohens_dz,
        p_value = tt$p_value,
        prob_improvement = prob_improvement(paired)
      )
    }
  )

  adjust_p(tbl, "p_value", methods = c("holm", "BH"))
}
