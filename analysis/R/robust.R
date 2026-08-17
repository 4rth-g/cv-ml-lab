#' Estimativas robustas com poucas execuções
#'
#' Com 5 seeds, a média é dominada por um outlier e a mediana descarta metade da
#' informação. As alternativas aqui seguem Agarwal et al. (NeurIPS 2021), que
#' propôs relatar intervalo em vez de ponto e usar agregados robustos quando o
#' número de execuções é pequeno.
#'
#' @name robust
NULL

#' Média interquartil (IQM)
#'
#' Descarta os 25% extremos de cada lado e promedia o restante. Mais estável que
#' a média com poucas amostras e mais eficiente que a mediana.
#'
#' @param x Vetor numérico.
#' @export
iqm <- function(x) {
  x <- sort(x[!is.na(x)])
  n <- length(x)
  if (n == 0) {
    return(NA_real_)
  }
  if (n < 4) {
    return(mean(x))
  }
  lo <- floor(n * 0.25) + 1
  hi <- ceiling(n * 0.75)
  mean(x[lo:hi])
}

#' Intervalo de confiança por bootstrap
#'
#' Usa BCa quando possível (corrige viés e assimetria) e cai para percentil
#' quando a amostra é pequena demais para o BCa ser estimável, o que é comum com
#' 5 seeds. A degradação é explícita na coluna `type` em vez de silenciosa.
#'
#' @param x Vetor numérico.
#' @param statistic Função de agregação; default [iqm()].
#' @param R Número de reamostragens.
#' @param conf Nível de confiança.
#' @export
bootstrap_ci <- function(x, statistic = iqm, R = 2000, conf = 0.95) {
  x <- x[!is.na(x)]
  if (length(x) < 2 || stats::sd(x) == 0) {
    est <- statistic(x)
    return(data.frame(estimate = est, ci_low = est, ci_high = est, type = "degenerate"))
  }

  boot_fn <- function(data, idx) statistic(data[idx])
  bs <- boot::boot(x, boot_fn, R = R)

  ci <- tryCatch(boot::boot.ci(bs, conf = conf, type = "bca"), error = function(e) NULL)
  type <- "bca"
  if (is.null(ci)) {
    ci <- tryCatch(boot::boot.ci(bs, conf = conf, type = "perc"), error = function(e) NULL)
    type <- "percentile"
  }
  if (is.null(ci)) {
    est <- statistic(x)
    return(data.frame(estimate = est, ci_low = NA_real_, ci_high = NA_real_, type = "failed"))
  }

  bounds <- if (type == "bca") ci$bca[4:5] else ci$percent[4:5]
  data.frame(estimate = bs$t0, ci_low = bounds[1], ci_high = bounds[2], type = type)
}

#' Probabilidade de melhoria
#'
#' P(tunada > baseline) sobre os pares de seed, com empate contando meio. É uma
#' resposta direta a "vale trocar?", que o p-valor não dá: p mede evidência
#' contra a hipótese nula, não a chance de a mudança ajudar.
#'
#' @param paired Saída de [paired_accuracies()].
#' @export
prob_improvement <- function(paired) {
  wins <- sum(paired$tuned > paired$baseline)
  ties <- sum(paired$tuned == paired$baseline)
  (wins + 0.5 * ties) / nrow(paired)
}

#' Bootstrap da diferença de acurácia no nível do EXEMPLO
#'
#' Reamostra `example_idx`, não seeds. Responde uma pergunta diferente do
#' bootstrap por seed: quanta incerteza vem do conjunto de teste ser finito, e
#' não de o treino ser estocástico. Só é possível porque as predições por
#' exemplo são exportadas.
#'
#' @param run Saída de [read_run()].
#' @param split Split a usar.
#' @param R Reamostragens.
#' @export
bootstrap_example_diff <- function(run, split = "test", R = 2000) {
  preds <- run$predictions
  base_seed <- run$manifest$mcnemar_baseline_seed
  tuned_seed <- run$manifest$mcnemar_tuned_seed

  a <- preds |>
    dplyr::filter(.data$arm == "baseline", .data$split == !!split, .data$seed_init == base_seed) |>
    dplyr::arrange(.data$example_idx)
  b <- preds |>
    dplyr::filter(.data$arm == "tuned", .data$split == !!split, .data$seed_init == tuned_seed) |>
    dplyr::arrange(.data$example_idx)

  delta <- as.integer(b$y_true == b$y_pred) - as.integer(a$y_true == a$y_pred)
  ci <- bootstrap_ci(delta, statistic = mean, R = R)
  ci$split <- split
  ci
}

#' Resumo robusto de um run
#' @param run Saída de [read_run()].
#' @param split Split a resumir.
#' @export
robust_summary <- function(run, split = "test") {
  paired <- paired_accuracies(run$seed_metrics, split)
  base_ci <- bootstrap_ci(paired$baseline)
  tuned_ci <- bootstrap_ci(paired$tuned)

  data.frame(
    split = split,
    n_seeds = nrow(paired),
    baseline_iqm = base_ci$estimate,
    baseline_ci_low = base_ci$ci_low,
    baseline_ci_high = base_ci$ci_high,
    tuned_iqm = tuned_ci$estimate,
    tuned_ci_low = tuned_ci$ci_low,
    tuned_ci_high = tuned_ci$ci_high,
    prob_improvement = prob_improvement(paired),
    ci_type = base_ci$type
  )
}
