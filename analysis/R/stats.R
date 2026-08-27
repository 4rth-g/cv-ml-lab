#' Inferência pareada sobre os resultados multi-seed
#'
#' Contém a paridade com o que saía do Python (teste t pareado, Cohen's d_z,
#' IC95%, McNemar) e o que só existe aqui (IC do tamanho de efeito, correção
#' para múltiplas comparações, equivalência).
#'
#' Nenhuma destas funções decide coisa alguma sobre o modelo: a seleção
#' baseline-vs-tunada acontece no Python, na validação, antes de qualquer
#' p-valor existir. Aqui é só reportagem.
#'
#' @name stats
NULL

#' Acurácias pareadas por seed
#'
#' @param seed_metrics Data frame de `seed_metrics.csv`.
#' @param split `"test"` ou `"val"`.
#' @return Data frame com uma linha por seed e colunas `baseline`, `tuned`, `diff`.
#' @export
paired_accuracies <- function(seed_metrics, split = "test") {
  wide <- seed_metrics |>
    dplyr::filter(.data$split == !!split) |>
    dplyr::select("seed_init", "arm", "acc") |>
    tidyr::pivot_wider(names_from = "arm", values_from = "acc") |>
    dplyr::arrange(.data$seed_init)

  if (!all(c("baseline", "tuned") %in% names(wide))) {
    stop("seed_metrics não tem os dois arms (baseline e tuned).", call. = FALSE)
  }
  # Pareamento por seed: é o que cancela o ruído comum às duas configs e dá
  # poder ao teste. Uma seed presente num arm e ausente no outro quebraria isso.
  if (any(is.na(wide$baseline)) || any(is.na(wide$tuned))) {
    stop("Há seeds sem par baseline/tunada — o teste pareado não se aplica.", call. = FALSE)
  }

  dplyr::mutate(wide, diff = .data$tuned - .data$baseline)
}

#' Teste t pareado com IC e tamanho de efeito
#'
#' Paridade com `scipy.stats.ttest_rel` mais `stats.t.ppf(0.975, n-1)`, que o
#' Python calculava à mão. Aqui o IC sai do próprio `t.test()`, e o Cohen's d_z
#' vem com intervalo — que o Python não fornecia e sem o qual o tamanho de efeito
#' é um número solto.
#'
#' @param paired Saída de [paired_accuracies()].
#' @return Lista de uma linha (data frame) com `t`, `df`, `p_value`, `diff_mean`,
#'   `ci_low`, `ci_high`, `cohens_dz` e o IC do efeito.
#' @export
paired_t <- function(paired) {
  n <- nrow(paired)
  if (n < 2) {
    stop("Teste pareado exige ao menos 2 seeds.", call. = FALSE)
  }

  # Diferenças todas iguais (inclusive todas zero) fazem o desvio ser 0 e o
  # t.test() falhar. É um cenário real em orçamento smoke, quando a busca devolve
  # a própria baseline; devolver NA é mais honesto que estourar.
  if (stats::sd(paired$diff) == 0) {
    return(data.frame(
      n_seeds = n, t = NA_real_, df = n - 1L, p_value = NA_real_,
      diff_mean = mean(paired$diff), ci_low = NA_real_, ci_high = NA_real_,
      cohens_dz = NA_real_, dz_ci_low = NA_real_, dz_ci_high = NA_real_
    ))
  }

  tt <- stats::t.test(paired$tuned, paired$baseline, paired = TRUE)
  # suppressMessages: o effectsize imprime uma sugestão sobre repeated_measures_d()
  # a cada chamada, que polui o relatório sem acrescentar nada à análise.
  d <- suppressMessages(
    effectsize::cohens_d(paired$tuned, paired$baseline, paired = TRUE, ci = 0.95)
  )

  data.frame(
    n_seeds = n,
    t = unname(tt$statistic),
    df = unname(tt$parameter),
    p_value = tt$p.value,
    diff_mean = unname(tt$estimate),
    ci_low = tt$conf.int[1],
    ci_high = tt$conf.int[2],
    cohens_dz = d$Cohens_d,
    dz_ci_low = d$CI_low,
    dz_ci_high = d$CI_high
  )
}

#' Wilcoxon signed-rank pareado
#'
#' Checagem de robustez do t. Com n = 5 seeds a normalidade das diferenças é
#' indemonstrável, então concordância entre os dois testes vale mais que o
#' p-valor de qualquer um deles isolado.
#'
#' @param paired Saída de [paired_accuracies()].
#' @export
paired_wilcoxon <- function(paired) {
  if (all(paired$diff == 0)) {
    return(data.frame(V = NA_real_, p_value = NA_real_))
  }
  w <- suppressWarnings(stats::wilcox.test(paired$tuned, paired$baseline, paired = TRUE))
  data.frame(V = unname(w$statistic), p_value = w$p.value)
}

#' Teste de McNemar entre dois classificadores no mesmo conjunto
#'
#' Paridade exata com o `mcnemar_test` que existia em `cvlab/tracking.py`,
#' inclusive o limiar de 25 discordâncias: abaixo dele, binomial exata; acima,
#' qui-quadrado com correção de continuidade. Manter a mesma regra é o que
#' permite comparar runs novos com os números já publicados no README.
#'
#' @param y_true,pred_a,pred_b Vetores de mesmo comprimento.
#' @return Data frame com `n01`, `n10`, `statistic`, `p_value`, `method`.
#' @export
mcnemar_pair <- function(y_true, pred_a, pred_b) {
  stopifnot(length(y_true) == length(pred_a), length(y_true) == length(pred_b))

  correct_a <- y_true == pred_a
  correct_b <- y_true == pred_b
  n01 <- sum(!correct_a & correct_b)
  n10 <- sum(correct_a & !correct_b)
  disc <- n01 + n10

  if (disc < 25) {
    statistic <- min(n01, n10)
    p_value <- if (disc == 0) 1 else min(1, 2 * stats::pbinom(statistic, disc, 0.5))
    method <- "binom_exact"
  } else {
    statistic <- (abs(n01 - n10) - 1)^2 / disc
    p_value <- stats::pchisq(statistic, df = 1, lower.tail = FALSE)
    method <- "chi2_continuity"
  }

  data.frame(
    n01 = as.integer(n01), n10 = as.integer(n10),
    statistic = as.numeric(statistic), p_value = as.numeric(p_value),
    method = method
  )
}

#' McNemar do par que o pipeline selecionaria
#'
#' Usa, de cada arm, a seed de melhor VALIDAÇÃO — registrada no manifest pelo
#' Python. Escolher pela melhor acurácia de teste seria vazamento: o teste
#' entraria na escolha do que reportar.
#'
#' @param run Saída de [read_run()].
#' @export
mcnemar_from_run <- function(run) {
  preds <- run$predictions
  base_seed <- run$manifest$mcnemar_baseline_seed
  tuned_seed <- run$manifest$mcnemar_tuned_seed

  a <- preds |>
    dplyr::filter(.data$arm == "baseline", .data$split == "test", .data$seed_init == base_seed) |>
    dplyr::arrange(.data$example_idx)
  b <- preds |>
    dplyr::filter(.data$arm == "tuned", .data$split == "test", .data$seed_init == tuned_seed) |>
    dplyr::arrange(.data$example_idx)

  if (nrow(a) == 0 || nrow(b) == 0) {
    stop("Predições de teste ausentes para as seeds indicadas no manifest.", call. = FALSE)
  }
  stopifnot(identical(a$y_true, b$y_true))

  mcnemar_pair(a$y_true, a$y_pred, b$y_pred)
}

#' Correção para múltiplas comparações
#'
#' A dívida mais grave do repo antes desta camada: as tabelas do README
#' reportavam vários testes sem correção nenhuma. Com dezenas de combinações
#' dataset x arquitetura, algum p < 0.05 aparece por acaso praticamente sempre.
#'
#' Holm controla a taxa de erro por família (conservador, sem supor
#' independência); BH controla a taxa de falsas descobertas e é o adequado
#' quando a varredura é exploratória.
#'
#' @param df Data frame com uma coluna de p-valores.
#' @param p_col Nome da coluna.
#' @param methods Métodos a aplicar.
#' @export
adjust_p <- function(df, p_col = "p_value", methods = c("holm", "BH")) {
  for (m in methods) {
    df[[paste0("p_", m)]] <- stats::p.adjust(df[[p_col]], method = m)
  }
  df
}

#' Teste de equivalência (TOST) sobre a diferença pareada
#'
#' Com poucas seeds, "não significativo" é o resultado mais provável e não
#' significa "equivalente". O TOST responde a pergunta que o teste t nunca
#' responde: a diferença é pequena o bastante para ser irrelevante?
#'
#' @param paired Saída de [paired_accuracies()].
#' @param bound Margem de equivalência em pontos de acurácia (default 1 p.p.).
#' @export
equivalence_tost <- function(paired, bound = 0.01) {
  if (!requireNamespace("TOSTER", quietly = TRUE)) {
    return(NULL)
  }
  if (nrow(paired) < 2 || stats::sd(paired$diff) == 0) {
    return(NULL)
  }
  res <- TOSTER::t_TOST(
    x = paired$tuned, y = paired$baseline, paired = TRUE,
    low_eqbound = -bound, high_eqbound = bound, eqbound_type = "raw"
  )
  data.frame(
    bound = bound,
    p_tost = max(res$TOST$p.value[2], res$TOST$p.value[3]),
    decision = res$decision$TOST
  )
}
