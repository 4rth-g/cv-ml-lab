#' Métricas que dependem de probabilidade, não só da classe predita
#'
#' AUC, ROC, precisão-recall e calibração não são recuperáveis a partir do
#' argmax: precisam da distribuição sobre as classes. O pipeline exporta isso em
#' `probabilities.csv.gz`, e é o que torna os resultados comparáveis com o
#' benchmark oficial do MedMNIST, que reporta AUC junto da acurácia.
#'
#' Tudo aqui é implementado em R base. AUC sai da identidade com Mann-Whitney,
#' que é exata e trata empates pela média dos ranks, então não há dependência
#' nova nem aproximação por interpolação de curva.
#'
#' @name probs
NULL

#' AUC binária pela estatística de Mann-Whitney
#'
#' @param labels Vetor 0/1 (ou lógico) com a classe positiva.
#' @param scores Escore contínuo (probabilidade da classe positiva).
#' @return AUC, ou `NA` se só houver uma classe presente.
#' @export
auc_binary <- function(labels, scores) {
  y <- as.integer(as.logical(labels))
  ok <- !is.na(y) & !is.na(scores)
  y <- y[ok]
  scores <- scores[ok]

  n_pos <- sum(y == 1)
  n_neg <- sum(y == 0)
  # Sem as duas classes a AUC é indefinida. Retornar NA é mais honesto que 0.5,
  # que sugeriria "aleatório" onde na verdade não há o que medir.
  if (n_pos == 0 || n_neg == 0) {
    return(NA_real_)
  }

  r <- rank(scores, ties.method = "average")
  (sum(r[y == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

#' AUC macro one-vs-rest
#'
#' Média das AUC por classe, cada uma tratando a classe como positiva e as demais
#' como negativas. É a definição que o MedMNIST usa para os subsets multiclasse.
#'
#' @param probs Data frame longo com `example_idx`, `class_idx`, `prob`.
#' @param truth Data frame com `example_idx`, `y_true`.
#' @return Data frame com a AUC por classe e a média macro.
#' @export
auc_macro_ovr <- function(probs, truth) {
  joined <- dplyr::left_join(probs, truth, by = "example_idx")
  per_class <- joined |>
    dplyr::group_by(.data$class_idx) |>
    dplyr::summarise(
      n_pos = sum(.data$y_true == .data$class_idx),
      auc = auc_binary(.data$y_true == .data$class_idx, .data$prob),
      .groups = "drop"
    )
  list(per_class = per_class, macro = mean(per_class$auc, na.rm = TRUE))
}

#' Probabilidades de um (arm, seed, split) em formato longo
#'
#' @param run Saída de [read_run()].
#' @param arm,split,seed Filtros; `seed = NULL` usa a de melhor validação do arm.
#' @export
probs_of <- function(run, arm = NULL, split = "test", seed = NULL) {
  if (is.null(run$probabilities)) {
    stop(
      "Este run não tem probabilities.csv.gz. Foi exportado por uma versão do ",
      "cvlab anterior ao export de probabilidades; retreine para ter AUC e calibração.",
      call. = FALSE
    )
  }
  arm <- arm %||% run$manifest$selected_arm
  if (is.null(seed)) {
    seed <- if (arm == "tuned") run$manifest$mcnemar_tuned_seed else run$manifest$mcnemar_baseline_seed
  }
  dplyr::filter(
    run$probabilities,
    .data$arm == !!arm, .data$split == !!split, .data$seed_init == !!seed
  )
}

#' Rótulos verdadeiros de um split, uma linha por exemplo
#' @param run Saída de [read_run()].
#' @param split Split desejado.
#' @export
truth_of <- function(run, split = "test") {
  run$predictions |>
    dplyr::filter(.data$split == !!split) |>
    dplyr::distinct(.data$example_idx, .data$y_true)
}

#' Curva ROC
#'
#' @param labels Vetor 0/1.
#' @param scores Escore da classe positiva.
#' @return Data frame com `fpr`, `tpr` e o limiar.
#' @export
roc_points <- function(labels, scores) {
  y <- as.integer(as.logical(labels))
  ord <- order(scores, decreasing = TRUE)
  y <- y[ord]
  tp <- cumsum(y)
  fp <- cumsum(1 - y)
  data.frame(
    threshold = c(Inf, scores[ord]),
    tpr = c(0, tp / max(1, sum(y))),
    fpr = c(0, fp / max(1, sum(1 - y)))
  )
}

#' Calibração por bins, com ECE
#'
#' Compara a confiança declarada com a taxa de acerto observada. Um modelo pode
#' ter boa acurácia e ser mal calibrado, dizendo "95% de certeza" onde acerta
#' 70% — o que importa em domínio médico, onde a probabilidade costuma alimentar
#' uma decisão de encaminhamento.
#'
#' @param correct Vetor lógico: a predição estava certa?
#' @param confidence Probabilidade da classe predita.
#' @param bins Número de faixas de confiança.
#' @return Lista com o data frame por bin e o ECE (expected calibration error).
#' @export
calibration_bins <- function(correct, confidence, bins = 10) {
  brk <- seq(0, 1, length.out = bins + 1)
  faixa <- cut(confidence, breaks = brk, include.lowest = TRUE)
  d <- data.frame(correct = as.integer(correct), confidence = confidence, faixa = faixa) |>
    dplyr::group_by(.data$faixa) |>
    dplyr::summarise(
      n = dplyr::n(),
      conf_media = mean(.data$confidence),
      acerto = mean(.data$correct),
      .groups = "drop"
    ) |>
    dplyr::filter(.data$n > 0)

  ece <- sum(d$n / sum(d$n) * abs(d$acerto - d$conf_media))
  list(bins = d, ece = ece)
}

#' Confiança e acerto por exemplo, prontos para calibração
#' @param run Saída de [read_run()].
#' @inheritParams probs_of
#' @export
confidence_table <- function(run, arm = NULL, split = "test", seed = NULL) {
  p <- probs_of(run, arm, split, seed)
  truth <- truth_of(run, split)
  p |>
    dplyr::group_by(.data$example_idx) |>
    dplyr::slice_max(.data$prob, n = 1, with_ties = FALSE) |>
    dplyr::ungroup() |>
    dplyr::left_join(truth, by = "example_idx") |>
    dplyr::mutate(correct = .data$class_idx == .data$y_true) |>
    dplyr::rename(confidence = "prob", y_pred = "class_idx")
}
