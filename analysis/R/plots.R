#' Gráficos do relatório
#'
#' Todo gráfico do cvlab é feito aqui, em ggplot2. O Python calcula e o R
#' desenha, para dataset e para resultado igualmente: metade das figuras em
#' matplotlib e metade em ggplot2 daria duas identidades visuais ao mesmo
#' relatório.
#'
#' Cada função recebe data frames e devolve um objeto ggplot, sem escrever
#' arquivo nem depender de estado global.
#'
#' @name plots
NULL

#' Tema comum do relatório
#' @export
theme_cvlab <- function(base_size = 11) {
  ggplot2::theme_minimal(base_size = base_size) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(face = "bold", size = base_size * 1.1),
      plot.subtitle = ggplot2::element_text(color = "grey35", size = base_size * 0.9),
      plot.caption = ggplot2::element_text(color = "grey45", hjust = 0, size = base_size * 0.8),
      panel.grid.minor = ggplot2::element_blank(),
      legend.position = "bottom"
    )
}

ARM_COLORS <- c(baseline = "#7f7f7f", tuned = "#1f77b4")

#' Acurácia por seed, com o pareamento explícito
#'
#' Cada linha liga o MESMO par de sementes nos dois arms. Desenhar o pareamento
#' não é decoração: é o que mostra visualmente por que o teste pareado tem mais
#' poder que um teste independente, e revela quando o ganho é consistente entre
#' seeds em vez de vir de uma execução de sorte.
#'
#' @param seed_metrics Data frame de `seed_metrics.csv`.
#' @param split Split a exibir.
#' @export
plot_paired_seeds <- function(seed_metrics, split = "test") {
  d <- seed_metrics[seed_metrics$split == split, ]
  d$arm <- factor(d$arm, levels = c("baseline", "tuned"))

  ggplot2::ggplot(d, ggplot2::aes(x = .data$arm, y = .data$acc)) +
    ggplot2::geom_line(
      ggplot2::aes(group = .data$seed_init),
      color = "grey60", linewidth = 0.4
    ) +
    ggplot2::geom_point(ggplot2::aes(color = .data$arm), size = 2.6) +
    ggplot2::stat_summary(
      fun = mean, geom = "crossbar", width = 0.35,
      color = "grey20", linewidth = 0.3, fatten = 1
    ) +
    ggplot2::scale_color_manual(values = ARM_COLORS, guide = "none") +
    ggplot2::scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
    ggplot2::labs(
      title = glue::glue("Acurácia de {split} por seed"),
      subtitle = "cada linha liga o mesmo par de sementes; a barra é a média",
      x = NULL, y = "acurácia"
    ) +
    theme_cvlab()
}

#' Intervalo de confiança da diferença pareada
#'
#' Um forest plot de uma linha só, mas com a referência em zero desenhada: é a
#' leitura que responde "o efeito é grande?", que o p-valor não responde.
#'
#' @param tt Saída de [paired_t()].
#' @export
plot_difference_ci <- function(tt) {
  if (is.na(tt$ci_low)) {
    return(NULL)
  }
  d <- data.frame(
    label = "tunada - baseline",
    est = 100 * tt$diff_mean,
    lo = 100 * tt$ci_low,
    hi = 100 * tt$ci_high
  )

  ggplot2::ggplot(d, ggplot2::aes(x = .data$est, y = .data$label)) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed", color = "grey50") +
    ggplot2::geom_errorbarh(
      ggplot2::aes(xmin = .data$lo, xmax = .data$hi),
      height = 0.12, color = "#1f77b4", linewidth = 0.7
    ) +
    ggplot2::geom_point(size = 3, color = "#1f77b4") +
    ggplot2::labs(
      title = "Diferença de acurácia com IC95%",
      subtitle = "se o intervalo cruza o zero, a direção do efeito não está estabelecida",
      x = "pontos percentuais", y = NULL
    ) +
    theme_cvlab()
}

#' Erro por classe
#'
#' @param joined Saída de [join_predictions_eda()].
#' @param split Split a exibir.
#' @export
plot_error_by_class <- function(joined, split = "test") {
  d <- joined |>
    dplyr::filter(.data$split == !!split) |>
    dplyr::group_by(.data$arm, .data$class_name) |>
    dplyr::summarise(erro = 1 - mean(.data$correct), n = dplyr::n(), .groups = "drop")

  ggplot2::ggplot(d, ggplot2::aes(
    x = stats::reorder(.data$class_name, .data$erro),
    y = .data$erro, fill = .data$arm
  )) +
    ggplot2::geom_col(position = ggplot2::position_dodge(width = 0.75), width = 0.7) +
    ggplot2::coord_flip() +
    ggplot2::scale_fill_manual(values = ARM_COLORS, name = NULL) +
    ggplot2::scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    ggplot2::labs(
      title = "Taxa de erro por classe",
      subtitle = "a média global esconde classes em que o modelo falha por completo",
      x = NULL, y = "erro"
    ) +
    theme_cvlab()
}

#' Matriz de confusão
#'
#' @param cm Saída de [confusion_matrix()].
#' @export
plot_confusion <- function(cm) {
  d <- as.data.frame(cm, stringsAsFactors = FALSE)
  names(d) <- c("true", "pred", "n")
  d <- d |>
    dplyr::group_by(.data$true) |>
    dplyr::mutate(prop = .data$n / sum(.data$n)) |>
    dplyr::ungroup()

  ggplot2::ggplot(d, ggplot2::aes(x = .data$pred, y = .data$true, fill = .data$prop)) +
    ggplot2::geom_tile(color = "white", linewidth = 0.4) +
    ggplot2::geom_text(
      ggplot2::aes(label = .data$n, color = .data$prop > 0.5),
      size = 3, show.legend = FALSE
    ) +
    ggplot2::scale_color_manual(values = c(`TRUE` = "white", `FALSE` = "grey25")) +
    ggplot2::scale_fill_gradient(
      low = "#f5f5f5", high = "#1f77b4",
      labels = scales::percent_format(accuracy = 1), name = "por linha"
    ) +
    ggplot2::labs(
      title = "Matriz de confusão",
      subtitle = "cor normalizada por linha (recall); número é a contagem",
      x = "predito", y = "verdadeiro"
    ) +
    theme_cvlab() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
}

#' Curva de desempenho esperado por orçamento de busca
#'
#' Reportar o melhor trial de uma busca como se fosse o desempenho do método é
#' um estimador enviesado para cima: quanto mais trials, maior o máximo, mesmo
#' sem nenhum ganho real. A curva mostra o que se esperaria com k trials
#' (Dodge et al., EMNLP 2019). Se ela ainda sobe no fim, o orçamento não saturou;
#' se achatou, gastar mais busca não compra nada.
#'
#' @param trials Data frame de `trials.csv`.
#' @param R Reamostragens por ponto.
#' @export
plot_evp_curve <- function(trials, R = 500) {
  vals <- trials$value[!is.na(trials$value) & trials$state == "COMPLETE"]
  n <- length(vals)
  if (n < 3) {
    return(NULL)
  }

  # E[max de k trials] por reamostragem da ORDEM da busca, sem reposição.
  curve <- purrr::map_dfr(seq_len(n), function(k) {
    maxes <- replicate(R, max(sample(vals, k)))
    data.frame(
      k = k,
      evp = mean(maxes),
      lo = stats::quantile(maxes, 0.25),
      hi = stats::quantile(maxes, 0.75)
    )
  })

  ggplot2::ggplot(curve, ggplot2::aes(x = .data$k, y = .data$evp)) +
    ggplot2::geom_ribbon(
      ggplot2::aes(ymin = .data$lo, ymax = .data$hi),
      fill = "#1f77b4", alpha = 0.15
    ) +
    ggplot2::geom_line(color = "#1f77b4", linewidth = 0.7) +
    ggplot2::geom_point(color = "#1f77b4", size = 1.6) +
    ggplot2::scale_y_continuous(labels = scales::percent_format(accuracy = 0.1)) +
    ggplot2::labs(
      title = "Desempenho esperado por orçamento de busca",
      subtitle = "faixa = quartis; reportar o máximo de n trials superestima o método",
      x = "trials", y = "val_acc esperada do melhor trial"
    ) +
    theme_cvlab()
}

#' Erro em função de uma propriedade da imagem
#'
#' Só possível porque o EDA e as predições compartilham `example_idx`. Responde
#' se o modelo falha num regime específico de imagem, e não apenas "quanto" ele
#' erra.
#'
#' @param joined Saída de [join_predictions_eda()].
#' @param var Coluna do EDA a usar no eixo x.
#' @param split Split a exibir.
#' @param bins Número de faixas.
#' @export
plot_error_vs_feature <- function(joined, var = "mean_intensity", split = "test", bins = 10) {
  d <- joined |>
    dplyr::filter(.data$split == !!split) |>
    dplyr::mutate(faixa = dplyr::ntile(.data[[var]], bins)) |>
    dplyr::group_by(.data$arm, .data$faixa) |>
    dplyr::summarise(
      erro = 1 - mean(.data$correct),
      centro = mean(.data[[var]]),
      n = dplyr::n(),
      .groups = "drop"
    )

  ggplot2::ggplot(d, ggplot2::aes(x = .data$centro, y = .data$erro, color = .data$arm)) +
    ggplot2::geom_line(linewidth = 0.6) +
    ggplot2::geom_point(ggplot2::aes(size = .data$n)) +
    ggplot2::scale_color_manual(values = ARM_COLORS, name = NULL) +
    ggplot2::scale_size_continuous(guide = "none") +
    ggplot2::scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    ggplot2::labs(
      title = glue::glue("Erro por faixa de {var}"),
      subtitle = "cruzamento só possível pela chave (split, example_idx)",
      x = var, y = "erro"
    ) +
    theme_cvlab()
}

#' Consistência do erro entre seeds
#'
#' Separa o que é difícil do que é instável: um exemplo que todas as seeds erram
#' é limitação do dado ou do modelo; um que só algumas erram é ruído de
#' otimização. Tratar os dois como o mesmo "erro" confunde os diagnósticos.
#'
#' @param cw Saída de [consistently_wrong()].
#' @export
plot_error_consistency <- function(cw) {
  d <- cw |>
    dplyr::group_by(.data$arm) |>
    dplyr::summarise(
      `sempre erram` = sum(.data$always_wrong),
      `às vezes erram` = sum(.data$sometimes_wrong),
      `sempre acertam` = sum(.data$n_wrong == 0),
      .groups = "drop"
    ) |>
    tidyr::pivot_longer(-"arm", names_to = "categoria", values_to = "n")

  d$categoria <- factor(d$categoria, levels = c("sempre acertam", "às vezes erram", "sempre erram"))

  ggplot2::ggplot(d, ggplot2::aes(x = .data$arm, y = .data$n, fill = .data$categoria)) +
    ggplot2::geom_col(position = "fill", width = 0.6) +
    ggplot2::scale_fill_manual(values = c("#d9d9d9", "#fdae61", "#d73027"), name = NULL) +
    ggplot2::scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
    ggplot2::labs(
      title = "Consistência do erro entre seeds",
      subtitle = "vermelho = dificuldade do dado; laranja = instabilidade de treino",
      x = NULL, y = NULL
    ) +
    theme_cvlab()
}
