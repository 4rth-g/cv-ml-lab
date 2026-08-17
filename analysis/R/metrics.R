#' Métricas calculadas a partir das predições exportadas
#'
#' O pipeline Python treina medindo apenas acurácia. Como as predições por
#' exemplo são exportadas, qualquer outra métrica vira cálculo pós-hoc — sobre
#' runs já feitos, sem retreinar e sem tocar no código de treino. A escolha da
#' métrica deixa de ser decisão de engenharia e passa a ser decisão de análise.
#'
#' Isso importa concretamente nos datasets médicos desbalanceados, onde acurácia
#' é métrica enganosa e a alternativa correta (F1 macro, balanced accuracy,
#' recall por classe) antes exigiria um novo treino.
#'
#' @name metrics
NULL

#' Matriz de confusão
#' @param y_true,y_pred Vetores de rótulos inteiros.
#' @param class_names Nomes das classes, na ordem dos índices.
#' @export
confusion_matrix <- function(y_true, y_pred, class_names = NULL) {
  levels_all <- sort(unique(c(y_true, y_pred)))
  cm <- table(
    true = factor(y_true, levels = levels_all),
    pred = factor(y_pred, levels = levels_all)
  )
  if (!is.null(class_names)) {
    labels <- class_names[levels_all + 1L]
    dimnames(cm) <- list(true = labels, pred = labels)
  }
  cm
}

#' Métricas por classe (precisão, recall, F1, suporte)
#' @inheritParams confusion_matrix
#' @export
per_class_metrics <- function(y_true, y_pred, class_names = NULL) {
  levels_all <- sort(unique(c(y_true, y_pred)))
  purrr::map_dfr(levels_all, function(k) {
    tp <- sum(y_true == k & y_pred == k)
    fp <- sum(y_true != k & y_pred == k)
    fn <- sum(y_true == k & y_pred != k)
    precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
    recall <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
    f1 <- if (!is.na(precision) && !is.na(recall) && (precision + recall) > 0) {
      2 * precision * recall / (precision + recall)
    } else {
      NA_real_
    }
    data.frame(
      class_idx = k,
      class_name = if (is.null(class_names)) as.character(k) else class_names[k + 1L],
      support = sum(y_true == k),
      precision = precision,
      recall = recall,
      f1 = f1
    )
  })
}

#' Métricas agregadas de uma predição
#'
#' `balanced_accuracy` é a média dos recalls por classe: é ela, e não a
#' acurácia, que responde "o modelo funciona nas classes raras?".
#'
#' @inheritParams confusion_matrix
#' @export
summary_metrics <- function(y_true, y_pred, class_names = NULL) {
  pc <- per_class_metrics(y_true, y_pred, class_names)
  data.frame(
    accuracy = mean(y_true == y_pred),
    balanced_accuracy = mean(pc$recall, na.rm = TRUE),
    f1_macro = mean(pc$f1, na.rm = TRUE),
    f1_weighted = stats::weighted.mean(pc$f1, pc$support, na.rm = TRUE),
    n = length(y_true)
  )
}

#' Kappa de Cohen com pesos quadráticos
#'
#' Métrica adequada a alvos ORDINAIS (ex.: RetinaMNIST, cujas classes são graus
#' de severidade). Acurácia trata errar por um grau e por quatro como o mesmo
#' erro, o que subestima modelos que erram "perto".
#'
#' @inheritParams confusion_matrix
#' @export
quadratic_weighted_kappa <- function(y_true, y_pred) {
  levels_all <- sort(unique(c(y_true, y_pred)))
  k <- length(levels_all)
  if (k < 2) {
    return(NA_real_)
  }

  obs <- table(factor(y_true, levels = levels_all), factor(y_pred, levels = levels_all))
  obs <- obs / sum(obs)
  expected <- outer(rowSums(obs), colSums(obs))

  weights <- outer(seq_len(k), seq_len(k), function(i, j) ((i - j)^2) / ((k - 1)^2))
  denom <- sum(weights * expected)
  if (denom == 0) {
    return(NA_real_)
  }
  1 - sum(weights * obs) / denom
}

#' Métricas de todas as combinações (arm, seed, split) de um run
#'
#' @param run Saída de [read_run()].
#' @param ordinal Se `TRUE`, acrescenta o kappa quadrático.
#' @export
run_metrics <- function(run, ordinal = FALSE) {
  class_names <- run$manifest$class_names
  run$predictions |>
    dplyr::group_by(.data$arm, .data$seed_init, .data$split) |>
    dplyr::group_modify(function(g, key) {
      out <- summary_metrics(g$y_true, g$y_pred, class_names)
      if (ordinal) {
        out$qwk <- quadratic_weighted_kappa(g$y_true, g$y_pred)
      }
      out
    }) |>
    dplyr::ungroup()
}

#' Exemplos que TODAS as seeds erram, por arm
#'
#' Separa dificuldade intrínseca do exemplo de instabilidade do treino: um
#' exemplo que só algumas seeds erram é ruído de otimização, um que todas erram é
#' problema do dado ou do modelo. É uma pergunta que exige as predições de todas
#' as seeds, e por isso não existia antes do export.
#'
#' @param run Saída de [read_run()].
#' @param split Split a analisar.
#' @export
consistently_wrong <- function(run, split = "test") {
  run$predictions |>
    dplyr::filter(.data$split == !!split) |>
    dplyr::group_by(.data$arm, .data$example_idx, .data$y_true) |>
    dplyr::summarise(
      n_seeds = dplyr::n(),
      n_wrong = sum(.data$y_true != .data$y_pred),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      always_wrong = .data$n_wrong == .data$n_seeds,
      sometimes_wrong = .data$n_wrong > 0 & .data$n_wrong < .data$n_seeds
    )
}
