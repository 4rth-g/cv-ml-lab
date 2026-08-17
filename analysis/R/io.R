#' Leitura dos artefatos exportados pelo pipeline Python
#'
#' Este arquivo é o ÚNICO ponto do pacote que conhece o layout de `results/`.
#' Todo o resto opera sobre data frames, o que mantém a fronteira entre as duas
#' linguagens num contrato de arquivos e não numa chamada de código.
#'
#' @keywords internal
#' @name io
NULL

#' Versão de schema que este pacote sabe ler
#'
#' Incrementada no Python (`cvlab.export.SCHEMA_VERSION`) sempre que uma coluna
#' muda. Recusar o desconhecido é melhor que interpretar errado em silêncio.
#' @export
SUPPORTED_SCHEMA_VERSION <- 1L

#' Localiza a raiz de resultados
#'
#' @param path Caminho para `results/`. O default sobe a partir do diretório de
#'   trabalho, para que os scripts funcionem tanto da raiz do repo quanto de
#'   dentro de `analysis/`.
#' @export
results_root <- function(path = NULL) {
  if (!is.null(path)) {
    return(normalizePath(path, mustWork = TRUE))
  }
  candidates <- c("results", file.path("..", "results"))
  for (candidate in candidates) {
    if (dir.exists(candidate)) {
      return(normalizePath(candidate))
    }
  }
  stop(
    "Diretório 'results/' não encontrado. Rode a partir da raiz do repositório, ",
    "ou passe o caminho explicitamente.",
    call. = FALSE
  )
}

#' Índice de todos os runs exportados
#'
#' O `index.csv` é append-only e existe justamente para habilitar a análise
#' ENTRE runs (Friedman sobre vários datasets) sem abrir N manifests.
#'
#' @param root Raiz de resultados; ver [results_root()].
#' @return Um data frame, um run por linha.
#' @export
read_index <- function(root = NULL) {
  root <- results_root(root)
  path <- file.path(root, "runs", "index.csv")
  if (!file.exists(path)) {
    stop(
      glue::glue("Índice não encontrado em {path}. Rode um treino antes (`just train`)."),
      call. = FALSE
    )
  }
  readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
}

#' Identificador do run mais recente
#' @param root Raiz de resultados.
#' @export
latest_run_id <- function(root = NULL) {
  idx <- read_index(root)
  if (nrow(idx) == 0) {
    stop("Nenhum run registrado no índice.", call. = FALSE)
  }
  idx$run_id[which.max(as.POSIXct(idx$timestamp_utc, tz = "UTC"))]
}

#' Lê um run completo
#'
#' @param run_id Identificador do run; `NULL` usa o mais recente.
#' @param root Raiz de resultados.
#' @return Lista com `manifest`, `seed_metrics`, `predictions` e `trials`.
#' @export
read_run <- function(run_id = NULL, root = NULL) {
  root <- results_root(root)
  if (is.null(run_id) || identical(run_id, "latest")) {
    run_id <- latest_run_id(root)
  }

  run_dir <- file.path(root, "runs", run_id)
  if (!dir.exists(run_dir)) {
    stop(glue::glue("Run '{run_id}' não encontrado em {run_dir}."), call. = FALSE)
  }

  manifest <- jsonlite::fromJSON(file.path(run_dir, "manifest.json"), simplifyVector = TRUE)
  check_schema(manifest$schema_version, run_id)

  trials_path <- file.path(run_dir, "trials.csv")

  list(
    run_id = run_id,
    dir = run_dir,
    manifest = manifest,
    seed_metrics = readr::read_csv(
      file.path(run_dir, "seed_metrics.csv"),
      show_col_types = FALSE, progress = FALSE
    ),
    predictions = readr::read_csv(
      file.path(run_dir, "predictions.csv.gz"),
      show_col_types = FALSE, progress = FALSE
    ),
    trials = if (file.exists(trials_path)) {
      readr::read_csv(trials_path, show_col_types = FALSE, progress = FALSE)
    } else {
      NULL
    }
  )
}

#' Valida a versão de schema de um artefato
#' @param version Valor de `schema_version` lido do arquivo.
#' @param what Nome do artefato, para a mensagem de erro.
#' @export
check_schema <- function(version, what = "artefato") {
  if (is.null(version) || is.na(version)) {
    stop(glue::glue("'{what}' não declara schema_version — export antigo demais."), call. = FALSE)
  }
  if (as.integer(version) != SUPPORTED_SCHEMA_VERSION) {
    stop(
      glue::glue(
        "'{what}' usa schema_version={version}, mas este pacote lê ",
        "{SUPPORTED_SCHEMA_VERSION}. Atualize analysis/ ou reexporte o run."
      ),
      call. = FALSE
    )
  }
  invisible(TRUE)
}

#' Lê o EDA de um dataset
#'
#' @param dataset_id Identificador do dataset (ex.: `medmnist-breastmnist-28`).
#' @param root Raiz de resultados.
#' @return Lista com `summary` e `examples`.
#' @export
read_dataset_eda <- function(dataset_id, root = NULL) {
  root <- results_root(root)
  dir <- file.path(root, "datasets", dataset_id)
  if (!dir.exists(dir)) {
    stop(
      glue::glue("EDA de '{dataset_id}' não encontrado. Rode `just eda <dataset>` antes."),
      call. = FALSE
    )
  }
  summary <- jsonlite::fromJSON(file.path(dir, "summary.json"), simplifyVector = TRUE)
  check_schema(summary$schema_version, dataset_id)
  list(
    summary = summary,
    examples = readr::read_csv(
      file.path(dir, "examples.csv.gz"),
      show_col_types = FALSE, progress = FALSE
    )
  )
}

#' Junta as predições de um run com o EDA do dataset
#'
#' É o join que justifica o desenho todo: `(split, example_idx)` liga o erro do
#' modelo à propriedade da imagem. Sem ele, o EDA seria descrição solta e as
#' predições seriam só um vetor de acertos.
#'
#' @param run Resultado de [read_run()].
#' @param eda Resultado de [read_dataset_eda()]; `NULL` tenta descobrir pelo manifest.
#' @param root Raiz de resultados.
#' @return Data frame de predições com as colunas do EDA e a flag `correct`.
#' @export
join_predictions_eda <- function(run, eda = NULL, root = NULL) {
  if (is.null(eda)) {
    eda <- read_dataset_eda(run$manifest$dataset_id, root)
  }
  joined <- dplyr::left_join(
    run$predictions,
    dplyr::select(
      eda$examples,
      "split", "example_idx", "class_name",
      "mean_intensity", "std_intensity", "nonzero_frac"
    ),
    by = c("split", "example_idx")
  )

  # Um NA aqui significa que o EDA e o run vieram de datasets diferentes; seguir
  # em frente produziria um relatório que cruza a imagem errada com o erro.
  missing <- sum(is.na(joined$class_name))
  if (missing > 0) {
    stop(
      glue::glue(
        "{missing} predições não casaram com o EDA de '{eda$summary$dataset_id}'. ",
        "O EDA foi gerado para outro dataset ou outra resolução?"
      ),
      call. = FALSE
    )
  }

  dplyr::mutate(joined, correct = .data$y_true == .data$y_pred)
}
