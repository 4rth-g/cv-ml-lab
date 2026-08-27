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
    trials = read_optional(file.path(run_dir, "trials.csv")),
    # Opcionais: runs exportados antes destes artefatos existirem devolvem NULL,
    # e o consumidor decide se degrada ou reclama. Não é motivo para bump de
    # schema_version, porque nenhuma coluna existente mudou.
    epochs = read_optional(file.path(run_dir, "epochs.csv")),
    probabilities = read_optional(file.path(run_dir, "probabilities.csv.gz"))
  )
}

#' Lê um CSV se existir, senão devolve `NULL`
#' @param path Caminho do arquivo.
#' @keywords internal
read_optional <- function(path) {
  if (!file.exists(path)) {
    return(NULL)
  }
  readr::read_csv(path, show_col_types = FALSE, progress = FALSE)
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
  examples <- readr::read_csv(
    file.path(dir, "examples.csv.gz"),
    show_col_types = FALSE, progress = FALSE
  )

  # class_name é RÓTULO, nunca número. Em datasets cujas classes se chamam "0".."9"
  # (MNIST, Fashion-MNIST) o readr infere numeric, e aí `lm(x ~ class_name)` vira
  # regressão linear no valor do dígito em vez de ANOVA por classe. O R² despenca
  # de 0,34 para 0,001 sem nenhum erro visível — o tipo de bug que passa batido
  # porque o resultado parece só "fraco", não errado.
  examples$class_name <- as.character(examples$class_name)

  list(summary = summary, examples = examples)
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

#' Lê a montagem PNG de uma classe
#'
#' O Python exporta uma grade de exemplos por classe justamente porque o R não
#' decodifica `.npz` nem os formatos do torchvision. Aqui ele lê um PNG comum, o
#' que mantém a fronteira intacta: nenhuma dependência de código Python, nenhuma
#' leitura de dataset bruto.
#'
#' @param eda Resultado de [read_dataset_eda()].
#' @param class_name Nome da classe (como em `summary$class_names`).
#' @param root Raiz de resultados.
#' @return Objeto `magick-image`.
#' @export
read_montage <- function(eda, class_name, root = NULL) {
  if (!requireNamespace("magick", quietly = TRUE)) {
    stop("Pacote 'magick' ausente. Rode dentro de `nix develop`.", call. = FALSE)
  }
  files <- eda$summary$montage$files
  if (is.null(files) || is.null(files[[class_name]])) {
    stop(
      glue::glue(
        "Sem montagem para '{class_name}'. Rode `just eda <dataset>` com uma versão ",
        "do cvlab que exporte montage/."
      ),
      call. = FALSE
    )
  }
  path <- file.path(results_root(root), "datasets", eda$summary$dataset_id, files[[class_name]])
  magick::image_read(path)
}

#' Montagens de todas as classes, compostas com rótulo
#'
#' Os rótulos são desenhados pelo ggplot, e não por `magick::image_annotate`:
#' o ImageMagick do ambiente Nix não encontra fonte (`unable to read font`),
#' enquanto o device gráfico do R resolve tipografia pelo systemfonts e funciona.
#'
#' @param eda Resultado de [read_dataset_eda()].
#' @param classes Quais classes; `NULL` usa todas.
#' @param root Raiz de resultados.
#' @param ncol Colunas na composição; `NULL` escolhe automaticamente.
#' @return Objeto patchwork (ggplot).
#' @export
montage_grid <- function(eda, classes = NULL, root = NULL, ncol = NULL) {
  if (!requireNamespace("patchwork", quietly = TRUE)) {
    stop("Pacote 'patchwork' ausente. Rode dentro de `nix develop`.", call. = FALSE)
  }
  classes <- classes %||% names(eda$summary$montage$files)

  plots <- lapply(classes, function(cl) {
    raster <- grDevices::as.raster(read_montage(eda, cl, root))
    ggplot2::ggplot() +
      ggplot2::annotation_raster(raster, xmin = 0, xmax = 1, ymin = 0, ymax = 1) +
      ggplot2::coord_fixed(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
      # strwrap em vez de deixar em uma linha: nomes de classe médicos são
      # longos ("actinic keratoses and intraepithelial carcinoma") e colidem com
      # o título do painel vizinho.
      ggplot2::labs(title = paste(strwrap(cl, width = 22), collapse = "\n")) +
      ggplot2::theme_void(base_size = 10) +
      ggplot2::theme(
        plot.title = ggplot2::element_text(
          hjust = 0.5, size = 8.5, lineheight = 1.05,
          margin = ggplot2::margin(b = 3)
        )
      )
  })

  patchwork::wrap_plots(plots, ncol = ncol %||% min(length(plots), 5))
}

`%||%` <- function(a, b) if (is.null(a)) b else a
