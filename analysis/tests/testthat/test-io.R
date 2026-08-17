# Contrato de leitura dos artefatos.
#
# O fixture em inst/extdata/results/ é um run REAL (orçamento smoke) gerado pelo
# pipeline Python. É o contrato compartilhado entre as duas linguagens: se o
# `cvlab.export` mudar uma coluna, tests/test_export.py e este arquivo divergem
# no mesmo commit, e alguém percebe antes do relatório sair errado.

fixture_root <- function() {
  path <- system.file("extdata", "results", package = "cvlabstats")
  if (path == "") {
    path <- testthat::test_path("..", "..", "inst", "extdata", "results")
  }
  normalizePath(path, mustWork = TRUE)
}

fixture_run_id <- function() {
  readLines(file.path(fixture_root(), "RUN_ID"), warn = FALSE)[1]
}

test_that("o índice é legível e tem as colunas do contrato", {
  idx <- read_index(fixture_root())
  expect_true(nrow(idx) >= 1)
  expect_true(all(c(
    "run_id", "timestamp_utc", "commit", "model", "dataset",
    "subset", "budget", "n_seeds", "selected_arm", "path"
  ) %in% names(idx)))
})

test_that("read_run devolve os quatro artefatos", {
  run <- read_run(fixture_run_id(), fixture_root())

  expect_equal(run$run_id, fixture_run_id())
  expect_equal(run$manifest$schema_version, SUPPORTED_SCHEMA_VERSION)
  expect_s3_class(run$seed_metrics, "data.frame")
  expect_s3_class(run$predictions, "data.frame")
  expect_s3_class(run$trials, "data.frame")
})

test_that("seed_metrics tem uma linha por (arm, seed, split)", {
  run <- read_run(fixture_run_id(), fixture_root())
  sm <- run$seed_metrics

  expect_setequal(unique(sm$arm), c("baseline", "tuned"))
  expect_setequal(unique(sm$split), c("val", "test"))
  expect_equal(nrow(sm), run$manifest$n_seeds * 2 * 2)
  expect_true(all(sm$acc >= 0 & sm$acc <= 1))
})

test_that("as três colunas de seed estão presentes e separadas", {
  sm <- read_run(fixture_run_id(), fixture_root())$seed_metrics
  expect_true(all(c("seed_init", "seed_data", "seed_split") %in% names(sm)))
  # Split oficial do MedMNIST: gravado como rótulo, não como número inventado.
  expect_equal(unique(sm$seed_split), "official")
})

test_that("predições cobrem val E test", {
  preds <- read_run(fixture_run_id(), fixture_root())$predictions
  expect_setequal(unique(preds$split), c("val", "test"))
  # Sem as de validação, escolher métrica pós-hoc teria que olhar o teste.
  expect_true(sum(preds$split == "val") > 0)
})

test_that("example_idx é contíguo dentro de cada (arm, seed, split)", {
  preds <- read_run(fixture_run_id(), fixture_root())$predictions
  grupo <- preds[preds$arm == "baseline" & preds$split == "test", ]
  grupo <- grupo[grupo$seed_init == min(grupo$seed_init), ]
  expect_equal(sort(grupo$example_idx), seq(0, nrow(grupo) - 1))
})

test_that("o join com o EDA casa 100% das predições", {
  run <- read_run(fixture_run_id(), fixture_root())
  joined <- join_predictions_eda(run, root = fixture_root())

  expect_equal(nrow(joined), nrow(run$predictions))
  expect_false(any(is.na(joined$class_name)))
  # y_true do run e do EDA têm que concordar exemplo a exemplo, senão o
  # example_idx estaria apontando para a imagem errada.
  expect_true(all(joined$correct == (joined$y_true == joined$y_pred)))
})

test_that("schema_version desconhecida é recusada, não interpretada", {
  expect_error(check_schema(999, "run-teste"), "schema_version")
  expect_error(check_schema(NA, "run-teste"), "schema_version")
})

test_that("run inexistente falha com mensagem útil", {
  expect_error(read_run("nao-existe", fixture_root()), "não encontrado")
})

test_that("EDA de outro dataset é recusado no join", {
  run <- read_run(fixture_run_id(), fixture_root())
  eda <- read_dataset_eda("medmnist-breastmnist-28", fixture_root())
  # Embaralhar o example_idx simula EDA de outro dataset/resolução.
  eda$examples$example_idx <- eda$examples$example_idx + 100000L
  expect_error(join_predictions_eda(run, eda), "não casaram")
})

test_that("métricas pós-hoc saem das predições sem retreinar", {
  run <- read_run(fixture_run_id(), fixture_root())
  met <- run_metrics(run)

  expect_true(all(c("accuracy", "balanced_accuracy", "f1_macro") %in% names(met)))
  expect_equal(nrow(met), run$manifest$n_seeds * 2 * 2)

  # Num dataset desbalanceado, balanced accuracy < accuracy. É exatamente o
  # diagnóstico que exigiria um retreino se as predições não fossem exportadas.
  test_rows <- met[met$split == "test", ]
  expect_true(all(test_rows$balanced_accuracy <= test_rows$accuracy))
})
