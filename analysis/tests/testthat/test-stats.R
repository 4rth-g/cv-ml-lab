# Métricas por classe: convenção de divisão por zero.

test_that("classe nunca predita entra na macro com F1 zero, não como NA", {
  # Cenário real e comum em dataset desbalanceado: o modelo prevê SÓ a classe
  # majoritária. A classe ausente das predições tem precisão 0/0.
  #
  # Se ela virar NA, o `na.rm = TRUE` da média macro a descarta e o F1 macro
  # infla — num binário, de 0,42 para 0,84. O bug é silencioso porque o número
  # resultante parece plausível.
  y_true <- c(rep(0L, 42), rep(1L, 114))
  y_pred <- rep(1L, 156)

  pc <- per_class_metrics(y_true, y_pred, c("raro", "comum"))

  ausente <- pc[pc$class_idx == 0, ]
  expect_equal(ausente$support, 42)
  expect_equal(ausente$recall, 0)
  expect_equal(ausente$precision, 0)
  expect_equal(ausente$f1, 0)

  presente <- pc[pc$class_idx == 1, ]
  expect_equal(presente$recall, 1)
  expect_equal(presente$precision, 114 / 156, tolerance = 1e-9)

  sm <- summary_metrics(y_true, y_pred)
  # macro = (0 + 0.8444) / 2
  expect_equal(sm$f1_macro, mean(c(0, presente$f1)), tolerance = 1e-9)
  expect_lt(sm$f1_macro, 0.5)
  # e continua coerente com a balanced accuracy de um preditor de classe única
  expect_equal(sm$balanced_accuracy, 0.5)
})

test_that("classe ausente dos RÓTULOS fica NA, não zero", {
  # Aqui a classe 2 não existe na verdade nem nas predições: não há o que medir,
  # e contá-la como zero puniria o modelo por uma classe inexistente.
  y_true <- c(0L, 0L, 1L, 1L)
  y_pred <- c(0L, 1L, 1L, 1L)
  pc <- per_class_metrics(y_true, y_pred, c("a", "b", "c"))
  expect_false(2 %in% pc$class_idx)
})
