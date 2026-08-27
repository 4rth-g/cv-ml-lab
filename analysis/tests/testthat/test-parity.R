# Paridade com a implementação Python que foi removida.
#
# Estes casos vieram de tests/test_mcnemar.py, que deixa de existir quando a
# estatística sai do Python. Os valores esperados são LITERAIS, calculados à
# mão, e não chamadas ao Python: é isso que torna a paridade verificável depois
# que o outro lado sumiu, sem que nenhum dos dois precise do outro em runtime.

test_that("McNemar usa binomial exata abaixo de 25 discordâncias", {
  # 10 exemplos: A acerta todos, B erra 3. n01 = 0, n10 = 3, disc = 3 < 25.
  y_true <- c(0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
  pred_a <- y_true
  pred_b <- c(1, 1, 0, 1, 1, 1, 0, 1, 0, 0)

  res <- mcnemar_pair(y_true, pred_a, pred_b)

  expect_equal(res$method, "binom_exact")
  expect_equal(res$n01, 0L)
  expect_equal(res$n10, 3L)
  expect_equal(res$statistic, 0)
  # 2 * pbinom(0, 3, 0.5) = 2 * 0.125 = 0.25
  expect_equal(res$p_value, 0.25, tolerance = 1e-12)
})

test_that("McNemar usa qui-quadrado com correção acima de 25 discordâncias", {
  # Construído para dar n01 = 30, n10 = 10 (disc = 40 >= 25).
  n <- 100
  y_true <- rep(0L, n)
  pred_a <- rep(0L, n)
  pred_b <- rep(0L, n)

  # 30 exemplos em que A erra e B acerta
  pred_a[1:30] <- 1L
  # 10 exemplos em que A acerta e B erra
  pred_b[31:40] <- 1L

  res <- mcnemar_pair(y_true, pred_a, pred_b)

  expect_equal(res$method, "chi2_continuity")
  expect_equal(res$n01, 30L)
  expect_equal(res$n10, 10L)
  # (|30-10| - 1)^2 / 40 = 361/40 = 9.025
  expect_equal(res$statistic, 9.025, tolerance = 1e-12)
  expect_equal(res$p_value, stats::pchisq(9.025, 1, lower.tail = FALSE), tolerance = 1e-12)
})

test_that("McNemar sem discordância devolve p = 1", {
  y_true <- c(0, 1, 0, 1)
  expect_equal(mcnemar_pair(y_true, y_true, y_true)$p_value, 1)
  expect_equal(mcnemar_pair(y_true, y_true, y_true)$n01, 0L)
  expect_equal(mcnemar_pair(y_true, y_true, y_true)$n10, 0L)
})

test_that("o ramo binomial concorda com o que o Python calculava", {
  # O Python fazia: 2 * binom.cdf(min(n01,n10), disc, 0.5), truncado em 1.
  # Em R o equivalente é 2 * pbinom(...). Confere em vários pontos.
  for (case in list(c(1, 4), c(2, 8), c(5, 19), c(0, 10))) {
    n01 <- case[1]
    n10 <- case[2]
    disc <- n01 + n10
    esperado <- min(1, 2 * stats::pbinom(min(n01, n10), disc, 0.5))

    y_true <- rep(0L, 200)
    pred_a <- rep(0L, 200)
    pred_b <- rep(0L, 200)
    if (n01 > 0) pred_a[seq_len(n01)] <- 1L
    if (n10 > 0) pred_b[(n01 + 1):(n01 + n10)] <- 1L

    res <- mcnemar_pair(y_true, pred_a, pred_b)
    expect_equal(res$method, "binom_exact")
    expect_equal(res$p_value, esperado, tolerance = 1e-12)
  }
})

test_that("a fronteira do método está exatamente em 25 discordâncias", {
  # O Python usava `if disc < 25`, então disc = 24 é exata e disc = 25 já é
  # qui-quadrado. Fixar a fronteira aqui é o que impede um "<=" de entrar
  # despercebido e mudar p-valores de runs futuros em relação aos publicados.
  build <- function(n01, n10) {
    n <- 200
    y_true <- rep(0L, n)
    pred_a <- rep(0L, n)
    pred_b <- rep(0L, n)
    if (n01 > 0) pred_a[seq_len(n01)] <- 1L
    if (n10 > 0) pred_b[(n01 + 1):(n01 + n10)] <- 1L
    mcnemar_pair(y_true, pred_a, pred_b)
  }

  expect_equal(build(12, 12)$method, "binom_exact") # disc = 24
  expect_equal(build(12, 13)$method, "chi2_continuity") # disc = 25
})

test_that("t pareado e IC batem com o cálculo manual que o Python fazia", {
  # O Python computava o IC à mão: diff_mean +/- t.ppf(0.975, n-1) * sd/sqrt(n).
  # Aqui ele sai do próprio t.test(); os dois têm que coincidir.
  baseline <- c(0.90, 0.91, 0.89, 0.92, 0.90)
  tuned <- c(0.92, 0.93, 0.90, 0.95, 0.91)
  paired <- data.frame(
    seed_init = 42:46, baseline = baseline, tuned = tuned,
    diff = tuned - baseline
  )

  res <- paired_t(paired)

  d <- tuned - baseline
  n <- length(d)
  se <- stats::sd(d) / sqrt(n)
  t_crit <- stats::qt(0.975, df = n - 1)

  expect_equal(res$diff_mean, mean(d), tolerance = 1e-12)
  expect_equal(res$t, mean(d) / se, tolerance = 1e-10)
  expect_equal(res$ci_low, mean(d) - t_crit * se, tolerance = 1e-10)
  expect_equal(res$ci_high, mean(d) + t_crit * se, tolerance = 1e-10)
  # Cohen's d_z = média das diferenças / desvio das diferenças (fórmula do Python)
  expect_equal(res$cohens_dz, mean(d) / stats::sd(d), tolerance = 1e-6)
})

test_that("d_z equivale a t/sqrt(n), como observado no estudo do repo", {
  baseline <- c(0.80, 0.82, 0.81, 0.79, 0.83)
  tuned <- c(0.85, 0.84, 0.86, 0.83, 0.88)
  paired <- data.frame(
    seed_init = 42:46, baseline = baseline, tuned = tuned,
    diff = tuned - baseline
  )
  res <- paired_t(paired)
  expect_equal(res$cohens_dz, res$t / sqrt(res$n_seeds), tolerance = 1e-6)
})

test_that("diferença constante não estoura, devolve NA", {
  # Cenário real em orçamento smoke: a busca devolve a própria baseline, todas
  # as diferenças são zero e sd = 0. Estourar aqui quebraria o relatório inteiro.
  paired <- data.frame(
    seed_init = 42:44,
    baseline = c(0.5, 0.6, 0.7),
    tuned = c(0.5, 0.6, 0.7),
    diff = c(0, 0, 0)
  )
  res <- paired_t(paired)
  expect_equal(res$diff_mean, 0)
  expect_true(is.na(res$p_value))
})
