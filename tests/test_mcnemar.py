"""Testes unitários para o teste de McNemar pareado (tracking.mcnemar_test)."""

import numpy as np
import pytest
from scipy.stats import chi2

from cvlab.tracking import mcnemar_test


def test_mcnemar_binom_exact() -> None:
    """Verifica se o método binom_exact é selecionado para poucas discordâncias (< 25)."""
    y_true = np.array([1, 1, 1, 1, 1])
    preds_a = np.array([1, 1, 1, 0, 0])
    preds_b = np.array([1, 1, 0, 1, 1])

    res = mcnemar_test(y_true, preds_a, preds_b)

    assert res["method"] == "binom_exact"
    assert res["n01"] == 2
    assert res["n10"] == 1
    assert res["statistic"] == 1.0
    assert 0.0 <= res["p_value"] <= 1.0


def test_mcnemar_chi2_continuity() -> None:
    """Verifica se o método chi2_continuity é selecionado para muitas discordâncias (>= 25)."""
    y_true = np.zeros(100, dtype=int)
    preds_a = np.zeros(100, dtype=int)
    preds_b = np.zeros(100, dtype=int)

    # Configura n01 = 30 e n10 = 10
    preds_a[0:30] = 1  # A errou os primeiros 30
    preds_b[30:40] = 1  # B errou dos índices 30 a 39

    res = mcnemar_test(y_true, preds_a, preds_b)

    assert res["method"] == "chi2_continuity"
    assert res["n01"] == 30
    assert res["n10"] == 10
    disc = 40
    expected_stat = ((abs(30 - 10) - 1) ** 2) / disc
    expected_p = float(chi2.sf(expected_stat, 1))

    assert res["statistic"] == pytest.approx(expected_stat)
    assert res["p_value"] == pytest.approx(expected_p)


def test_mcnemar_zero_discordance() -> None:
    """Verifica se quando não há nenhuma discordância (n01=0, n10=0), p_value é 1.0."""
    y_true = np.array([0, 1, 0, 1])
    preds_a = np.array([0, 1, 0, 1])
    preds_b = np.array([0, 1, 0, 1])

    res = mcnemar_test(y_true, preds_a, preds_b)

    assert res["n01"] == 0
    assert res["n10"] == 0
    assert res["p_value"] == 1.0
