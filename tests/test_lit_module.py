"""Testes do LitClassifier: seleção de otimizador e scheduler por configuração."""

import pytest
import torch
from torch import optim

from cvlab.models.lit_module import OPTIMIZERS, SCHEDULERS, LitClassifier
from cvlab.models.perceptron import LinearPerceptron


def _lit(**kwargs: object) -> LitClassifier:
    return LitClassifier(model=LinearPerceptron(in_channels=1, num_classes=10), num_classes=10, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("name,expected", [("adam", optim.Adam), ("adamw", optim.AdamW), ("sgd", optim.SGD)])
def test_optimizer_selected_by_name(name: str, expected: type) -> None:
    """Cada nome da tabela OPTIMIZERS resolve para a classe correspondente."""
    lit = _lit(optimizer=name)
    lit(torch.randn(2, 1, 28, 28))  # materializa o LazyLinear
    assert isinstance(lit.configure_optimizers(), expected)


def test_default_has_no_scheduler() -> None:
    """Sem scheduler configurado, `configure_optimizers` devolve só o otimizador.

    É o que garante que os resultados publicados antes da introdução dos
    schedulers continuem reproduzindo o mesmo treino, bit a bit.
    """
    lit = _lit()
    lit(torch.randn(2, 1, 28, 28))
    assert lit.scheduler_name == "none"
    assert isinstance(lit.configure_optimizers(), optim.Optimizer)


@pytest.mark.parametrize("name", [n for n in SCHEDULERS if n != "none"])
def test_scheduler_is_attached(name: str) -> None:
    """Com scheduler nomeado, o retorno é o dict que o Lightning espera."""

    class _FakeTrainer:
        max_epochs = 30

    lit = _lit(scheduler=name)
    lit(torch.randn(2, 1, 28, 28))
    lit.trainer = _FakeTrainer()  # type: ignore[assignment]

    cfg = lit.configure_optimizers()
    assert isinstance(cfg, dict)
    assert isinstance(cfg["optimizer"], optim.Optimizer)
    assert cfg["lr_scheduler"] is not None


def test_unknown_names_fail_loudly() -> None:
    """Nome inválido deve apontar as opções válidas, não falhar silenciosamente."""
    lit = _lit(optimizer="rmsprop")
    lit(torch.randn(2, 1, 28, 28))
    with pytest.raises(ValueError, match="Otimizador"):
        lit.configure_optimizers()

    lit2 = _lit(scheduler="triangular")
    lit2(torch.randn(2, 1, 28, 28))
    with pytest.raises(ValueError, match="Scheduler"):
        lit2.configure_optimizers()


def test_tables_cover_the_documented_choices() -> None:
    """As tabelas são o contrato com o search_space dos configs de tuning."""
    assert set(OPTIMIZERS) == {"adam", "adamw", "sgd"}
    assert set(SCHEDULERS) == {"none", "cosine", "step"}
