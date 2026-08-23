"""Travas da definição de acurácia no lado Python.

Existem por causa de um bug concreto: `MulticlassAccuracy(num_classes=N)` do
torchmetrics tem `average="macro"` por default, então a métrica de validação
logada era acurácia BALANCEADA enquanto a de teste era calculada como micro — e
as duas iam para a mesma coluna `acc` do `seed_metrics.csv`. Ver `cvlab.metrics`.
"""

import pytest

from cvlab.metrics import (
    accuracy,
    balanced_accuracy,
    objective_metric_key,
    objective_score,
)


def test_accuracy_and_balanced_accuracy_differ_under_imbalance() -> None:
    """O caso que o bug escondia: num split desbalanceado as duas divergem muito.

    9 exemplos da classe 0 (todos certos) e 1 da classe 1 (errado). Micro dá
    0,90 e macro dá 0,50 — a diferença entre "quase perfeito" e "não aprendeu a
    classe rara". É a ordem de grandeza observada no fixture do BreastMNIST.
    """
    targets = [0] * 9 + [1]
    preds = [0] * 10

    assert accuracy(preds, targets) == pytest.approx(0.90)
    assert balanced_accuracy(preds, targets) == pytest.approx(0.50)


def test_balanced_accuracy_ignores_classes_absent_from_targets() -> None:
    """Classe que não ocorre nos rótulos não entra na média.

    Contá-la como recall zero puxaria a métrica para baixo por um motivo que não
    é do modelo. Mesma convenção do scikit-learn e de `per_class_metrics()` em
    `analysis/R/metrics.R`, que reserva NA para classe ausente dos rótulos.
    """
    # Só as classes 0 e 1 aparecem; a 2 existe no dataset mas não neste split.
    targets = [0, 0, 1, 1]
    preds = [0, 0, 1, 2]

    # recalls: classe 0 = 1.0, classe 1 = 0.5 -> média 0.75 (a classe 2 não conta)
    assert balanced_accuracy(preds, targets) == pytest.approx(0.75)


def test_accuracy_agrees_with_balanced_when_classes_are_equally_sized() -> None:
    """Com classes do mesmo tamanho as duas coincidem — o que torna o bug invisível.

    É exatamente por isso que ele sobreviveu: em MNIST (razão 1,235) micro e
    macro quase não se separam, e só um dataset desbalanceado o expõe.
    """
    targets = [0, 0, 1, 1]
    preds = [0, 1, 1, 1]
    assert accuracy(preds, targets) == pytest.approx(balanced_accuracy(preds, targets))


def test_objective_score_dispatches_by_name() -> None:
    targets = [0] * 9 + [1]
    preds = [0] * 10

    assert objective_score(preds, targets, "accuracy") == pytest.approx(0.90)
    assert objective_score(preds, targets, "balanced_accuracy") == pytest.approx(0.50)


def test_unknown_objective_raises_with_the_valid_names() -> None:
    """Erro cedo e legível: um nome errado na config não pode virar silêncio."""
    with pytest.raises(ValueError, match="balanced_accuracy"):
        objective_score([0], [0], "f1_macro")


def test_objective_metric_key_matches_what_lit_module_logs() -> None:
    """As chaves precisam existir em `callback_metrics`, senão o early stopping lê 0.0.

    `LitClassifier` loga `val_acc`/`val_bal_acc` (e os equivalentes de teste);
    se este mapeamento sair de sincronia com aquele, o callback de restauração
    passa a escolher sempre a primeira época sem falhar em lugar nenhum.
    """
    assert objective_metric_key("accuracy", "val") == "val_acc"
    assert objective_metric_key("balanced_accuracy", "val") == "val_bal_acc"
    assert objective_metric_key("accuracy", "test") == "test_acc"


def test_mismatched_lengths_are_fatal() -> None:
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        accuracy([0, 1], [0])
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        balanced_accuracy([0, 1], [0])
