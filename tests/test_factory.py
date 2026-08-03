"""Testes da fábrica de modelos e da separação de hiperparâmetros."""

import torch
from torch import nn

from cvlab.models.factory import arch_params, build_model, lit_params

_MLP_CFG = {
    "_target_": "cvlab.models.mlp.MLP",
    "hidden_units": 64,
    "n_layers": 2,
    "dropout_rate": 0.1,
    "lr": 1e-3,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "batch_size": 32,
}


def test_split_config() -> None:
    """arch_params isola a arquitetura; lit_params isola o treino (sem batch_size/_target_)."""
    assert arch_params(_MLP_CFG) == {"hidden_units": 64, "n_layers": 2, "dropout_rate": 0.1}
    assert lit_params(_MLP_CFG) == {"lr": 1e-3, "optimizer": "adam", "weight_decay": 0.0}


def test_build_model_dispatch() -> None:
    """build_model instancia o alvo correto e produz logits com o shape esperado."""
    model = build_model(_MLP_CFG, in_channels=1, num_classes=10)
    assert isinstance(model, nn.Module)
    out = model(torch.randn(2, 1, 28, 28))
    assert out.shape == (2, 10)

    perceptron = build_model(
        {"_target_": "cvlab.models.perceptron.LinearPerceptron"}, in_channels=1, num_classes=10
    )
    assert perceptron(torch.randn(2, 1, 28, 28)).shape == (2, 10)
