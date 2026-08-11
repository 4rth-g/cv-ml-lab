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


_RESNET_CFG = {
    "_target_": "cvlab.models.resnet.ResNet",
    "width": 16,
    "blocks_per_stage": 1,
    "n_stages": 2,
    "lr": 1e-2,
    "optimizer": "sgd",
    "weight_decay": 5e-4,
    "scheduler": "cosine",
    "batch_size": 128,
}


def test_split_config() -> None:
    """arch_params isola a arquitetura; lit_params isola o treino (sem batch_size/_target_)."""
    assert arch_params(_MLP_CFG) == {"hidden_units": 64, "n_layers": 2, "dropout_rate": 0.1}
    assert lit_params(_MLP_CFG) == {"lr": 1e-3, "optimizer": "adam", "weight_decay": 0.0}


def test_scheduler_routed_as_training_param() -> None:
    """`scheduler` vai ao LitClassifier, nunca ao construtor do modelo.

    Se vazasse para arch_params, o build_model quebraria com um kwarg inesperado.
    """
    assert "scheduler" not in arch_params(_RESNET_CFG)
    assert lit_params(_RESNET_CFG)["scheduler"] == "cosine"
    assert arch_params(_RESNET_CFG) == {"width": 16, "blocks_per_stage": 1, "n_stages": 2}


def test_build_resnet_from_config() -> None:
    """A ResNet é instanciável pela fábrica como qualquer outra arquitetura."""
    model = build_model(_RESNET_CFG, in_channels=3, num_classes=10)
    assert isinstance(model, nn.Module)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_build_model_dispatch() -> None:
    """build_model instancia o alvo correto e produz logits com o shape esperado."""
    model = build_model(_MLP_CFG, in_channels=1, num_classes=10)
    assert isinstance(model, nn.Module)
    out = model(torch.randn(2, 1, 28, 28))
    assert out.shape == (2, 10)

    perceptron = build_model({"_target_": "cvlab.models.perceptron.LinearPerceptron"}, in_channels=1, num_classes=10)
    assert perceptron(torch.randn(2, 1, 28, 28)).shape == (2, 10)
