"""Perceptron multicamada (MLP) totalmente conectado, configurável."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """MLP configurável sobre a imagem achatada.

    Estrutura: ``Flatten`` -> ``n_layers`` blocos ``Linear -> ReLU -> Dropout`` ->
    camada linear de saída. A primeira camada é ``nn.LazyLinear`` para dispensar o
    cálculo manual da dimensão de entrada (in_channels x altura x largura).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        hidden_units: int = 256,
        n_layers: int = 2,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        layers: list[nn.Module] = [nn.Flatten()]
        for i in range(n_layers):
            layers.append(nn.LazyLinear(hidden_units) if i == 0 else nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_units, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
