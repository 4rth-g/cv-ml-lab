"""Perceptron linear: um único mapeamento linear das features para as classes."""

from __future__ import annotations

import torch
from torch import nn


class LinearPerceptron(nn.Module):
    """Classificador linear (perceptron de camada única) sobre a imagem achatada.

    Não possui camadas ocultas nem não-linearidade: é a fronteira de decisão linear
    ``W·x + b``. Serve de baseline mínima — qualquer arquitetura mais complexa deve
    superá-la. Usa ``nn.LazyLinear`` para não depender de altura/largura da imagem.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
