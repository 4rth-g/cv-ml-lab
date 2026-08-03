"""CNN configurável dataset-agnóstica com nn.LazyLinear."""

from __future__ import annotations

import torch
from torch import nn


class ConfigurableCNN(nn.Module):
    """CNN configurável com quantidade variável de blocos convolucionais e LazyLinear.

    O uso de nn.LazyLinear no classificador dispensa o cálculo manual da
    dimensão de flatten, permitindo alterar in_channels, altura/largura da imagem
    e número de blocos convolucionais sem quebrar o forward.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        num_filters: int = 32,
        dropout_rate: float = 0.25,
        fc_units: int = 128,
        n_conv_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.fc_units = fc_units
        self.n_conv_blocks = n_conv_blocks

        layers: list[nn.Module] = []
        current_channels = in_channels

        for b in range(n_conv_blocks):
            out_channels = num_filters * (2**b)
            layers.extend(
                [
                    nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            current_channels = out_channels

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(fc_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
