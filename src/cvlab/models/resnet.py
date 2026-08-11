"""ResNet configurável com conexões residuais, no estilo CIFAR."""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """Bloco residual de duas convoluções 3x3: ``y = relu(F(x) + atalho(x))``.

    O atalho é a identidade sempre que possível. Quando o bloco muda a forma —
    por reduzir a resolução (``stride=2``) ou por alterar o número de canais —
    a identidade não é somável, e usa-se uma projeção 1x1 no atalho.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    """ResNet configurável em profundidade e largura, para imagens pequenas.

    A conexão residual existe para tornar redes profundas *treináveis*: somar a
    entrada à saída do bloco dá ao gradiente um caminho direto para as camadas
    iniciais, evitando a degradação que faz uma rede mais profunda ficar pior
    que uma rasa mesmo no treino.

    **O stem é 3x3 com stride 1 e sem maxpool**, ao contrário das ResNet de
    ImageNet (7x7 stride 2 + maxpool). Aquele stem reduz a entrada 4x logo de
    cara, o que é adequado a 224x224 mas destrói uma imagem 32x32 antes do
    primeiro bloco. É o erro clássico de reaproveitar `torchvision.models.resnet`
    em CIFAR.

    A cabeça usa `AdaptiveAvgPool2d(1)`, então a rede é agnóstica à resolução de
    entrada: as mesmas configurações servem para 28x28x1 e 32x32x3.

    Profundidade total = ``2 * blocks_per_stage * n_stages + 2``. Com o padrão
    (2 blocos, 3 estágios) são 14 camadas com peso.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        width: int = 64,
        blocks_per_stage: int = 2,
        n_stages: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.width = width
        self.blocks_per_stage = blocks_per_stage
        self.n_stages = n_stages

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )

        # Cada estágio dobra os canais e (menos o primeiro) reduz a resolução
        # pela metade — o compromisso usual entre capacidade e custo.
        blocks: list[nn.Module] = []
        channels = width
        for stage in range(n_stages):
            out_channels = width * (2**stage)
            for block in range(blocks_per_stage):
                stride = 2 if (block == 0 and stage > 0) else 1
                blocks.append(BasicBlock(channels, out_channels, stride))
                channels = out_channels
        self.stages = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))
