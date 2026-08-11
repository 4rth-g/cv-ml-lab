"""Testes unitários para a ResNet e o bloco residual."""

import pytest
import torch

from cvlab.models.resnet import BasicBlock, ResNet


def test_resnet_forward_shapes() -> None:
    """Forward funciona com resoluções e canais diferentes, sem reconfigurar o modelo."""
    batch_size = 4

    rgb = ResNet(in_channels=3, num_classes=10)
    assert rgb(torch.randn(batch_size, 3, 32, 32)).shape == (batch_size, 10)

    # A cabeça usa AdaptiveAvgPool2d, então 28x28 em 1 canal também vale.
    gray = ResNet(in_channels=1, num_classes=100)
    assert gray(torch.randn(batch_size, 1, 28, 28)).shape == (batch_size, 100)


@pytest.mark.parametrize("blocks,stages", [(1, 2), (2, 3), (3, 2)])
def test_resnet_depth_is_configurable(blocks: int, stages: int) -> None:
    """A profundidade acompanha blocks_per_stage e n_stages sem quebrar o forward."""
    model = ResNet(in_channels=3, num_classes=10, width=16, blocks_per_stage=blocks, n_stages=stages)
    assert len(model.stages) == blocks * stages
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_shortcut_is_identity_when_shape_is_preserved() -> None:
    """Sem mudança de forma o atalho é identidade; com mudança, vira projeção 1x1.

    É a propriedade que define o bloco residual — se o atalho projetasse sempre,
    o caminho direto do gradiente até as camadas iniciais se perderia.
    """
    same = BasicBlock(32, 32, stride=1)
    assert isinstance(same.shortcut, torch.nn.Identity)

    downsample = BasicBlock(32, 64, stride=2)
    assert not isinstance(downsample.shortcut, torch.nn.Identity)
    assert downsample(torch.randn(2, 32, 16, 16)).shape == (2, 64, 8, 8)


def test_stem_preserves_resolution() -> None:
    """O stem NÃO pode reduzir a resolução: em 32x32 isso destruiria o sinal.

    ResNets de ImageNet usam conv 7x7/stride 2 + maxpool, o que reduziria uma
    entrada 32x32 para 8x8 antes do primeiro bloco.
    """
    model = ResNet(in_channels=3, num_classes=10, width=16)
    out = model.stem(torch.randn(2, 3, 32, 32))
    assert out.shape[-2:] == (32, 32)
