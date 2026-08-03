"""Testes unitários para o modelo ConfigurableCNN."""

import torch

from cvlab.models.cnn import ConfigurableCNN


def test_cnn_forward_shapes() -> None:
    """Verifica se o forward pass funciona com diferentes in_channels, num_classes e n_conv_blocks."""
    batch_size = 4

    # Teste 1: 1 canal (MNIST/Fashion), 10 classes, 2 blocos conv
    model1 = ConfigurableCNN(in_channels=1, num_classes=10, n_conv_blocks=2)
    x1 = torch.randn(batch_size, 1, 28, 28)
    out1 = model1(x1)
    assert out1.shape == (batch_size, 10)

    # Teste 2: 3 canais (RGB), 100 classes, 3 blocos conv
    model2 = ConfigurableCNN(in_channels=3, num_classes=100, n_conv_blocks=3)
    x2 = torch.randn(batch_size, 3, 32, 32)
    out2 = model2(x2)
    assert out2.shape == (batch_size, 100)


def test_cnn_lazy_linear_materialization() -> None:
    """Verifica se a camada LazyLinear é instanciada corretamente após a primeira passagem de dados."""
    model = ConfigurableCNN(in_channels=1, num_classes=10, fc_units=64)
    x = torch.randn(2, 1, 28, 28)
    # Antes do forward, o LazyLinear ainda é Uninitialized
    out = model(x)
    assert out.shape == (2, 10)
