"""Testes do perceptron linear."""

import torch

from cvlab.models.perceptron import LinearPerceptron


def test_perceptron_forward_shapes() -> None:
    """Forward com 1 e 3 canais e num_classes variável."""
    m1 = LinearPerceptron(in_channels=1, num_classes=10)
    out1 = m1(torch.randn(4, 1, 28, 28))
    assert out1.shape == (4, 10)

    m2 = LinearPerceptron(in_channels=3, num_classes=100)
    out2 = m2(torch.randn(4, 3, 32, 32))
    assert out2.shape == (4, 100)


def test_perceptron_is_linear() -> None:
    """Sem camadas ocultas: apenas Flatten + uma Linear (nenhuma não-linearidade)."""
    m = LinearPerceptron(in_channels=1, num_classes=10)
    m(torch.randn(2, 1, 28, 28))  # materializa LazyLinear
    linears = [mod for mod in m.modules() if isinstance(mod, torch.nn.Linear)]
    assert len(linears) == 1
    assert not any(isinstance(mod, torch.nn.ReLU) for mod in m.modules())
