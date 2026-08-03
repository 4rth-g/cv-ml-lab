"""Testes do perceptron multicamada (MLP)."""

import torch

from cvlab.models.mlp import MLP


def test_mlp_forward_shapes() -> None:
    """Forward com 1 e 3 canais e num_classes variável."""
    m1 = MLP(in_channels=1, num_classes=10, hidden_units=64, n_layers=2)
    out1 = m1(torch.randn(4, 1, 28, 28))
    assert out1.shape == (4, 10)

    m2 = MLP(in_channels=3, num_classes=100, hidden_units=128, n_layers=1)
    out2 = m2(torch.randn(4, 3, 32, 32))
    assert out2.shape == (4, 100)


def test_mlp_depth() -> None:
    """n_layers controla a quantidade de camadas ocultas (Linear = n_layers + saída)."""
    for n_layers in (1, 2, 3):
        m = MLP(in_channels=1, num_classes=10, hidden_units=32, n_layers=n_layers)
        m(torch.randn(2, 1, 28, 28))  # materializa LazyLinear
        linears = [mod for mod in m.modules() if isinstance(mod, torch.nn.Linear)]
        assert len(linears) == n_layers + 1
