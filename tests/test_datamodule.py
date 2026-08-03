"""Testes unitários para DataModules (MNIST e FashionMNIST)."""

from pathlib import Path

import pytest

from cvlab.data.fashion_mnist import FashionMNISTDataModule
from cvlab.data.mnist import MNISTDataModule


def test_mnist_datamodule_attributes() -> None:
    """Testa atributos básicos do MNISTDataModule sem necessitar de download prévio."""
    dm = MNISTDataModule(data_dir="./data", batch_size=32, val_size=5000)
    assert dm.in_channels == 1
    assert dm.num_classes == 10
    assert len(dm.class_names) == 10
    assert dm.batch_size == 32
    assert dm.val_size == 5000


def test_fashion_mnist_datamodule_attributes() -> None:
    """Testa atributos específicos do FashionMNISTDataModule (mean/std/classes)."""
    dm = FashionMNISTDataModule(data_dir="./data")
    assert dm.in_channels == 1
    assert dm.num_classes == 10
    assert dm.class_names[0] == "Camiseta/top"
    assert dm.mean == (0.2860,)
    assert dm.std == (0.3530,)


def test_datamodule_setup_if_data_exists(tmp_path: Path) -> None:
    """Se o dataset MNIST estiver baixado no diretório data/, testa a divisão dos DataLoaders."""
    data_dir = Path("./data")
    if not (data_dir / "MNIST").exists():
        pytest.skip("Dataset MNIST não encontrado localmente em ./data, pulando teste de divisão de DataLoader.")

    dm = MNISTDataModule(data_dir=str(data_dir), batch_size=64, val_size=10000)
    dm.setup("fit")

    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    test_dl = dm.test_dataloader()

    assert len(train_dl.dataset) == 50000
    assert len(val_dl.dataset) == 10000
    assert len(test_dl.dataset) == 10000

    x, y = next(iter(train_dl))
    assert x.shape[1:] == (1, 28, 28)
    assert len(y) == 64
