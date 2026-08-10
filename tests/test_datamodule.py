"""Testes unitários para DataModules (MNIST, Fashion-MNIST e CIFAR-10)."""

from pathlib import Path

import pytest
import torch

from cvlab.data.base import BaseImageClfDataModule
from cvlab.data.cifar10 import CIFAR10DataModule
from cvlab.data.fashion_mnist import FashionMNISTDataModule
from cvlab.data.mnist import MNISTDataModule

# (classe, subdiretório criado pelo torchvision em ./data, shape de uma amostra)
DATAMODULES = [
    (MNISTDataModule, "MNIST", (1, 28, 28)),
    (FashionMNISTDataModule, "FashionMNIST", (1, 28, 28)),
    (CIFAR10DataModule, "cifar-10-batches-py", (3, 32, 32)),
]


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


def test_cifar10_datamodule_attributes() -> None:
    """Testa atributos específicos do CIFAR10DataModule (RGB, mean/std/classes/val_size)."""
    dm = CIFAR10DataModule(data_dir="./data")
    assert dm.in_channels == 3
    assert dm.num_classes == 10
    assert dm.class_names[0] == "avião"
    assert dm.mean == (0.4914, 0.4822, 0.4465)
    assert dm.std == (0.2470, 0.2435, 0.2616)
    # Default próprio: CIFAR-10 tem 50k de treino, contra os 60k do MNIST.
    assert dm.val_size == 5000


def test_base_requires_dataset_cls() -> None:
    """A base sem `dataset_cls` deve falhar com mensagem clara, não com AttributeError."""
    dm = BaseImageClfDataModule(data_dir="./data")
    with pytest.raises(NotImplementedError, match="dataset_cls"):
        dm.setup("fit")


@pytest.mark.parametrize("loader", ["train_dataloader", "val_dataloader", "test_dataloader"])
def test_dataloader_before_setup_raises_clear_error(loader: str) -> None:
    """Pedir um DataLoader antes de `setup` deve apontar a causa, não quebrar no torch.

    Sem o guard, o split `None` seguia para dentro do DataLoader e o erro
    resultante não dizia que faltou chamar setup().
    """
    dm = CIFAR10DataModule(data_dir="./data", num_workers=0)
    with pytest.raises(RuntimeError, match=r"setup\('fit'\)"):
        getattr(dm, loader)()


@pytest.mark.parametrize("dm_cls,data_subdir,sample_shape", DATAMODULES)
def test_setup_fit_populates_all_splits(
    dm_cls: type[BaseImageClfDataModule], data_subdir: str, sample_shape: tuple[int, ...]
) -> None:
    """`setup('fit')` deve popular os TRÊS splits, inclusive o de teste.

    O pipeline (`cvlab.tuning.rigorous`) chama apenas `setup('fit')` e em seguida
    usa `test_dataloader()`. Um `setup` guardado por `stage` deixaria
    `test_dataset` em None e quebraria a avaliação — este teste fecha essa porta.
    """
    if not (Path("./data") / data_subdir).exists():
        pytest.skip(f"Dataset {dm_cls.__name__} não encontrado em ./data/{data_subdir}.")

    dm = dm_cls(data_dir="./data", batch_size=64, num_workers=0)
    dm.setup("fit")

    assert dm.train_dataset is not None
    assert dm.val_dataset is not None
    assert dm.test_dataset is not None, "setup('fit') deixou o split de teste vazio"

    x, y = next(iter(dm.train_dataloader()))
    assert x.shape[1:] == sample_shape
    assert len(y) == 64


def test_mnist_split_sizes_if_data_exists() -> None:
    """Se o MNIST estiver baixado em ./data, confere os tamanhos dos splits."""
    if not (Path("./data") / "MNIST").exists():
        pytest.skip("Dataset MNIST não encontrado localmente em ./data.")

    dm = MNISTDataModule(data_dir="./data", batch_size=64, val_size=10000, num_workers=0)
    dm.setup("fit")

    assert len(dm.train_dataloader().dataset) == 50000
    assert len(dm.val_dataloader().dataset) == 10000
    assert len(dm.test_dataloader().dataset) == 10000


def test_cifar10_split_sizes_if_data_exists() -> None:
    """Se o CIFAR-10 estiver baixado em ./data, confere os splits 45k/5k/10k."""
    if not (Path("./data") / "cifar-10-batches-py").exists():
        pytest.skip("Dataset CIFAR-10 não encontrado localmente em ./data.")

    dm = CIFAR10DataModule(data_dir="./data", batch_size=64, num_workers=0)
    dm.setup("fit")

    assert len(dm.train_dataloader().dataset) == 45000
    assert len(dm.val_dataloader().dataset) == 5000
    assert len(dm.test_dataloader().dataset) == 10000


def test_augmentation_only_on_train() -> None:
    """Augmentation deve valer só no treino: mesmo índice varia no treino, não na validação.

    Garante que `_augment_ops` está de fato ligado ao split de treino (o CIFAR
    já teve uma versão em que as ops eram código morto) e que validação/teste
    permanecem determinísticos.
    """
    if not (Path("./data") / "cifar-10-batches-py").exists():
        pytest.skip("Dataset CIFAR-10 não encontrado localmente em ./data.")

    dm = CIFAR10DataModule(data_dir="./data", num_workers=0)
    dm.setup("fit")

    torch.manual_seed(0)
    train_a = dm.train_dataset[0][0]
    train_b = dm.train_dataset[0][0]
    assert not torch.allclose(train_a, train_b), "augmentation não está ativa no treino"

    val_a = dm.val_dataset[0][0]
    val_b = dm.val_dataset[0][0]
    assert torch.allclose(val_a, val_b), "validação não deve ter augmentation"
