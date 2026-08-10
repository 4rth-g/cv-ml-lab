"""CIFAR-10 DataModule para o cvlab."""

from __future__ import annotations

from torchvision import datasets, transforms

from cvlab.data.base import BaseImageClfDataModule

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
CIFAR_CLASSES = [
    "avião",
    "automóvel",
    "pássaro",
    "gato",
    "cervo",
    "cachorro",
    "sapo",
    "cavalo",
    "navio",
    "caminhão",
]


class CIFAR10DataModule(BaseImageClfDataModule):
    """DataModule para o dataset CIFAR-10 (32x32 RGB, 10 classes)."""

    dataset_cls = datasets.CIFAR10
    default_in_channels = 3
    default_val_size = 5_000
    default_mean = CIFAR_MEAN
    default_std = CIFAR_STD
    default_classes = CIFAR_CLASSES

    def _augment_ops(self) -> list:
        # Objetos naturais: crop com padding + flip horizontal são válidos (um
        # gato espelhado continua um gato). É a receita padrão do CIFAR-10.
        return [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
