"""Fashion-MNIST DataModule para o cvlab."""

from __future__ import annotations

from torchvision import datasets, transforms

from cvlab.data.base import BaseImageClfDataModule

FASHION_MEAN = (0.2860,)
FASHION_STD = (0.3530,)
FASHION_CLASSES = [
    "Camiseta/top",
    "Calça",
    "Pulôver",
    "Vestido",
    "Casaco",
    "Sandália",
    "Camisa",
    "Tênis",
    "Bolsa",
    "Bota",
]


class FashionMNISTDataModule(BaseImageClfDataModule):
    """DataModule para o dataset Fashion-MNIST (28x28 em escala de cinza, 10 peças)."""

    dataset_cls = datasets.FashionMNIST
    default_mean = FASHION_MEAN
    default_std = FASHION_STD
    default_classes = FASHION_CLASSES

    def _augment_ops(self) -> list:
        # Roupas toleram flip horizontal + crop leve (são ~simétricas).
        return [transforms.RandomCrop(28, padding=2), transforms.RandomHorizontalFlip()]
