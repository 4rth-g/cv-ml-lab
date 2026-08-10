"""MNIST DataModule para o cvlab."""

from __future__ import annotations

from torchvision import datasets, transforms

from cvlab.data.base import BaseImageClfDataModule

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
MNIST_CLASSES = [str(i) for i in range(10)]


class MNISTDataModule(BaseImageClfDataModule):
    """DataModule para o dataset MNIST (28x28 em escala de cinza, 10 dígitos)."""

    dataset_cls = datasets.MNIST
    default_mean = MNIST_MEAN
    default_std = MNIST_STD
    default_classes = MNIST_CLASSES

    def _augment_ops(self) -> list:
        # Dígitos: rotação/translação leve; NUNCA flip (espelhar muda o dígito).
        return [transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))]
