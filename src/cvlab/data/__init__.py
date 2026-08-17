"""Módulos de dados e Datamodules do cvlab."""

from cvlab.data.base import BaseImageClfDataModule
from cvlab.data.cifar10 import CIFAR10DataModule
from cvlab.data.fashion_mnist import FashionMNISTDataModule
from cvlab.data.med_mnist import SUPPORTED_SUBSETS, MedMNISTDataModule
from cvlab.data.mnist import MNISTDataModule
from cvlab.data.registry import get_datamodule

__all__ = [
    "BaseImageClfDataModule",
    "MNISTDataModule",
    "FashionMNISTDataModule",
    "CIFAR10DataModule",
    "MedMNISTDataModule",
    "SUPPORTED_SUBSETS",
    "get_datamodule",
]
