# Dados

DataModules de classificação de imagem. O base define o contrato completo —
download, splits treino/validação/teste e o ponto de extensão da augmentation.
Cada dataset especializa apenas os atributos de classe (`dataset_cls`, mean/std,
classes) e a política de augmentation em `_augment_ops`.

## Base

::: cvlab.data.base

## Registry

::: cvlab.data.registry

## MNIST

::: cvlab.data.mnist

## Fashion-MNIST

::: cvlab.data.fashion_mnist

## CIFAR-10

::: cvlab.data.cifar10
