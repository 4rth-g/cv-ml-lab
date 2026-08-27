"""DataModule base para classificação de imagens no cvlab."""

from __future__ import annotations

from typing import Any, ClassVar, Sequence

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms


class BaseImageClfDataModule(L.LightningDataModule):
    """LightningDataModule base para datasets de classificação de imagem.

    Expõe in_channels, num_classes e class_names.
    Aplica data augmentation APENAS no conjunto de treino. Validação e teste
    utilizam transformações simples sem augmentation.

    Uma subclasse concreta normalmente só declara os atributos de classe
    (``dataset_cls``, ``default_mean``, ``default_std``, ...) e sobrescreve
    `_augment_ops` com a política de augmentation do domínio — `prepare_data`
    e `setup` genéricos vêm de graça daqui.
    """

    #: Classe do ``torchvision.datasets`` a instanciar, com a assinatura usual
    #: ``(root, train, download, transform)``. ``None`` obriga a subclasse a
    #: sobrescrever `prepare_data` e `setup` (ex.: dataset fora do torchvision).
    dataset_cls: ClassVar[type | None] = None

    #: Valores usados quando o argumento correspondente do __init__ vem vazio.
    default_in_channels: ClassVar[int] = 1
    default_num_classes: ClassVar[int] = 10
    default_val_size: ClassVar[int] = 10_000
    default_mean: ClassVar[Sequence[float] | None] = None
    default_std: ClassVar[Sequence[float] | None] = None
    default_classes: ClassVar[Sequence[str] | None] = None

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        val_size: int | None = None,
        num_workers: int = 2,
        seed: int = 42,
        in_channels: int | None = None,
        num_classes: int | None = None,
        class_names: Sequence[str] | None = None,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        # Fontes de variância separadas (ver `configure_run_seeds`). Começam
        # iguais a `seed`, que é o comportamento histórico.
        self.split_seed = seed
        self.data_seed = seed
        self.val_size = val_size if val_size is not None else self.default_val_size
        self.in_channels = in_channels if in_channels is not None else self.default_in_channels
        self.num_classes = num_classes if num_classes is not None else self.default_num_classes
        self.class_names = class_names or self.default_classes or [str(i) for i in range(self.num_classes)]
        self.mean = mean or self.default_mean or (0.5,) * self.in_channels
        self.std = std or self.default_std or (0.5,) * self.in_channels

        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    #: ``True`` quando o dataset publica seu próprio split de validação, caso em
    #: que `split_seed` não tem efeito (ver `cvlab.data.med_mnist`).
    uses_official_split: ClassVar[bool] = False

    @property
    def dataset_id(self) -> str:
        """Identificador do dataset nos artefatos exportados (ex.: ``mnist``).

        Precisa vir de UM lugar só: `cvlab.export` grava este valor no manifest e
        `cvlab.eda` o usa para nomear o diretório do EDA. Se cada lado derivasse
        o nome por conta própria, os dois divergiriam e o join por
        ``(split, example_idx)`` não encontraria o par — silenciosamente, porque
        o diretório simplesmente não existiria.

        Subclasses com variantes (resolução, subset) devem sobrescrever.
        """
        return type(self).__name__.removesuffix("DataModule").lower()

    def configure_run_seeds(self, split_seed: int | None = None, data_seed: int | None = None) -> None:
        """Ajusta as fontes de variância que NÃO são a inicialização dos pesos.

        Separar isto de ``seed`` é o que permite medir a variância real de um
        resultado. Um multi-seed que varia só a inicialização mantém split e
        ordem de dados fixos e, por isso, **subestima** a variância — o intervalo
        de confiança sai estreito demais (Bouthillier et al., MLSys 2021).

        Chame ANTES de `setup` (o split) e antes de pedir o `train_dataloader`
        (a ordem). Passar ``None`` preserva o valor atual.
        """
        if split_seed is not None:
            self.split_seed = split_seed
        if data_seed is not None:
            self.data_seed = data_seed

    def _augment_ops(self) -> list:
        """Ops de data augmentation (aplicadas SÓ no treino). Sobrescreva por dataset.

        A política correta depende do domínio — ex.: dígitos NÃO podem ser
        espelhados (um '3'/'7' espelhado deixa de ser válido; '6'↔'9'). Por isso
        o base não assume nada (retorna []) e cada DataModule define a sua.
        """
        return []

    def get_transform(self, augment: bool = False) -> transforms.Compose:
        """Retorna as transformações para o dataset.

        Data augmentation (via `_augment_ops`, específica do dataset) só quando augment=True.
        """
        steps: list[Any] = list(self._augment_ops()) if augment else []
        steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(tuple(self.mean), tuple(self.std)),
            ]
        )
        return transforms.Compose(steps)

    def _require_dataset_cls(self) -> type:
        """Garante que a subclasse declarou ``dataset_cls`` antes de usá-lo."""
        if self.dataset_cls is None:
            raise NotImplementedError(
                f"{type(self).__name__} não define 'dataset_cls'. Declare a classe do "
                "torchvision.datasets a usar, ou sobrescreva prepare_data() e setup()."
            )
        return self.dataset_cls

    def _require_split(self, name: str) -> Dataset:
        """Devolve o split ``name`` já materializado, ou falha dizendo o porquê.

        Sem isto, chamar um ``*_dataloader()`` antes de `setup` passaria ``None``
        adiante e o erro só apareceria lá dentro do DataLoader, sem indicar a
        causa real.
        """
        split: Dataset | None = getattr(self, f"{name}_dataset")
        if split is None:
            raise RuntimeError(
                f"O split '{name}' não existe ainda em {type(self).__name__}. "
                "Chame setup('fit') antes de pedir o DataLoader."
            )
        return split

    def prepare_data(self) -> None:
        """Baixa o dataset (treino e teste) se ainda não estiver em ``data_dir``."""
        dataset_cls = self._require_dataset_cls()
        dataset_cls(root=self.data_dir, train=True, download=True)
        dataset_cls(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        """Materializa os splits de treino (com augmentation), validação e teste.

        Sem guarda por ``stage`` de propósito: o pipeline chama apenas
        ``setup("fit")`` e em seguida usa `test_dataloader` (ver
        `cvlab.tuning.rigorous`). Popular os três splits de uma vez é o que
        mantém esse uso válido.
        """
        dataset_cls = self._require_dataset_cls()
        tfm_aug = self.get_transform(augment=True)
        tfm_plain = self.get_transform(augment=False)

        self._setup_splits(
            dataset_cls(root=self.data_dir, train=True, download=False, transform=tfm_aug),
            dataset_cls(root=self.data_dir, train=True, download=False, transform=tfm_plain),
            dataset_cls(root=self.data_dir, train=False, download=False, transform=tfm_plain),
        )

    def _setup_splits(self, full_aug_dataset: Dataset, full_plain_dataset: Dataset, test_ds: Dataset) -> None:
        """Divide o dataset em treino (com augmentation) e validação (sem augmentation)."""
        n_total = len(full_aug_dataset)
        n_train = n_total - self.val_size
        gen = torch.Generator().manual_seed(self.split_seed)

        train_split, val_split = random_split(full_aug_dataset, [n_train, self.val_size], generator=gen)
        # Validação usa exatamente os mesmos índices, mas do dataset sem augmentation
        val_set = Subset(full_plain_dataset, val_split.indices)

        self.train_dataset = train_split
        self.val_dataset = val_set
        self.test_dataset = test_ds

    def train_dataloader(self) -> DataLoader:
        gen = torch.Generator().manual_seed(self.data_seed)
        return DataLoader(
            self._require_split("train"),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            generator=gen,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._require_split("val"),
            batch_size=256,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._require_split("test"),
            batch_size=256,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
