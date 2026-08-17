"""MedMNIST v2 DataModule para o cvlab (https://medmnist.com/).

O MedMNIST v2 é uma *coleção* de 18 datasets biomédicos padronizados, não um
dataset único. Este módulo expõe os **11 subsets 2D de rótulo único** através de
uma classe parametrizada: trocar de dataset é trocar o argumento ``subset``, não
escrever código.

    just train cnn medmnist dataset.subset=dermamnist

Ficam de fora, por decisão de escopo:

- ``chestmnist``: multi-label (14 rótulos por exemplo). Exigiria
  ``BCEWithLogitsLoss`` e métrica multilabel no :class:`~cvlab.models.lit_module.LitClassifier`,
  além de quebrar o schema de export (``y_true`` escalar).
- os 6 subsets ``*3d``: volumes 28³, incompatíveis com as arquiteturas ``Conv2d``
  do cvlab. É família de modelo nova, não parametrização.

Ambos falham cedo, no ``__init__``, com mensagem explícita.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import medmnist
from medmnist import INFO
from torch.utils.data import Dataset
from torchvision import transforms

from cvlab.data.base import BaseImageClfDataModule

#: Os 11 subsets 2D de rótulo único do MedMNIST v2 (ver docstring do módulo).
SUPPORTED_SUBSETS = (
    "pathmnist",
    "dermamnist",
    "octmnist",
    "pneumoniamnist",
    "retinamnist",
    "breastmnist",
    "bloodmnist",
    "tissuemnist",
    "organamnist",
    "organcmnist",
    "organsmnist",
)

#: Resoluções publicadas pelo MedMNIST+ (medmnist >= 3.0). As arquiteturas do
#: cvlab são agnósticas à resolução (LazyLinear / AdaptiveAvgPool2d), então
#: qualquer uma destas roda sem tocar em código.
SUPPORTED_SIZES = (28, 64, 128, 224)

#: Subsets em que espelhar a imagem preserva o rótulo: microscopia e
#: dermatoscopia não têm orientação canônica.
#:
#: Nos demais o flip horizontal é ATIVAMENTE ERRADO, não apenas inútil. Em
#: ``organ{a,c,s}mnist`` as classes incluem ``femur-left``/``femur-right``,
#: ``kidney-left``/``kidney-right`` e ``lung-left``/``lung-right``: espelhar
#: converte uma classe na outra, e o rótulo passa a mentir. Em radiografia
#: (``pneumoniamnist``) a lateralidade também é diagnóstica.
_FLIP_SAFE = frozenset({"pathmnist", "dermamnist", "bloodmnist", "tissuemnist"})


def _to_scalar_label(target: Any) -> int:
    """Converte o rótulo ``(1,)`` do MedMNIST no escalar que a CrossEntropyLoss espera.

    Função de módulo, e não ``lambda``, de propósito: ``target_transform`` é
    serializada para os workers do DataLoader (``num_workers > 0``) e uma lambda
    não é picklável.
    """
    return int(target[0])


class MedMNISTDataModule(BaseImageClfDataModule):
    """DataModule parametrizado para os subsets 2D de rótulo único do MedMNIST v2.

    ``in_channels``, ``num_classes`` e ``class_names`` são derivados de
    ``medmnist.INFO[subset]``, nunca escritos à mão na config: o metadado é a
    fonte da verdade, e um subset novo não pode divergir dela.

    Args:
        subset: Um de :data:`SUPPORTED_SUBSETS`.
        size: Resolução, um de :data:`SUPPORTED_SIZES`.
        mean/std: Normalização. O default do MedMNIST é 0.5 por canal, que é o
            que os benchmarks publicados usam; sobrescreva para usar as
            estatísticas do próprio subset.
    """

    #: Deliberadamente ``None``: a assinatura do medmnist é ``(split=...)``, não
    #: ``(train=bool)``, então `prepare_data` e `setup` são sobrescritos aqui.
    dataset_cls = None

    #: Não usado: o MedMNIST traz split de validação OFICIAL, então nada é
    #: fatiado do treino. Ver `setup`.
    default_val_size = 0

    #: O split vem do próprio dataset, então `split_seed` não tem efeito aqui.
    uses_official_split = True

    def __init__(
        self,
        subset: str = "pathmnist",
        size: int = 28,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 2,
        seed: int = 42,
        mean: Any = None,
        std: Any = None,
    ) -> None:
        info = self._require_supported(subset)
        if size not in SUPPORTED_SIZES:
            raise ValueError(f"size={size} inválido para MedMNIST. Use um de: {list(SUPPORTED_SIZES)}.")

        n_channels = int(info["n_channels"])
        labels = info["label"]
        # O dict do INFO é {"0": nome, "1": nome, ...}; a ordem numérica é a
        # ordem dos índices de classe, então iterar o dict direto seria frágil.
        class_names = [labels[str(i)] for i in range(len(labels))]

        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            val_size=0,
            num_workers=num_workers,
            seed=seed,
            in_channels=n_channels,
            num_classes=len(class_names),
            class_names=class_names,
            mean=mean or (0.5,) * n_channels,
            std=std or (0.5,) * n_channels,
        )
        self.subset = subset
        self.size = size
        self.task = str(info["task"])
        self.n_samples = dict(info["n_samples"])

    @staticmethod
    def _require_supported(subset: str) -> dict[str, Any]:
        """Valida o subset no ``__init__``, e não no meio do treino."""
        if subset in SUPPORTED_SUBSETS:
            return INFO[subset]

        if subset == "chestmnist":
            raise ValueError(
                "chestmnist é multi-label (14 rótulos por exemplo) e está fora do escopo do cvlab: "
                "exigiria BCEWithLogitsLoss e métrica multilabel no LitClassifier. "
                f"Use um de: {list(SUPPORTED_SUBSETS)}."
            )
        if subset.endswith("3d"):
            raise ValueError(
                f"{subset} é volumétrico (28³) e as arquiteturas do cvlab são Conv2d. "
                f"Use um de: {list(SUPPORTED_SUBSETS)}."
            )
        raise ValueError(f"Subset MedMNIST desconhecido: {subset!r}. Use um de: {list(SUPPORTED_SUBSETS)}.")

    @property
    def dataset_id(self) -> str:
        """Identificador estável do dataset para os artefatos de export (ex.: ``medmnist-dermamnist-28``)."""
        return f"medmnist-{self.subset}-{self.size}"

    def _medmnist_cls(self) -> type:
        """Classe concreta do subset (ex.: ``medmnist.DermaMNIST``)."""
        return getattr(medmnist, str(INFO[self.subset]["python_class"]))

    def _augment_ops(self) -> list:
        """Augmentation por subset. Ver :data:`_FLIP_SAFE` para o porquê do flip ser restrito."""
        ops: list = [transforms.RandomCrop(self.size, padding=max(2, self.size // 7))]
        if self.subset in _FLIP_SAFE:
            ops.extend([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()])
        return ops

    def prepare_data(self) -> None:
        """Baixa os três splits oficiais se ausentes. O medmnist exige que ``root`` exista."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        cls = self._medmnist_cls()
        for split in ("train", "val", "test"):
            cls(split=split, root=self.data_dir, download=True, size=self.size)

    def setup(self, stage: str | None = None) -> None:
        """Materializa os três splits OFICIAIS do MedMNIST.

        Não usa `BaseImageClfDataModule._setup_splits`, que fatiaria validação do
        treino com ``random_split``: o MedMNIST publica seu próprio split de
        validação, e ignorá-lo custaria a comparabilidade com os números do
        paper. É por isso que ``val_size`` não tem efeito aqui.
        """
        cls = self._medmnist_cls()
        tfm_aug = self.get_transform(augment=True)
        tfm_plain = self.get_transform(augment=False)

        def build(split: str, transform: Any) -> Dataset:
            return cls(
                split=split,
                root=self.data_dir,
                download=False,
                transform=transform,
                target_transform=_to_scalar_label,
                size=self.size,
            )

        self.train_dataset = build("train", tfm_aug)
        self.val_dataset = build("val", tfm_plain)
        self.test_dataset = build("test", tfm_plain)
