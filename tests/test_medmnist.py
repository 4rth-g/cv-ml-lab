"""Testes do MedMNISTDataModule parametrizado.

A maioria não precisa de download: o contrato importante é que os metadados
venham de ``medmnist.INFO`` e que os subsets fora de escopo falhem cedo.
"""

from pathlib import Path

import pytest
from medmnist import INFO

from cvlab.data.med_mnist import (
    MULTILABEL_SUBSETS,
    SINGLE_LABEL_2D,
    SUPPORTED_SUBSETS,
    VOLUMETRIC_SUBSETS,
    MedMNISTDataModule,
    _to_multilabel,
    _to_scalar_label,
    _volume_to_channels,
)


def test_multilabel_and_3d_are_supported() -> None:
    """As três famílias entram pela mesma classe parametrizada."""
    assert "chestmnist" in MULTILABEL_SUBSETS
    assert len(VOLUMETRIC_SUBSETS) == 6
    assert len(SUPPORTED_SUBSETS) == 18


def test_metadata_comes_from_info() -> None:
    """in_channels/num_classes/class_names são derivados do INFO, não da config."""
    dm = MedMNISTDataModule(subset="dermamnist", data_dir="./data")
    info = INFO["dermamnist"]

    assert dm.in_channels == info["n_channels"] == 3
    assert dm.num_classes == len(info["label"]) == 7
    assert len(dm.class_names) == 7
    assert dm.task == "multi-class"


def test_class_names_follow_label_index_order() -> None:
    """A ordem de class_names deve ser a dos índices de classe, não a de iteração do dict.

    Em organamnist isso é verificável: as classes lateralizadas ocupam índices
    adjacentes e trocá-las inverteria a leitura de qualquer matriz de confusão.
    """
    dm = MedMNISTDataModule(subset="organamnist", data_dir="./data")
    assert dm.class_names[0] == "bladder"
    assert dm.class_names[1] == "femur-left"
    assert dm.class_names[2] == "femur-right"
    assert dm.num_classes == 11


@pytest.mark.parametrize("subset", SUPPORTED_SUBSETS)
def test_all_subsets_instantiate(subset: str) -> None:
    """Os 18 subsets constroem sem download, com mean/std do tamanho certo."""
    dm = MedMNISTDataModule(subset=subset, data_dir="./data")
    assert dm.num_classes >= 2
    assert len(dm.mean) == dm.in_channels
    assert len(dm.std) == dm.in_channels


@pytest.mark.parametrize("subset", SINGLE_LABEL_2D)
def test_single_label_2d_shape(subset: str) -> None:
    dm = MedMNISTDataModule(subset=subset, data_dir="./data")
    assert dm.in_channels in (1, 3)
    assert not dm.is_3d and not dm.is_multilabel


def test_chestmnist_is_multilabel() -> None:
    """14 achados que coexistem: exige BCEWithLogitsLoss, não CrossEntropy."""
    dm = MedMNISTDataModule(subset="chestmnist", data_dir="./data")
    assert dm.is_multilabel
    assert dm.num_classes == 14
    assert dm._target_transform() is _to_multilabel


@pytest.mark.parametrize("subset", VOLUMETRIC_SUBSETS)
def test_volumetric_uses_slices_as_channels(subset: str) -> None:
    """As 28 fatias viram canais, o que permite reusar as arquiteturas Conv2d."""
    dm = MedMNISTDataModule(subset=subset, data_dir="./data")
    assert dm.is_3d
    assert dm.in_channels == 28
    # Augmentation 2D não se aplica a um tensor de canais.
    assert dm._augment_ops() == []


def test_volumetric_rejects_other_resolutions() -> None:
    """Os 3D só existem em 28³; pedir outra resolução tem que falhar cedo."""
    with pytest.raises(ValueError, match="28"):
        MedMNISTDataModule(subset="organmnist3d", size=64)


def test_volume_to_channels_squeezes_and_normalizes() -> None:
    import numpy as np

    vol = np.full((1, 28, 28, 28), 255, dtype=np.uint8)
    t = _volume_to_channels(vol)
    assert tuple(t.shape) == (28, 28, 28)
    assert float(t.max()) == 1.0


def test_multilabel_target_is_float_vector() -> None:
    t = _to_multilabel([0, 1, 1, 0])
    assert tuple(t.shape) == (4,)
    assert t.dtype.is_floating_point


def test_unknown_subset_rejected() -> None:
    with pytest.raises(ValueError, match="desconhecido"):
        MedMNISTDataModule(subset="mednist")


def test_invalid_size_rejected() -> None:
    with pytest.raises(ValueError, match="size=32"):
        MedMNISTDataModule(subset="pathmnist", size=32)


def test_dataset_id_is_stable_and_includes_size() -> None:
    """dataset_id é a chave dos artefatos de export: precisa distinguir subset E resolução."""
    assert MedMNISTDataModule(subset="bloodmnist").dataset_id == "medmnist-bloodmnist-28"
    assert MedMNISTDataModule(subset="bloodmnist", size=64).dataset_id == "medmnist-bloodmnist-64"


def test_flip_only_where_orientation_is_free() -> None:
    """Flip é augmentation válida em microscopia, e ERRADA onde há lateralidade.

    organamnist tem femur-left/femur-right e kidney-left/kidney-right: espelhar
    converteria uma classe na outra e o rótulo passaria a mentir.
    """
    path_ops = [type(op).__name__ for op in MedMNISTDataModule(subset="pathmnist")._augment_ops()]
    assert "RandomHorizontalFlip" in path_ops

    organ_ops = [type(op).__name__ for op in MedMNISTDataModule(subset="organamnist")._augment_ops()]
    assert "RandomHorizontalFlip" not in organ_ops
    assert "RandomVerticalFlip" not in organ_ops
    assert "RandomCrop" in organ_ops

    chest_ops = [type(op).__name__ for op in MedMNISTDataModule(subset="pneumoniamnist")._augment_ops()]
    assert "RandomHorizontalFlip" not in chest_ops


def test_to_scalar_label_unwraps_medmnist_target() -> None:
    """O MedMNIST entrega o rótulo como array (1,); a CrossEntropyLoss quer um escalar."""
    assert _to_scalar_label([3]) == 3
    assert isinstance(_to_scalar_label([3]), int)


def test_official_val_split_is_used_not_random_split() -> None:
    """Os splits devem bater com INFO['n_samples'] — prova de que o split oficial foi usado.

    Se `setup` caísse no `_setup_splits` da base, a validação seria fatiada do
    treino e os três tamanhos divergiriam do publicado, quebrando a
    comparabilidade com o paper do MedMNIST.
    """
    subset = "breastmnist"
    if not (Path("./data") / f"{subset}.npz").exists():
        pytest.skip(f"{subset} não encontrado em ./data. Rode prepare_data() antes.")

    dm = MedMNISTDataModule(subset=subset, data_dir="./data", num_workers=0)
    dm.setup("fit")

    expected = INFO[subset]["n_samples"]
    assert len(dm.train_dataloader().dataset) == expected["train"]
    assert len(dm.val_dataloader().dataset) == expected["val"]
    assert len(dm.test_dataloader().dataset) == expected["test"]


def test_sample_shape_and_label_type() -> None:
    """Uma amostra deve sair como (C, H, W) e o rótulo como inteiro escalar."""
    subset = "breastmnist"
    if not (Path("./data") / f"{subset}.npz").exists():
        pytest.skip(f"{subset} não encontrado em ./data. Rode prepare_data() antes.")

    dm = MedMNISTDataModule(subset=subset, data_dir="./data", num_workers=0)
    dm.setup("fit")

    x, y = dm.test_dataset[0]
    assert x.shape == (1, 28, 28)
    assert isinstance(y, int)
