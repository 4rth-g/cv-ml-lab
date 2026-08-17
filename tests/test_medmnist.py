"""Testes do MedMNISTDataModule parametrizado.

A maioria não precisa de download: o contrato importante é que os metadados
venham de ``medmnist.INFO`` e que os subsets fora de escopo falhem cedo.
"""

from pathlib import Path

import pytest
from medmnist import INFO

from cvlab.data.med_mnist import (
    SUPPORTED_SUBSETS,
    MedMNISTDataModule,
    _to_scalar_label,
)


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
def test_all_supported_subsets_instantiate(subset: str) -> None:
    """Os 11 subsets em escopo constroem sem download e são single-label."""
    dm = MedMNISTDataModule(subset=subset, data_dir="./data")
    assert dm.num_classes >= 2
    assert dm.in_channels in (1, 3)
    assert "multi-label" not in dm.task
    assert len(dm.mean) == dm.in_channels
    assert len(dm.std) == dm.in_channels


def test_chestmnist_rejected_with_reason() -> None:
    """chestmnist é multi-label: deve falhar no __init__, não no meio do treino."""
    with pytest.raises(ValueError, match="multi-label"):
        MedMNISTDataModule(subset="chestmnist")


@pytest.mark.parametrize("subset", ["organmnist3d", "nodulemnist3d", "synapsemnist3d"])
def test_3d_subsets_rejected_with_reason(subset: str) -> None:
    """Os volumétricos não passam por Conv2d: erro explícito citando o motivo."""
    with pytest.raises(ValueError, match="volumétrico"):
        MedMNISTDataModule(subset=subset)


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
