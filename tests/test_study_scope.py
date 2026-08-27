"""Trava do escopo do estudo Optuna, sem treinar nada.

O nome do estudo decide se uma busca é RETOMADA ou COMEÇADA. Como o estudo é
aberto com ``load_if_exists=True`` e o orçamento restante é
``n_trials - len(study.trials)``, dois datasets que colidam no mesmo nome fazem o
segundo pular a busca inteira e herdar a config do primeiro — sem erro, sem
aviso, e com o ``search_best_val_acc`` do manifest apontando para o dataset
errado. Foi o que aconteceu com os 18 subsets do MedMNIST, que compartilhavam o
nome ``medmnist`` derivado da classe do datamodule.
"""

from typing import Any

import pytest
from omegaconf import OmegaConf

from cvlab.tuning.search import study_tag


class FakeDM:
    """Só o que `study_tag` consome: a identidade canônica do dataset."""

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id


def cfg_for(target: str) -> Any:
    return OmegaConf.create({"model": {"_target_": target}})


CNN = cfg_for("cvlab.models.cnn.ConfigurableCNN")
RESNET = cfg_for("cvlab.models.resnet.ResNet")


def test_subsets_diferentes_nao_colidem() -> None:
    """A regressão concreta: três subsets do MedMNIST, três estudos.

    Antes, os três davam `medmnist_configurablecnn` e dividiam o mesmo SQLite.
    """
    tags = {study_tag(FakeDM(f"medmnist-{s}-28"), CNN) for s in ("breastmnist", "dermamnist", "organmnist3d")}
    assert len(tags) == 3


def test_resolucoes_diferentes_nao_colidem() -> None:
    """Busca feita em 28px não deve ser reaproveitada em 224px."""
    assert study_tag(FakeDM("medmnist-pathmnist-28"), CNN) != study_tag(FakeDM("medmnist-pathmnist-224"), CNN)


def test_arquiteturas_diferentes_nao_colidem() -> None:
    """O espaço de busca é por arquitetura; misturá-los não faria sentido."""
    assert study_tag(FakeDM("mnist"), CNN) != study_tag(FakeDM("mnist"), RESNET)


def test_mesmo_dataset_e_modelo_retomam_o_mesmo_estudo() -> None:
    """A outra metade do contrato: retomar é um recurso, não um acidente.

    Interromper um grid e relançá-lo tem de continuar de onde parou.
    """
    assert study_tag(FakeDM("mnist"), CNN) == study_tag(FakeDM("mnist"), CNN)


def test_tag_usa_o_dataset_id_e_nao_a_classe_do_datamodule() -> None:
    """Fixa a fonte do nome, que é a raiz do bug.

    `dataset_id` é o mesmo valor que nomeia o diretório do EDA e entra no
    manifest. Derivar o nome de qualquer outra coisa reabre a divergência.
    """
    assert study_tag(FakeDM("medmnist-breastmnist-28"), CNN) == "medmnist-breastmnist-28_configurablecnn"
    assert study_tag(FakeDM("mnist"), CNN) == "mnist_configurablecnn"


@pytest.mark.parametrize("dataset_id", ["mnist", "cifar10", "medmnist-chestmnist-64"])
def test_tag_serve_como_nome_de_arquivo(dataset_id: str) -> None:
    """A tag vira `optuna_study_<tag>.db` no disco."""
    tag = study_tag(FakeDM(dataset_id), CNN)
    assert "/" not in tag and " " not in tag
