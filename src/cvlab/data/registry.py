"""Registry de DataModules no cvlab."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from cvlab.data.base import BaseImageClfDataModule


def get_datamodule(cfg: DictConfig) -> BaseImageClfDataModule:
    """Instancia o DataModule a partir da configuração do Hydra (cfg.dataset)."""
    dm: BaseImageClfDataModule = hydra.utils.instantiate(cfg.dataset)
    return dm
