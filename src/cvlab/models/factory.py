"""Fábrica de modelos e utilidades de configuração compartilhadas pelo tuning.

Separa a configuração em dois grupos, permitindo que qualquer arquitetura seja
instanciada e otimizada apenas por configuração (sem código específico no tuning):

- Parâmetros de TREINO (comuns a toda arquitetura): ``lr``, ``optimizer``,
  ``weight_decay`` (passados ao :class:`LitClassifier`) e ``batch_size``
  (usado no DataLoader da busca).
- Parâmetros de ARQUITETURA (específicos do modelo): tudo o mais, repassado
  ao construtor do modelo via ``_target_`` (Hydra ``instantiate``).
"""

from __future__ import annotations

from typing import Any

from hydra.utils import instantiate
from torch import nn

_LIT_KEYS = ("lr", "optimizer", "weight_decay")
_LOADER_KEYS = ("batch_size",)

#: Chaves de hiperparâmetro que são de treino (não vão ao construtor do modelo).
TRAIN_KEYS = frozenset(_LIT_KEYS + _LOADER_KEYS)


def arch_params(config: dict[str, Any]) -> dict[str, Any]:
    """Extrai os parâmetros de arquitetura (tudo que não é treino nem ``_target_``)."""
    return {k: v for k, v in config.items() if k not in TRAIN_KEYS and k != "_target_"}


def lit_params(config: dict[str, Any]) -> dict[str, Any]:
    """Extrai os parâmetros do :class:`LitClassifier` presentes na config."""
    return {k: config[k] for k in _LIT_KEYS if k in config}


def build_model(config: dict[str, Any], in_channels: int, num_classes: int) -> nn.Module:
    """Instancia o modelo a partir de ``config['_target_']`` + parâmetros de arquitetura.

    Todos os modelos do cvlab aceitam ``in_channels`` e ``num_classes``; os demais
    campos de arquitetura são específicos de cada um (ver ``configs/model/``).
    """
    target = str(config["_target_"])
    return instantiate(
        {"_target_": target},
        in_channels=in_channels,
        num_classes=num_classes,
        **arch_params(config),
    )


def suggest_params(trial: Any, search_space: Any) -> dict[str, Any]:
    """Traduz um ``search_space`` tipado (Hydra) em sugestões do Optuna.

    Cada entrada tem a forma ``{type: categorical|float|int, ...}``:

    - ``categorical``: ``{type: categorical, choices: [...]}``
    - ``float``: ``{type: float, low: x, high: y, log: bool}``
    - ``int``: ``{type: int, low: x, high: y, log: bool}``
    """
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec["type"]
        if kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        elif kind == "float":
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
            )
        elif kind == "int":
            params[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log", False))
            )
        else:
            raise ValueError(f"Tipo de parâmetro desconhecido em search_space: {kind!r}")
    return params


def baseline_params(cfg: Any) -> dict[str, Any]:
    """Monta os parâmetros baseline (valores de ``cfg.model``) restritos ao search_space.

    Usado para enfileirar a baseline como primeiro trial do Optuna, garantindo que
    o melhor da busca não fique abaixo da baseline. Para floats em escala log,
    limita o valor ao piso do intervalo (ex.: weight_decay 0.0 -> low).
    """
    params: dict[str, Any] = {}
    for name, spec in cfg.tuning.search_space.items():
        val = cfg.model[name]
        if spec["type"] == "float":
            val = float(val)
            if bool(spec.get("log", False)):
                val = max(val, float(spec["low"]))
        elif spec["type"] == "int":
            val = int(val)
        params[name] = val
    return params
