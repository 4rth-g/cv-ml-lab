"""Módulos de busca e comparação estatística rigorosa do cvlab."""

from cvlab.tuning.rigorous import rigorous_compare
from cvlab.tuning.search import optuna_search

__all__ = [
    "optuna_search",
    "rigorous_compare",
]
