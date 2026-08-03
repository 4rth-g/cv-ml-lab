"""Módulos de rede, fábrica de modelos e LitClassifier do cvlab."""

from cvlab.models.cnn import ConfigurableCNN
from cvlab.models.factory import arch_params, baseline_params, build_model, lit_params, suggest_params
from cvlab.models.lit_module import LitClassifier
from cvlab.models.mlp import MLP
from cvlab.models.perceptron import LinearPerceptron

__all__ = [
    "MLP",
    "ConfigurableCNN",
    "LinearPerceptron",
    "LitClassifier",
    "arch_params",
    "baseline_params",
    "build_model",
    "lit_params",
    "suggest_params",
]
