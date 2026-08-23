"""A definição única de acurácia no lado Python.

Este módulo existe por causa de um bug real. As métricas do treino vinham de
``torchmetrics.MulticlassAccuracy(num_classes=N)``, cujo default de ``average``
é ``"macro"`` (herdado de ``MulticlassStatScores``) — ou seja, acurácia
balanceada. Já a acurácia de teste era calculada à mão como
``np.mean(preds == targets)``, que é micro. As duas iam para a MESMA coluna
``acc`` do ``seed_metrics.csv``, uma no split ``val`` e outra no ``test``, e o
lado R comparava as duas como se fossem a mesma quantidade.

A correção tem duas partes, e ambas passam por aqui:

1. **O que se reporta** é sempre :func:`accuracy` (micro), em todo split. É o
   ACC publicado pelo benchmark do MedMNIST e é o mesmo número que
   ``analysis/R/metrics.R::summary_metrics()`` recomputa das predições — o que
   torna o contrato verificável em vez de suposto.
2. **O que decide** é :func:`objective_score`, escolhido por ``cfg.tuning.objective``
   e registrado no manifest. Desbalanceamento (DermaMNIST tem razão 58,7) é um
   problema de critério de seleção, não de métrica de relatório, e tratá-lo aqui
   torna a escolha explícita em vez de um default acidental de biblioteca.

Sem torch de propósito: são funções sobre listas de inteiros, o que as deixa
utilizáveis pelo eixo tabular (árvores) sem arrastar o stack de deep learning.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def accuracy(preds: Sequence[int], targets: Sequence[int]) -> float:
    """Acurácia simples (micro): fração de exemplos classificados corretamente.

    É a métrica de RELATÓRIO do cvlab — a coluna ``acc`` de ``seed_metrics.csv``
    é esta função em todo split.
    """
    if len(preds) != len(targets):
        raise ValueError(f"preds e targets com tamanhos diferentes: {len(preds)} != {len(targets)}.")
    if not targets:
        raise ValueError("não é possível calcular acurácia sobre zero exemplos.")
    return float(np.mean(np.asarray(preds) == np.asarray(targets)))


def balanced_accuracy(preds: Sequence[int], targets: Sequence[int]) -> float:
    """Acurácia balanceada: recall médio entre as classes PRESENTES em ``targets``.

    A média é sobre as classes que ocorrem nos rótulos, não sobre ``num_classes``.
    Uma classe que o dataset não contém neste split não tem recall definido, e
    contá-la como zero puxaria a métrica para baixo por um motivo que não é do
    modelo. É a mesma convenção do ``balanced_accuracy_score`` do scikit-learn e
    do ``per_class_metrics()`` em ``analysis/R/metrics.R``, que reserva ``NA``
    para classe ausente dos rótulos.
    """
    if len(preds) != len(targets):
        raise ValueError(f"preds e targets com tamanhos diferentes: {len(preds)} != {len(targets)}.")
    if not targets:
        raise ValueError("não é possível calcular acurácia balanceada sobre zero exemplos.")
    p = np.asarray(preds)
    t = np.asarray(targets)
    recalls = [float(np.mean(p[t == c] == c)) for c in np.unique(t)]
    return float(np.mean(recalls))


#: Métricas elegíveis como objetivo de busca e critério de seleção, por nome.
#: Tabela declarativa pelo mesmo motivo de `OPTIMIZERS` em `lit_module.py`: o
#: nome vem da config, e acrescentar um objetivo é acrescentar uma linha.
OBJECTIVES: dict[str, Callable[[Sequence[int], Sequence[int]], float]] = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
}

#: Chave em `trainer.callback_metrics` que o `LitClassifier` loga para cada
#: objetivo. É o que permite escolher a melhor ÉPOCA pelo mesmo critério que
#: escolhe a melhor CONFIG — sem isso, o early stopping otimizaria uma métrica e
#: a seleção final, outra.
OBJECTIVE_METRIC_SUFFIX: dict[str, str] = {
    "accuracy": "acc",
    "balanced_accuracy": "bal_acc",
}


def objective_score(preds: Sequence[int], targets: Sequence[int], objective: str = "accuracy") -> float:
    """Calcula a métrica de SELEÇÃO nomeada por ``objective``.

    Nunca vai para a coluna ``acc`` do export: aquilo é sempre :func:`accuracy`.
    Este valor decide qual época restaurar, qual trial vence a busca e se a
    config tunada substitui a baseline.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"Objetivo '{objective}' desconhecido. Use um de: {sorted(OBJECTIVES)}.")
    return OBJECTIVES[objective](preds, targets)


def objective_metric_key(objective: str, stage: str = "val") -> str:
    """Nome da métrica logada correspondente a ``objective`` (ex.: ``val_bal_acc``)."""
    if objective not in OBJECTIVE_METRIC_SUFFIX:
        raise ValueError(f"Objetivo '{objective}' desconhecido. Use um de: {sorted(OBJECTIVE_METRIC_SUFFIX)}.")
    return f"{stage}_{OBJECTIVE_METRIC_SUFFIX[objective]}"
