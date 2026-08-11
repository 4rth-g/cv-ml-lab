"""LightningModule wrapper para modelos de classificação no cvlab."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import lightning as L
import torch
from torch import nn, optim
from torchmetrics.classification import MulticlassAccuracy

#: Otimizadores disponíveis, por nome. Tabela declarativa em vez de if/elif: o
#: nome é o que o Optuna sugere (`suggest_categorical` precisa de valores
#: simples), e acrescentar um otimizador é acrescentar uma linha aqui mais uma
#: entrada em `choices` no search_space — sem tocar em `configure_optimizers`.
OPTIMIZERS: dict[str, Callable[..., optim.Optimizer]] = {
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "sgd": partial(optim.SGD, momentum=0.9),
}

#: Schedulers de learning rate, por nome. Recebem (optimizer, epochs) e devolvem
#: um scheduler por época. `none` mantém o LR constante — é o default, para que
#: configurações antigas continuem reproduzindo exatamente o mesmo treino.
#:
#: Deixar o scheduler como dimensão de busca (e não como constante do código) é
#: deliberado: quem decide se o cosine ajuda passa a ser o método — busca,
#: multi-seed e teste pareado — em vez de uma escolha embutida no fonte.
SCHEDULERS: dict[str, Callable[[optim.Optimizer, int], Any] | None] = {
    "none": None,
    "cosine": lambda opt, epochs: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs)),
    "step": lambda opt, epochs: optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.1),
}


class LitClassifier(L.LightningModule):
    """LightningModule genérico para classificação multiclasse.

    Agnóstico à arquitetura: recebe qualquer `nn.Module` que mapeie imagem em
    logits. Otimizador e scheduler são escolhidos por nome (ver `OPTIMIZERS` e
    `SCHEDULERS`), o que permite que ambos sejam hiperparâmetros de busca.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        lr: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        scheduler: str = "none",
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.train_acc(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.val_acc(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.test_acc(logits, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Any:
        """Monta otimizador e (opcionalmente) scheduler a partir dos nomes configurados."""
        opt_name = self.optimizer_name.lower()
        if opt_name not in OPTIMIZERS:
            raise ValueError(f"Otimizador '{self.optimizer_name}' desconhecido. Use um de: {sorted(OPTIMIZERS)}.")
        optimizer = OPTIMIZERS[opt_name](self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        sched_name = self.scheduler_name.lower()
        if sched_name not in SCHEDULERS:
            raise ValueError(f"Scheduler '{self.scheduler_name}' desconhecido. Use um de: {sorted(SCHEDULERS)}.")
        make_scheduler = SCHEDULERS[sched_name]
        if make_scheduler is None:
            return optimizer

        # max_epochs vem do Trainer, disponível aqui: o scheduler precisa saber o
        # horizonte (T_max do cosine, step_size do step) e amarrá-lo ao orçamento
        # do run evita que mudar `final_epochs` desalinhe silenciosamente o decay.
        epochs = getattr(self.trainer, "max_epochs", None) or 1
        return {"optimizer": optimizer, "lr_scheduler": make_scheduler(optimizer, int(epochs))}
