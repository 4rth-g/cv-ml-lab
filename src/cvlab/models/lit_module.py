"""LightningModule wrapper para modelos de classificação no cvlab."""

from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch import nn, optim
from torchmetrics.classification import MulticlassAccuracy


class LitClassifier(L.LightningModule):
    """LightningModule genérico para classificação multiclasse."""

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        lr: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay

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
        opt_lower = self.optimizer_name.lower()
        if opt_lower == "adam":
            return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if opt_lower == "adamw":
            return optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if opt_lower == "sgd":
            return optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        raise ValueError(f"Otimizador '{self.optimizer_name}' desconhecido. Use adam, adamw ou sgd.")
