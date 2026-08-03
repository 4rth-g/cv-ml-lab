"""Módulo de busca de hiperparâmetros com Optuna para o cvlab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, Subset

from cvlab.data.base import BaseImageClfDataModule
from cvlab.models.cnn import ConfigurableCNN
from cvlab.models.lit_module import LitClassifier
from cvlab.xpu import build_trainer

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optuna_search(datamodule: BaseImageClfDataModule, cfg: Any) -> optuna.Study:
    """Executa a busca de hiperparâmetros usando Optuna.

    Enfileira a configuração baseline no primeiro trial, garantindo por construção
    que best_val_acc >= baseline_val_acc na etapa de busca.
    """
    datamodule.setup("fit")

    subset_size = min(cfg.tuning.gs_subset_size, len(datamodule.train_dataset))
    L.seed_everything(cfg.seed, workers=True)
    indices = list(range(len(datamodule.train_dataset)))
    import torch

    perm_indices = torch.randperm(len(indices), generator=torch.Generator().manual_seed(cfg.seed))[
        :subset_size
    ].tolist()
    gs_train_dataset = Subset(datamodule.train_dataset, perm_indices)

    output_dir = Path(cfg.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Escopa o estudo pelo dataset p/ não colidir entre datasets no mesmo output_dir
    ds = str(cfg.dataset.get("_target_", "run")).split(".")[-1].replace("DataModule", "").lower() or "run"
    db_path = output_dir / f"optuna_study_{ds}.db"

    study = optuna.create_study(
        study_name=f"cvlab_optuna_{ds}",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=cfg.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    # Configuração baseline extraída da config do modelo
    baseline_params = {
        "num_filters": cfg.model.num_filters,
        "dropout_rate": float(cfg.model.dropout_rate),
        "fc_units": cfg.model.fc_units,
        "n_conv_blocks": cfg.model.get("n_conv_blocks", 2),
        "lr": float(cfg.model.lr),
        "batch_size": cfg.model.batch_size,
        "optimizer": str(cfg.model.optimizer),
        "weight_decay": max(float(cfg.model.weight_decay), 1e-6),
    }

    if len(study.trials) == 0:
        study.enqueue_trial(baseline_params)

    def objective(trial: optuna.Trial) -> float:
        sp = cfg.tuning.search_space
        num_filters = trial.suggest_categorical("num_filters", list(sp.num_filters))
        dropout_rate = trial.suggest_float("dropout_rate", float(sp.dropout_rate[0]), float(sp.dropout_rate[1]))
        fc_units = trial.suggest_categorical("fc_units", list(sp.fc_units))
        n_conv_blocks = trial.suggest_categorical("n_conv_blocks", list(sp.n_conv_blocks))
        lr = trial.suggest_float("lr", float(sp.lr[0]), float(sp.lr[1]), log=True)
        batch_size = trial.suggest_categorical("batch_size", list(sp.batch_size))
        optimizer_name = trial.suggest_categorical("optimizer", list(sp.optimizer))
        weight_decay = trial.suggest_float(
            "weight_decay", float(sp.weight_decay[0]), float(sp.weight_decay[1]), log=True
        )

        L.seed_everything(cfg.seed, workers=True)

        model = ConfigurableCNN(
            in_channels=datamodule.in_channels,
            num_classes=datamodule.num_classes,
            num_filters=num_filters,
            dropout_rate=dropout_rate,
            fc_units=fc_units,
            n_conv_blocks=n_conv_blocks,
        )
        lit_module = LitClassifier(
            model=model,
            num_classes=datamodule.num_classes,
            lr=lr,
            optimizer=optimizer_name,
            weight_decay=weight_decay,
        )

        train_dl = DataLoader(
            gs_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_dl = datamodule.val_dataloader()

        trainer = build_trainer(
            cfg.trainer.get("accelerator", "auto"),
            max_epochs=cfg.tuning.epochs_gs,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )

        best_val_acc = 0.0

        class PyTorchLightningPruningCallback(L.Callback):
            def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
                epoch = trainer.current_epoch
                val_acc = float(trainer.callback_metrics.get("val_acc", 0.0))
                nonlocal best_val_acc
                best_val_acc = max(best_val_acc, val_acc)
                trial.report(val_acc, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        trainer.callbacks.append(PyTorchLightningPruningCallback())
        trainer.fit(lit_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

        return best_val_acc

    remaining = max(0, cfg.tuning.n_trials - len(study.trials))
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)

    return study
