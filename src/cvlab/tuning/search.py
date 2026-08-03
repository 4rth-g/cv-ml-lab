"""Busca de hiperparâmetros com Optuna — primeira etapa do método rigoroso.

A busca encontra uma configuração *candidata* a superar a baseline. O rigor da
comparação (retreino multi-seed, testes pareados, seleção na validação) fica em
`cvlab.tuning.rigorous`; aqui o foco é explorar o espaço de forma barata e
com a baseline garantida como piso. Ver `optuna_search`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, Subset

from cvlab.data.base import BaseImageClfDataModule
from cvlab.models.factory import baseline_params, build_model, lit_params, suggest_params
from cvlab.models.lit_module import LitClassifier
from cvlab.xpu import build_trainer

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optuna_search(datamodule: BaseImageClfDataModule, cfg: Any) -> optuna.Study:
    """Busca hiperparâmetros com Optuna, com a baseline como piso garantido.

    Três decisões de projeto importam mais que o sampler em si:

    - **Baseline enfileirada** (``study.enqueue_trial``): o primeiro trial é a
      própria configuração baseline. Assim, por construção, o melhor trial nunca
      fica abaixo da baseline na métrica de busca — a regra "o grid nunca deve
      ser inferior à baseline".
    - **Busca barata em subconjunto** (``gs_subset_size``) com pruning
      (``MedianPruner``): explora muitas configurações em parte do treino e
      interrompe cedo as ruins, economizando computação. Esta métrica de
      subconjunto NÃO decide a seleção final — ela é refeita em multi-seed por
      `cvlab.tuning.rigorous.rigorous_compare`.
    - **Estudo persistido e escopado por dataset+modelo**: o SQLite permite
      retomar a busca de onde parou; o escopo evita que estudos de datasets ou
      arquiteturas diferentes colidam no mesmo ``output_dir``.

    Args:
        datamodule: DataModule do dataset. ``setup('fit')`` é chamado aqui para
            materializar o split de treino de onde sai o subconjunto de busca.
        cfg: Config Hydra. Usa ``cfg.tuning`` (``n_trials``, ``epochs_gs``,
            ``gs_subset_size``, ``search_space``), ``cfg.model._target_`` (a
            arquitetura a instanciar), ``cfg.trainer.accelerator``, ``cfg.seed`` e
            ``cfg.output_dir``.

    Returns:
        optuna.Study: o estudo concluído. ``study.best_params`` traz a melhor
        configuração encontrada, a ser confirmada em multi-seed depois.
    """
    datamodule.setup("fit")

    # Subconjunto do treino p/ busca: reduz o custo por trial. A permutação é
    # semeada por cfg.seed → o mesmo subconjunto em toda execução (reprodutível).
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
    # Escopa o estudo pelo dataset E pelo modelo p/ não colidir no mesmo output_dir
    ds = str(cfg.dataset.get("_target_", "run")).split(".")[-1].replace("DataModule", "").lower() or "run"
    model_name = str(cfg.model.get("_target_", "model")).split(".")[-1].lower() or "model"
    tag = f"{ds}_{model_name}"
    db_path = output_dir / f"optuna_study_{tag}.db"

    study = optuna.create_study(
        study_name=f"cvlab_optuna_{tag}",
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=cfg.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    # Enfileira a baseline como 1º trial (só num estudo novo) → piso garantido.
    # Se o estudo foi retomado (já tem trials), não reenfileira.
    if len(study.trials) == 0:
        study.enqueue_trial(baseline_params(cfg))

    model_target = str(cfg.model.get("_target_"))

    def objective(trial: optuna.Trial) -> float:
        """Treina uma configuração e devolve a melhor val_acc atingida (a maximizar)."""
        params = suggest_params(trial, cfg.tuning.search_space)

        # Mesma seed em todo trial: fixa init dos pesos e ordem dos dados, então a
        # diferença de val_acc entre trials reflete os HIPERPARÂMETROS, não o
        # ruído de seed. A variabilidade por seed é medida depois, no multi-seed.
        L.seed_everything(cfg.seed, workers=True)

        model = build_model(
            {"_target_": model_target, **params},
            in_channels=datamodule.in_channels,
            num_classes=datamodule.num_classes,
        )
        lit_module = LitClassifier(
            model=model,
            num_classes=datamodule.num_classes,
            **lit_params(params),
        )

        train_dl = DataLoader(
            gs_train_dataset,
            batch_size=int(params.get("batch_size", datamodule.batch_size)),
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
            """Reporta val_acc por época ao Optuna e poda o trial se for ruim.

            A cada validação, informa a val_acc ao trial; o ``MedianPruner``
            aborta (``TrialPruned``) trials que ficam abaixo da mediana histórica
            no mesmo passo — evita gastar épocas numa configuração já perdedora.
            """

            def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
                epoch = trainer.current_epoch
                val_acc = float(trainer.callback_metrics.get("val_acc", 0.0))
                nonlocal best_val_acc
                # Score do trial = melhor val_acc ao longo das épocas (não a última).
                best_val_acc = max(best_val_acc, val_acc)
                trial.report(val_acc, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        trainer.callbacks.append(PyTorchLightningPruningCallback())
        trainer.fit(lit_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

        return best_val_acc

    # remaining evita re-treinar trials já completos ao retomar um estudo salvo.

    remaining = max(0, cfg.tuning.n_trials - len(study.trials))
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)

    return study
