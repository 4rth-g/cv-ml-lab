"""Módulo de comparação estatística pareada e seleção na validação para o cvlab."""

from __future__ import annotations

from typing import Any

import lightning as L
import numpy as np
import torch
from scipy import stats

from cvlab.data.base import BaseImageClfDataModule
from cvlab.models.cnn import ConfigurableCNN
from cvlab.models.lit_module import LitClassifier
from cvlab.tracking import mcnemar_test
from cvlab.xpu import build_trainer


def _train_single_run(
    config: dict[str, Any],
    seed: int,
    datamodule: BaseImageClfDataModule,
    epochs: int,
    accelerator: str = "auto",
    pl_logger: Any = None,
) -> tuple[float, float, L.LightningModule, list[int], list[int]]:
    """Treina uma rodada no split completo com restauração da melhor época (val).

    Retorna: (test_acc, best_val_acc, best_model, preds, true_labels).
    """
    L.seed_everything(seed, workers=True)
    datamodule.setup("fit")

    model = ConfigurableCNN(
        in_channels=datamodule.in_channels,
        num_classes=datamodule.num_classes,
        num_filters=config["num_filters"],
        dropout_rate=config["dropout_rate"],
        fc_units=config["fc_units"],
        n_conv_blocks=config.get("n_conv_blocks", 2),
    )
    lit_module = LitClassifier(
        model=model,
        num_classes=datamodule.num_classes,
        lr=config["lr"],
        optimizer=config["optimizer"],
        weight_decay=config["weight_decay"],
    )

    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()
    test_dl = datamodule.test_dataloader()

    best_val_acc = 0.0
    best_state: dict[str, torch.Tensor] | None = None

    class BestValRestoreCallback(L.Callback):
        def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
            nonlocal best_val_acc, best_state
            val_acc = float(trainer.callback_metrics.get("val_acc", 0.0))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()}

    trainer = build_trainer(
        accelerator,
        max_epochs=epochs,
        enable_checkpointing=False,
        logger=pl_logger if pl_logger is not None else False,
        enable_progress_bar=False,
        callbacks=[BestValRestoreCallback()],
    )

    trainer.fit(lit_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    if best_state is not None:
        lit_module.load_state_dict(best_state)

    # Avaliação no teste com coleta de predições e rótulos
    lit_module.eval()
    all_preds: list[int] = []
    all_targets: list[int] = []

    device = lit_module.device
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            logits = lit_module(x)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(y.tolist())

    test_acc = float(np.mean(np.array(all_preds) == np.array(all_targets)))
    return test_acc, best_val_acc, lit_module, all_preds, all_targets


def train_final(
    config: dict[str, Any],
    datamodule: BaseImageClfDataModule,
    cfg: Any,
    pl_logger: Any = None,
) -> tuple[L.LightningModule, float, float]:
    """Treina 1x a config final (logger opcional p/ curvas por época no W&B).

    Usado após a seleção para gerar o checkpoint final e as curvas de treino
    (train/val por época) do modelo escolhido. Retorna (model, val_acc, test_acc).
    """
    accelerator = cfg.trainer.get("accelerator", "auto")
    test_acc, val_acc, model, _, _ = _train_single_run(
        config, cfg.seed, datamodule, cfg.tuning.final_epochs, accelerator=accelerator, pl_logger=pl_logger
    )
    return model, val_acc, test_acc


def rigorous_compare(
    baseline_cfg: dict[str, Any],
    best_cfg: dict[str, Any],
    datamodule: BaseImageClfDataModule,
    cfg: Any,
) -> dict[str, Any]:
    """Realiza a comparação estatística rigorosa pareada entre baseline e melhor config.

    Passos do MÉTODO RIGOROSO:
    1. Retreino multi-seed (N_SEEDS) de baseline e melhor config no split completo.
    2. Restauração dos pesos da melhor época segundo val_acc.
    3. Estatística PAREADA: Paired t-test, Cohen's d_z, IC95% e teste de McNemar.
    4. Seleção garantida NA VALIDAÇÃO (best_val_mean >= baseline_val_mean).
    """
    n_seeds = cfg.tuning.n_seeds
    final_epochs = cfg.tuning.final_epochs
    base_seed = cfg.seed
    seeds = [base_seed + i for i in range(n_seeds)]
    accelerator = cfg.trainer.get("accelerator", "auto")

    baseline_runs: list[dict[str, Any]] = []
    baseline_models: list[tuple[float, float, L.LightningModule, list[int], list[int]]] = []
    for s in seeds:
        test_acc, val_acc, model, preds, targets = _train_single_run(
            baseline_cfg, s, datamodule, final_epochs, accelerator=accelerator
        )
        baseline_runs.append({"seed": s, "val_acc": val_acc, "test_acc": test_acc})
        baseline_models.append((test_acc, val_acc, model, preds, targets))

    best_runs: list[dict[str, Any]] = []
    best_models: list[tuple[float, float, L.LightningModule, list[int], list[int]]] = []
    for s in seeds:
        test_acc, val_acc, model, preds, targets = _train_single_run(
            best_cfg, s, datamodule, final_epochs, accelerator=accelerator
        )
        best_runs.append({"seed": s, "val_acc": val_acc, "test_acc": test_acc})
        best_models.append((test_acc, val_acc, model, preds, targets))

    baseline_val_arr = np.array([r["val_acc"] for r in baseline_runs])
    best_val_arr = np.array([r["val_acc"] for r in best_runs])
    baseline_test_arr = np.array([r["test_acc"] for r in baseline_runs])
    best_test_arr = np.array([r["test_acc"] for r in best_runs])

    baseline_val_mean = float(baseline_val_arr.mean())
    best_val_mean = float(best_val_arr.mean())
    baseline_test_mean = float(baseline_test_arr.mean())
    best_test_mean = float(best_test_arr.mean())

    baseline_val_std = float(baseline_val_arr.std(ddof=1)) if n_seeds > 1 else 0.0
    best_val_std = float(best_val_arr.std(ddof=1)) if n_seeds > 1 else 0.0
    baseline_test_std = float(baseline_test_arr.std(ddof=1)) if n_seeds > 1 else 0.0
    best_test_std = float(best_test_arr.std(ddof=1)) if n_seeds > 1 else 0.0

    # Paired t-test
    diff_acc = best_test_arr - baseline_test_arr
    t_stat, p_value = stats.ttest_rel(best_test_arr, baseline_test_arr)
    diff_mean = float(diff_acc.mean())
    diff_std = float(diff_acc.std(ddof=1)) if n_seeds > 1 else 0.0
    cohens_d = diff_mean / diff_std if diff_std > 0 else 0.0

    # IC95% para a diferença pareada
    if n_seeds > 1 and diff_std > 0:
        se = diff_std / np.sqrt(n_seeds)
        t_crit = stats.t.ppf(0.975, df=n_seeds - 1)
        ci95 = (float(diff_mean - t_crit * se), float(diff_mean + t_crit * se))
    else:
        ci95 = (diff_mean, diff_mean)

    # Teste de McNemar na melhor seed de validação
    best_baseline_idx = int(np.argmax(baseline_val_arr))
    best_tuned_idx = int(np.argmax(best_val_arr))

    preds_baseline_best_val = baseline_models[best_baseline_idx][3]
    preds_tuned_best_val = best_models[best_tuned_idx][3]
    y_test_labels = best_models[best_tuned_idx][4]

    mcnemar_res = mcnemar_test(y_test_labels, preds_baseline_best_val, preds_tuned_best_val)

    # Seleção estrita NA VALIDAÇÃO
    tuned_ge_baseline = bool(best_val_mean >= baseline_val_mean)
    if tuned_ge_baseline:
        selected_name = "melhor config (Optuna)"
        selected_config = best_cfg
        selected_model = best_models[best_tuned_idx][2]
    else:
        selected_name = "baseline (fallback)"
        selected_config = baseline_cfg
        selected_model = baseline_models[best_baseline_idx][2]

    return {
        "n_seeds": n_seeds,
        "baseline_config": baseline_cfg,
        "best_config": best_cfg,
        "baseline_val_mean": baseline_val_mean,
        "baseline_val_std": baseline_val_std,
        "best_val_mean": best_val_mean,
        "best_val_std": best_val_std,
        "baseline_test_mean": baseline_test_mean,
        "baseline_test_std": baseline_test_std,
        "best_test_mean": best_test_mean,
        "best_test_std": best_test_std,
        "baseline_test_accs": baseline_test_arr.tolist(),
        "best_test_accs": best_test_arr.tolist(),
        "t_stat": float(t_stat) if not np.isnan(t_stat) else 0.0,
        "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
        "cohens_d": float(cohens_d),
        "ci95_diff": ci95,
        "mcnemar": mcnemar_res,
        "tuned_ge_baseline": tuned_ge_baseline,
        "selected_name": selected_name,
        "selected_config": selected_config,
        "selected_model": selected_model,
    }
