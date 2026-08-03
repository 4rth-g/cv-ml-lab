"""Script principal de treino e tuning do cvlab."""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from cvlab.data.registry import get_datamodule
from cvlab.models.factory import baseline_params
from cvlab.tracking import init_wandb, log_experiment
from cvlab.tuning.rigorous import rigorous_compare, train_final
from cvlab.tuning.search import optuna_search
from cvlab.xpu import XPUAccelerator

# Caminho absoluto p/ configs/ (na raiz do repo) — robusto quando rodado via o
# entry-point instalado `cvlab-train` (config_path relativo é interpretado como
# módulo pelo Hydra nesse caso e falha).
_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="train")
def train(cfg: DictConfig) -> None:
    """Entry point principal do pipeline de treino e tuning no cvlab."""
    print("=== cv-ml-lab: Pipeline de Treino e Tuning Rigoroso ===")
    print(f"Dataset: {cfg.dataset._target_}")
    print(f"Modelo:  {cfg.model._target_}")

    L.seed_everything(cfg.seed, workers=True)

    # Auto-detecta Intel Arc (XPU) — o Lightning não seleciona XPU no 'auto'.
    if cfg.trainer.get("accelerator", "auto") == "auto" and XPUAccelerator.is_available():
        with open_dict(cfg):
            cfg.trainer.accelerator = "xpu"
        print("🔵 Intel XPU (Arc) detectada — accelerator='xpu'.")

    datamodule = get_datamodule(cfg)
    datamodule.prepare_data()  # baixa os dados se ausentes (setup usa download=False)
    output_dir = Path(cfg.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Busca Optuna
    print("\n--- 1. Otimização de Hiperparâmetros (Optuna) ---")
    study = optuna_search(datamodule, cfg)
    print(f"Melhor val_acc na busca: {study.best_value:.4f}")
    print("Melhores parâmetros encontrados:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Configs baseline e melhor, ambas carregando _target_ p/ a fábrica de modelos
    model_target = str(cfg.model.get("_target_"))
    baseline_cfg = {"_target_": model_target, **baseline_params(cfg)}
    best_cfg = {"_target_": model_target, **dict(study.best_params)}

    # 2. Comparação Estatística Rigorosa
    print("\n--- 2. Comparação Estatística Rigorosa Multi-Seed ---")
    res = rigorous_compare(baseline_cfg, best_cfg, datamodule, cfg)

    print(f"\nBaseline (Val Mean): {res['baseline_val_mean'] * 100:.2f}% ± {res['baseline_val_std'] * 100:.2f}%")
    print(f"Melhor Config (Val Mean): {res['best_val_mean'] * 100:.2f}% ± {res['best_val_std'] * 100:.2f}%")
    print(f"Baseline (Test Mean): {res['baseline_test_mean'] * 100:.2f}% ± {res['baseline_test_std'] * 100:.2f}%")
    print(f"Melhor Config (Test Mean): {res['best_test_mean'] * 100:.2f}% ± {res['best_test_std'] * 100:.2f}%")
    print(f"Paired t-test: t={res['t_stat']:.3f}, p={res['p_value']:.4f}")
    print(f"Cohen's d_z: {res['cohens_d']:.3f}")
    print(f"McNemar: p={res['mcnemar']['p_value']:.4e} ({res['mcnemar']['method']})")
    print(f"Config Selecionada (Validação): {res['selected_name']}")

    # 3. Experiment Tracking
    wandb_logger = init_wandb(cfg)
    final_model = res["selected_model"]
    if wandb_logger is not None:
        metrics_to_log = {
            "search_best_val_acc": study.best_value,
            "baseline_val_mean": res["baseline_val_mean"],
            "best_val_mean": res["best_val_mean"],
            "baseline_test_mean": res["baseline_test_mean"],
            "best_test_mean": res["best_test_mean"],
            "t_stat": res["t_stat"],
            "p_value": res["p_value"],
            "cohens_d": res["cohens_d"],
            "mcnemar_p_value": res["mcnemar"]["p_value"],
        }
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        wandb_logger.experiment.summary.update(metrics_to_log)
        # Retreino final da config selecionada COM o logger → curvas train/val por
        # época do modelo escolhido no W&B; é também o modelo salvo no checkpoint.
        final_model, _, _ = train_final(res["selected_config"], datamodule, cfg, pl_logger=wandb_logger)

    log_experiment(
        model_name=res["selected_name"],
        config=res["selected_config"],
        metrics={
            "baseline_val_mean": res["baseline_val_mean"],
            "best_val_mean": res["best_val_mean"],
            "baseline_test_mean": res["baseline_test_mean"],
            "best_test_mean": res["best_test_mean"],
            "t_stat": res["t_stat"],
            "p_value": res["p_value"],
            "cohens_d": res["cohens_d"],
            "mcnemar": res["mcnemar"],
        },
        seed=cfg.seed,
        db_path=output_dir / "experiment_tracker.db",
    )

    # 4. Salvar Checkpoint Final
    checkpoint_path = output_dir / "best.ckpt"
    selected_model: L.LightningModule = final_model
    torch.save(
        {
            "state_dict": selected_model.state_dict(),
            "config": res["selected_config"],
            "metrics": res,
        },
        checkpoint_path,
    )
    print(f"\nModelo final salvo em: {checkpoint_path}")


def main() -> None:
    train()


if __name__ == "__main__":
    main()
