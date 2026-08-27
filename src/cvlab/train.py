"""Script principal de treino e tuning do cvlab."""

from __future__ import annotations

from pathlib import Path

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from cvlab.data.registry import get_datamodule
from cvlab.export import export_run_artifacts
from cvlab.models.factory import baseline_params
from cvlab.tracking import init_wandb, log_experiment, run_name
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

    # Nome do run calculado UMA vez e fixado em cfg.logger.name: o W&B e o arquivo
    # de checkpoint precisam da mesma string (o timestamp mudaria entre chamadas).
    # Respeita um `logger.name=...` passado pelo usuário.
    if not cfg.logger.get("name", None):
        with open_dict(cfg):
            cfg.logger.name = run_name(cfg)
    print(f"Run:     {cfg.logger.name}")

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
    print(f"Δ teste (tunada - baseline): {res['diff_test_mean'] * 100:+.2f} pp")
    print(f"Config Selecionada (Validação): {res['selected_name']}")

    # 3. Export dos artefatos brutos (acurácia por seed, predições por exemplo,
    # histórico da busca). ANTES do W&B de propósito: uma falha no logger não
    # pode custar a evidência do run, que é o insumo da análise em R.
    run_dir = export_run_artifacts(
        run_id=cfg.logger.name,
        res=res,
        cfg=cfg,
        datamodule=datamodule,
        output_dir=output_dir,
        study=study,
    )
    # A sugestão precisa carregar o output_dir: um smoke grava em results/_smoke,
    # e `just report` sozinho procuraria em results/ e não acharia nada. Mandar o
    # comando errado aqui é o que transforma "rode o relatório" num beco sem saída.
    report_root = "" if output_dir.as_posix().rstrip("/") == "results" else f" latest {output_dir}"
    print(f"Artefatos: {run_dir}/")
    print(f"Análise estatística (R): just report{report_root}")

    # 4. Experiment Tracking
    wandb_logger = init_wandb(cfg)
    final_model = res["selected_model"]
    if wandb_logger is not None:
        # Só descritivo: médias, desvios e o delta. Os testes de hipótese saem
        # do relatório em R, sobre os artefatos, e não do logger do treino.
        metrics_to_log = {
            "search_best_val_acc": study.best_value,
            "n_seeds": res["n_seeds"],
            "baseline_val_mean": res["baseline_val_mean"],
            "baseline_val_std": res["baseline_val_std"],
            "best_val_mean": res["best_val_mean"],
            "best_val_std": res["best_val_std"],
            "baseline_test_mean": res["baseline_test_mean"],
            "baseline_test_std": res["baseline_test_std"],
            "best_test_mean": res["best_test_mean"],
            "best_test_std": res["best_test_std"],
            "diff_test_mean": res["diff_test_mean"],
            "tuned_ge_baseline": res["tuned_ge_baseline"],
            "selected_arm": res["selected_arm"],
            "run_id": cfg.logger.name,
        }
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        wandb_logger.experiment.summary.update(metrics_to_log)
        # Retreino final da config selecionada COM o logger → curvas train/val por
        # época do modelo escolhido no W&B; é também o modelo salvo no checkpoint.
        final_model, _, _ = train_final(res["selected_config"], datamodule, cfg, pl_logger=wandb_logger)

    log_experiment(
        model_name=res["selected_name"],
        config=res["selected_config"],
        # O tracker vira o índice histórico ("que config foi escolhida, com que
        # médias") mais um ponteiro para onde mora a evidência bruta.
        metrics={
            "n_seeds": res["n_seeds"],
            "baseline_val_mean": res["baseline_val_mean"],
            "baseline_val_std": res["baseline_val_std"],
            "best_val_mean": res["best_val_mean"],
            "best_val_std": res["best_val_std"],
            "baseline_test_mean": res["baseline_test_mean"],
            "baseline_test_std": res["baseline_test_std"],
            "best_test_mean": res["best_test_mean"],
            "best_test_std": res["best_test_std"],
            "diff_test_mean": res["diff_test_mean"],
            "selected_arm": res["selected_arm"],
            "run_id": cfg.logger.name,
            "artifacts_dir": str(run_dir),
        },
        seed=cfg.seed,
        db_path=output_dir / "experiment_tracker.db",
    )

    # 5. Salvar Checkpoint Final
    # Nomeado como o run do W&B (modelo-dataset-data_hora) para os dois serem
    # correlacionáveis; um nome fixo faria cada execução sobrescrever a anterior.
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f"{cfg.logger.name}.ckpt"
    selected_model: L.LightningModule = final_model
    # Só o resumo, nunca `res` inteiro: ele carrega `selected_model` (o módulo
    # seria pickled DUAS vezes no mesmo arquivo) e agora também as predições por
    # exemplo. A evidência bruta mora em results/runs/<run_id>/, não aqui.
    torch.save(
        {
            "state_dict": selected_model.state_dict(),
            "config": res["selected_config"],
            "metrics": {
                k: res[k]
                for k in (
                    "n_seeds",
                    "baseline_val_mean",
                    "baseline_val_std",
                    "best_val_mean",
                    "best_val_std",
                    "baseline_test_mean",
                    "baseline_test_std",
                    "best_test_mean",
                    "best_test_std",
                    "diff_test_mean",
                    "tuned_ge_baseline",
                    "selected_arm",
                    "selected_name",
                )
            },
            "run_id": cfg.logger.name,
            "artifacts_dir": str(run_dir),
        },
        checkpoint_path,
    )
    print(f"\nModelo final salvo em: {checkpoint_path}")


def main() -> None:
    train()


if __name__ == "__main__":
    main()
