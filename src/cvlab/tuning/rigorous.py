"""Comparação estatística rigorosa entre a baseline e a configuração tunada.

Segunda etapa do método (a primeira é `cvlab.tuning.search`). Enquanto a
busca escolhe uma configuração candidata olhando UMA seed num subconjunto, aqui a
decisão é tomada com rigor:

- **Retreino multi-seed** de ambas as configurações no split completo, para medir
  variância em vez de confiar num resultado de sorte.
- **Estatística pareada** (mesma seed nos dois modelos) para cancelar o ruído
  seed-a-seed e ganhar poder: teste t pareado, Cohen's d_z, IC95% e McNemar.
- **Seleção na validação**, com *fallback* para a baseline se a config tunada não
  a superar — o teste é usado uma única vez, só para reportar.

Ver `rigorous_compare`.
"""

from __future__ import annotations

from typing import Any

import lightning as L
import numpy as np
import torch
from scipy import stats

from cvlab.data.base import BaseImageClfDataModule
from cvlab.models.factory import build_model, lit_params
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
    """Treina uma vez no split completo e avalia no teste, com early stopping por validação.

    Restaura os pesos da época de melhor ``val_acc`` (via ``BestValRestoreCallback``)
    antes de avaliar — assim o resultado reportado é o do melhor ponto segundo a
    validação, não o da última época (que pode já estar em overfitting).

    Args:
        config: Configuração achatada com ``_target_`` + hiperparâmetros de
            arquitetura e de treino (ver `cvlab.models.factory`).
        seed: Semente desta rodada. No multi-seed, é o que difere entre execuções.
        datamodule: DataModule do dataset (train/val/test).
        epochs: Número de épocas de treino (``final_epochs``).
        accelerator: ``"auto"``, ``"xpu"``, ``"cpu"``... repassado a ``build_trainer``.
        pl_logger: Logger opcional do Lightning (ex.: W&B) para registrar curvas por
            época. ``None`` desliga o logging (usado no multi-seed, que só coleta
            métricas finais).

    Returns:
        tuple: ``(test_acc, best_val_acc, model, preds, true_labels)`` — onde
        ``preds`` e ``true_labels`` são as predições e rótulos do conjunto de
        teste (usados no McNemar), e ``model`` já tem os pesos da melhor época.
    """
    L.seed_everything(seed, workers=True)
    datamodule.setup("fit")

    model = build_model(
        config,
        in_channels=datamodule.in_channels,
        num_classes=datamodule.num_classes,
    )
    lit_module = LitClassifier(
        model=model,
        num_classes=datamodule.num_classes,
        **lit_params(config),
    )

    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()
    test_dl = datamodule.test_dataloader()

    best_val_acc = 0.0
    best_state: dict[str, torch.Tensor] | None = None

    class BestValRestoreCallback(L.Callback):
        """Guarda uma cópia dos pesos na época de melhor val_acc (early stopping).

        Não interrompe o treino; apenas mantém o melhor snapshot em CPU. Ao final,
        `_train_single_run` recarrega esse estado, então a avaliação no teste
        usa o melhor ponto segundo a validação — sem espiar o teste.
        """

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

    # Avaliação no teste. Coleta predições por exemplo (não só a acurácia) porque
    # o McNemar precisa comparar acertos/erros exemplo a exemplo entre dois modelos.
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
    """Treina uma única vez a configuração já selecionada, com logger opcional.

    Chamado após `rigorous_compare` ter decidido a configuração. Serve para
    (a) gerar o checkpoint final e (b) registrar no W&B as curvas de treino/val por
    época do modelo escolhido — que o multi-seed não loga (roda sem logger).

    Args:
        config: Configuração selecionada (baseline ou tunada).
        datamodule: DataModule do dataset.
        cfg: Config Hydra; usa ``cfg.seed``, ``cfg.tuning.final_epochs`` e
            ``cfg.trainer.accelerator``.
        pl_logger: Logger do Lightning (ex.: WandbLogger) ou ``None``.

    Returns:
        tuple: ``(model, val_acc, test_acc)`` do treino final.
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
    """Compara baseline e config tunada com rigor estatístico e escolhe uma delas.

    Executa o núcleo do método:

    1. **Retreino multi-seed**: treina baseline e tunada com as MESMAS N seeds no
       split completo. Usar as mesmas seeds torna as amostras *pareadas* (cada seed
       vira um par baseline/tunada sob idêntica init e ordem de dados).
    2. **Testes pareados** sobre a acurácia de teste:
       - *Teste t pareado* (``ttest_rel``): compara par a par, cancelando o ruído
         comum a cada seed → mais poder que um t independente.
       - *Cohen's d_z*: tamanho do efeito (média das diferenças / desvio das
         diferenças); diz se a diferença é grande, além de significativa.
       - *IC95%* da diferença média: magnitude com incerteza.
       - *McNemar*: teste ao nível de exemplo (discordâncias no mesmo test set),
         complementar ao t (que é ao nível de seed).
    3. **Seleção na validação**: a tunada só é escolhida se
       ``best_val_mean >= baseline_val_mean``; caso contrário, *fallback* para a
       baseline. O teste nunca entra na decisão — só é reportado.

    Args:
        baseline_cfg: Configuração baseline (com ``_target_``).
        best_cfg: Configuração tunada vinda de ``study.best_params`` (com ``_target_``).
        datamodule: DataModule do dataset.
        cfg: Config Hydra; usa ``cfg.tuning.n_seeds``, ``cfg.tuning.final_epochs``,
            ``cfg.seed`` e ``cfg.trainer.accelerator``.

    Returns:
        dict: métricas agregadas (médias/desvios de val e teste por config),
        estatísticas (``t_stat``, ``p_value``, ``cohens_d``, ``ci95_diff``,
        ``mcnemar``), o flag ``tuned_ge_baseline`` e a escolha final
        (``selected_name``, ``selected_config``, ``selected_model``).
    """
    n_seeds = cfg.tuning.n_seeds
    final_epochs = cfg.tuning.final_epochs
    base_seed = cfg.seed
    # As MESMAS seeds nos dois grupos → amostras pareadas (um par baseline/tunada
    # por seed). É isso que habilita os testes pareados mais adiante.
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

    # Teste t PAREADO sobre a acurácia de teste: opera nas diferenças por seed
    # (best - baseline), então o ruído comum a cada seed se cancela → mais poder.
    diff_acc = best_test_arr - baseline_test_arr
    t_stat, p_value = stats.ttest_rel(best_test_arr, baseline_test_arr)
    diff_mean = float(diff_acc.mean())
    diff_std = float(diff_acc.std(ddof=1)) if n_seeds > 1 else 0.0
    # Cohen's d_z: tamanho do efeito = média das diferenças / desvio das diferenças.
    cohens_d = diff_mean / diff_std if diff_std > 0 else 0.0

    # IC95% para a diferença pareada
    if n_seeds > 1 and diff_std > 0:
        se = diff_std / np.sqrt(n_seeds)
        t_crit = stats.t.ppf(0.975, df=n_seeds - 1)
        ci95 = (float(diff_mean - t_crit * se), float(diff_mean + t_crit * se))
    else:
        ci95 = (diff_mean, diff_mean)

    # McNemar precisa de UM modelo por lado. Escolhe-se, de cada lado, a seed de
    # melhor VALIDAÇÃO (nunca a de melhor teste → sem vazamento). Compara acerto/erro
    # exemplo a exemplo; mcnemar_test usa binomial exata (<25 discordâncias) ou
    # chi2 com correção de continuidade.
    best_baseline_idx = int(np.argmax(baseline_val_arr))
    best_tuned_idx = int(np.argmax(best_val_arr))

    preds_baseline_best_val = baseline_models[best_baseline_idx][3]
    preds_tuned_best_val = best_models[best_tuned_idx][3]
    y_test_labels = best_models[best_tuned_idx][4]

    mcnemar_res = mcnemar_test(y_test_labels, preds_baseline_best_val, preds_tuned_best_val)

    # Seleção NA VALIDAÇÃO (nunca no teste): a tunada só vence se empatar ou superar
    # a baseline na validação média. Senão, fallback p/ baseline — o guard-rail que
    # impede reportar algo pior que a baseline mesmo quando a busca "achou" outra coisa.
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
