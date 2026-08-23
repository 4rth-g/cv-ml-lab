"""Retreino multi-seed pareado e seleção da configuração final.

Segunda etapa do método (a primeira é `cvlab.tuning.search`). Enquanto a
busca escolhe uma configuração candidata olhando UMA seed num subconjunto, aqui a
decisão é tomada com rigor:

- **Retreino multi-seed** de ambas as configurações no split completo, para medir
  variância em vez de confiar num resultado de sorte.
- **Pareamento por seed** (as mesmas sementes nos dois lados), que cancela o
  ruído comum e é o que torna válido o teste pareado feito depois.
- **Seleção na validação**, com *fallback* para a baseline se a config tunada não
  a superar. O conjunto de teste nunca entra na decisão.

**A inferência estatística não acontece aqui.** Teste t, Cohen's d_z, IC95%,
McNemar, correção para múltiplas comparações e bootstrap vivem em ``analysis/``
(R), alimentados pelos artefatos de `cvlab.export`. A separação não é
organizacional: como nenhum p-valor é calculado durante o treino, nenhum p-valor
pode influenciar qual modelo é entregue.

Ver `rigorous_compare`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightning as L
import numpy as np
import torch

from cvlab.data.base import BaseImageClfDataModule
from cvlab.metrics import accuracy, objective_metric_key, objective_score
from cvlab.models.factory import build_model, lit_params
from cvlab.models.lit_module import LitClassifier
from cvlab.xpu import build_trainer


@dataclass(frozen=True)
class SingleRunResult:
    """Resultado de um treino, com a métrica de RELATÓRIO separada da de SELEÇÃO.

    A separação é o conserto de um bug: ``val_acc`` vinha do torchmetrics (macro,
    por causa do default de ``average``) e ``test_acc`` era calculado à mão
    (micro), e ambos iam para a mesma coluna ``acc`` do export. Agora ``val_acc``
    e ``test_acc`` são a MESMA função (:func:`cvlab.metrics.accuracy`) nos dois
    splits, e o que decide é ``val_score`` — outro campo, outro nome, outra
    métrica possível. Ver `cvlab.metrics`.

    ``val_score`` nunca é exportado como acurácia; ele aparece no manifest junto
    do nome do objetivo, para que o relatório possa dizer em que critério a
    seleção foi feita.
    """

    val_acc: float
    test_acc: float
    val_score: float
    model: L.LightningModule
    per_split: dict[str, dict[str, list]]
    epoch_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class RunSeeds:
    """As três fontes de variância de um treino, mantidas separadas.

    Um multi-seed que varia só ``init`` mede a variância da inicialização dos
    pesos e nada mais: split e ordem de dados continuam fixos, e o intervalo de
    confiança resultante sai estreito demais (Bouthillier et al., MLSys 2021).

    Registrar as três em colunas próprias é o que permite ao relatório dizer
    *qual* variância foi medida, em vez de apresentar todas como se fossem uma.
    O default do cvlab varia apenas ``init``, preservando o protocolo histórico;
    ver ``configs/train.yaml`` (bloco ``variance``).
    """

    init: int
    data: int
    split: int


def plan_run_seeds(cfg: Any, n_seeds: int) -> list[RunSeeds]:
    """Monta as ``n_seeds`` triplas de sementes do multi-seed a partir de ``cfg.variance``.

    Cada fonte ligada recebe ``base_seed + i``; cada fonte desligada fica fixa em
    ``base_seed``. O default (só ``init`` ligada) reproduz exatamente o protocolo
    histórico do repo — importante para não invalidar em silêncio os resultados
    já reportados.

    Ligar ``vary_data_order`` e ``vary_split`` alarga o intervalo de confiança,
    porque passa a medir variância que antes ficava escondida. Isso é uma
    correção, não uma piora do resultado.
    """
    base = int(cfg.seed)
    variance = cfg.get("variance", {}) if hasattr(cfg, "get") else {}
    vary_init = bool(variance.get("vary_init", True))
    vary_data = bool(variance.get("vary_data_order", False))
    vary_split = bool(variance.get("vary_split", False))

    return [
        RunSeeds(
            init=base + i if vary_init else base,
            data=base + i if vary_data else base,
            split=base + i if vary_split else base,
        )
        for i in range(n_seeds)
    ]


def objective_of(cfg: Any) -> str:
    """Lê a métrica de seleção de ``cfg.tuning.objective``, com default ``accuracy``.

    O default preserva o protocolo histórico. Datasets desbalanceados
    (DermaMNIST, razão 58,7) devem rodar com ``balanced_accuracy``: é uma
    escolha de critério, e ela vai ao manifest para que o relatório possa
    declará-la.
    """
    tuning = cfg.get("tuning", {}) if hasattr(cfg, "get") else {}
    return str(tuning.get("objective", "accuracy") or "accuracy")


def _seed_rows(baseline_runs: list[dict[str, Any]], best_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Achata os runs em formato longo: uma linha por (arm, seed, split).

    Formato longo, e não largo, porque é o que o R consome direto
    (``dplyr::group_by``) e o que sobrevive a acrescentar splits ou arms depois
    sem renomear coluna nenhuma.
    """
    rows: list[dict[str, Any]] = []
    for arm, runs in (("baseline", baseline_runs), ("tuned", best_runs)):
        for r in runs:
            rs: RunSeeds = r["seeds"]
            for split, key in (("val", "val_acc"), ("test", "test_acc")):
                rows.append(
                    {
                        "arm": arm,
                        "seed_init": rs.init,
                        "seed_data": rs.data,
                        "seed_split": rs.split,
                        "split": split,
                        "acc": float(r[key]),
                    }
                )
    return rows


def _epoch_rows(baseline_runs: list[dict[str, Any]], best_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Métricas por época em formato longo: uma linha por (arm, seed, época)."""
    rows: list[dict[str, Any]] = []
    for arm, runs in (("baseline", baseline_runs), ("tuned", best_runs)):
        for r in runs:
            rs: RunSeeds = r["seeds"]
            for e in r.get("epochs", []):
                rows.append({"arm": arm, "seed_init": rs.init, **e})
    return rows


def _prediction_rows(baseline_runs: list[dict[str, Any]], best_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Predições por exemplo de cada (arm, seed, split), prontas para o export."""
    out: list[dict[str, Any]] = []
    for arm, runs in (("baseline", baseline_runs), ("tuned", best_runs)):
        for r in runs:
            rs: RunSeeds = r["seeds"]
            for split, payload in r["per_split"].items():
                out.append(
                    {
                        "arm": arm,
                        "seed_init": rs.init,
                        "split": split,
                        "preds": payload["preds"],
                        "targets": payload["targets"],
                        "probs": payload["probs"],
                    }
                )
    return out


def _train_single_run(
    config: dict[str, Any],
    seeds: RunSeeds,
    datamodule: BaseImageClfDataModule,
    epochs: int,
    accelerator: str = "auto",
    pl_logger: Any = None,
    objective: str = "accuracy",
) -> SingleRunResult:
    """Treina uma vez no split completo e avalia no teste, com early stopping por validação.

    Restaura os pesos da melhor época segundo ``objective`` (via
    ``BestValRestoreCallback``) antes de avaliar — assim o resultado reportado é
    o do melhor ponto segundo a validação, não o da última época (que pode já
    estar em overfitting).

    As acurácias devolvidas são recomputadas das PREDIÇÕES do modelo restaurado,
    não lidas de ``callback_metrics``. É o que garante que ``val`` e ``test``
    sejam a mesma função (:func:`cvlab.metrics.accuracy`) e que a coluna ``acc``
    do export bata com o que ``analysis/R/metrics.R`` recomputa das mesmas
    predições — antes, val vinha do torchmetrics em macro e test era micro.

    Args:
        config: Configuração achatada com ``_target_`` + hiperparâmetros de
            arquitetura e de treino (ver `cvlab.models.factory`).
        seeds: As três sementes desta rodada (ver :class:`RunSeeds`). No
            multi-seed, é o que difere entre execuções.
        datamodule: DataModule do dataset (train/val/test).
        epochs: Número de épocas de treino (``final_epochs``).
        accelerator: ``"auto"``, ``"xpu"``, ``"cpu"``... repassado a ``build_trainer``.
        pl_logger: Logger opcional do Lightning (ex.: W&B) para registrar curvas por
            época. ``None`` desliga o logging (usado no multi-seed, que só coleta
            métricas finais).
        objective: Métrica de seleção (``accuracy`` ou ``balanced_accuracy``).
            Decide qual época restaurar e alimenta ``val_score``; não afeta
            ``val_acc``/``test_acc``.

    Returns:
        SingleRunResult: acurácias de relatório, score de seleção, o modelo já com
        os pesos da melhor época, as predições por exemplo de cada split e uma
        linha por época com loss, acurácia e learning rate.
    """
    L.seed_everything(seeds.init, workers=True)
    datamodule.configure_run_seeds(split_seed=seeds.split, data_seed=seeds.data)
    datamodule.setup("fit")

    model = build_model(
        config,
        in_channels=datamodule.in_channels,
        num_classes=datamodule.num_classes,
    )
    lit_module = LitClassifier(
        model=model,
        num_classes=datamodule.num_classes,
        multilabel=bool(getattr(datamodule, "is_multilabel", False)),
        **lit_params(config),
    )

    train_dl = datamodule.train_dataloader()
    val_dl = datamodule.val_dataloader()
    test_dl = datamodule.test_dataloader()

    # A época é escolhida pela MESMA métrica que escolhe a config
    # (`tuning.objective`). Fixar isto em `val_acc` faria o early stopping
    # otimizar micro enquanto a seleção final otimizava macro — duas decisões
    # em critérios diferentes dentro do mesmo run.
    best_epoch_key = objective_metric_key(objective, "val")
    best_epoch_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    class BestValRestoreCallback(L.Callback):
        """Guarda uma cópia dos pesos na melhor época segundo o objetivo (early stopping).

        Não interrompe o treino; apenas mantém o melhor snapshot em CPU. Ao final,
        `_train_single_run` recarrega esse estado, então a avaliação no teste
        usa o melhor ponto segundo a validação — sem espiar o teste.
        """

        def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
            nonlocal best_epoch_score, best_state
            score = float(trainer.callback_metrics.get(best_epoch_key, 0.0))
            if score > best_epoch_score:
                best_epoch_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()}

    # Métricas por época. O treino só as mandava para o W&B, então elas não
    # existiam no contrato de artefatos e a curva de aprendizado — o diagnóstico
    # mais direto de overfitting, underfitting e LR errado — era impossível de
    # reproduzir fora do W&B.
    epoch_rows: list[dict[str, Any]] = []

    class EpochMetricsCallback(L.Callback):
        """Coleta loss, acurácia e LR ao fim de cada época de TREINO.

        O hook é `on_train_epoch_end`, e não `on_validation_end`, porque o loop de
        validação roda ANTES da agregação das métricas de treino: colher ali
        deixava `train_loss` vazio na primeira época e defasado de uma época em
        todas as seguintes. Aqui as duas metades já estão em `callback_metrics`.
        """

        def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
            if trainer.sanity_checking:
                return
            m = trainer.callback_metrics

            def get(key: str) -> float | None:
                v = m.get(key)
                return float(v) if v is not None else None

            lrs = [g["lr"] for g in trainer.optimizers[0].param_groups] if trainer.optimizers else []
            epoch_rows.append(
                {
                    "epoch": int(trainer.current_epoch),
                    "train_loss": get("train_loss"),
                    "train_acc": get("train_acc"),
                    "val_loss": get("val_loss"),
                    "val_acc": get("val_acc"),
                    "lr": float(lrs[0]) if lrs else None,
                }
            )

    trainer = build_trainer(
        accelerator,
        max_epochs=epochs,
        enable_checkpointing=False,
        logger=pl_logger if pl_logger is not None else False,
        enable_progress_bar=False,
        callbacks=[BestValRestoreCallback(), EpochMetricsCallback()],
    )

    trainer.fit(lit_module, train_dataloaders=train_dl, val_dataloaders=val_dl)

    if best_state is not None:
        lit_module.load_state_dict(best_state)

    # Predições por exemplo, não só a acurácia. Dois motivos: o McNemar compara
    # acerto/erro exemplo a exemplo, e o export delas é o que torna qualquer
    # outra métrica (F1, AUC, kappa, matriz de confusão) calculável DEPOIS, sem
    # retreinar. Val entra junto do teste para que a escolha de métrica não
    # precise olhar o teste.
    lit_module.eval()
    device = lit_module.device
    per_split: dict[str, dict[str, list]] = {}

    for split_name, loader in (("val", val_dl), ("test", test_dl)):
        preds_out: list[int] = []
        targets_out: list[int] = []
        probs_out: list[list[float]] = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                logits = lit_module(x)
                # Probabilidade, e não só o argmax. É o que habilita AUC, ROC,
                # curva precisão-recall e calibração — nenhuma delas é
                # recuperável a partir da classe predita. Importa em particular
                # porque o benchmark oficial do MedMNIST reporta AUC junto da
                # acurácia, então sem isto não há como comparar com o publicado.
                probs = torch.softmax(logits, dim=1)
                preds_out.extend(torch.argmax(logits, dim=1).cpu().tolist())
                probs_out.extend([[round(float(v), 5) for v in row] for row in probs.cpu()])
                targets_out.extend(y.tolist())
        per_split[split_name] = {"preds": preds_out, "targets": targets_out, "probs": probs_out}

    # Ambas as acurácias saem da MESMA função, sobre as predições do modelo
    # restaurado. Ler a de validação de `callback_metrics` era o bug: aquele
    # valor era macro e este é micro, e os dois iam para a coluna `acc`.
    val_acc = accuracy(per_split["val"]["preds"], per_split["val"]["targets"])
    test_acc = accuracy(per_split["test"]["preds"], per_split["test"]["targets"])
    val_score = objective_score(per_split["val"]["preds"], per_split["val"]["targets"], objective)

    return SingleRunResult(
        val_acc=val_acc,
        test_acc=test_acc,
        val_score=val_score,
        model=lit_module,
        per_split=per_split,
        epoch_rows=epoch_rows,
    )


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
    base_seeds = RunSeeds(init=cfg.seed, data=cfg.seed, split=cfg.seed)
    run = _train_single_run(
        config,
        base_seeds,
        datamodule,
        cfg.tuning.final_epochs,
        accelerator=accelerator,
        pl_logger=pl_logger,
        objective=objective_of(cfg),
    )
    return run.model, run.val_acc, run.test_acc


def rigorous_compare(
    baseline_cfg: dict[str, Any],
    best_cfg: dict[str, Any],
    datamodule: BaseImageClfDataModule,
    cfg: Any,
) -> dict[str, Any]:
    """Retreina baseline e config tunada multi-seed e escolhe uma delas.

    Executa o núcleo do método:

    1. **Retreino multi-seed**: treina baseline e tunada com as MESMAS N seeds no
       split completo (ver :class:`RunSeeds`). Usar as mesmas seeds torna as
       amostras *pareadas* — cada seed vira um par baseline/tunada — e é isso que
       autoriza o teste pareado feito depois, em R.
    2. **Coleta de evidência**: acurácia por seed e predições por exemplo, de
       validação e teste, no formato que `cvlab.export` grava. Nenhuma
       estatística inferencial é computada aqui.
    3. **Seleção na validação**: a tunada só é escolhida se a média do
       ``val_score`` (métrica de ``cfg.tuning.objective``) empatar ou superar a
       da baseline; caso contrário, *fallback* para a baseline. O conjunto de
       teste nunca entra na decisão. A acurácia REPORTADA (``acc`` no export) é
       sempre micro e independe do objetivo — ver `cvlab.metrics`.

    Args:
        baseline_cfg: Configuração baseline (com ``_target_``).
        best_cfg: Configuração tunada vinda de ``study.best_params`` (com ``_target_``).
        datamodule: DataModule do dataset.
        cfg: Config Hydra; usa ``cfg.tuning.n_seeds``, ``cfg.tuning.final_epochs``,
            ``cfg.seed``, ``cfg.variance`` e ``cfg.trainer.accelerator``.

    Returns:
        dict: médias e desvios de val e teste por config, ``diff_test_mean``, as
        linhas tabulares para o export (``seed_rows``, ``predictions``), as seeds
        de melhor validação de cada lado (``mcnemar_baseline_seed``,
        ``mcnemar_tuned_seed``), o flag ``tuned_ge_baseline`` e a escolha final
        (``selected_arm``, ``selected_name``, ``selected_config``,
        ``selected_model``).
    """
    n_seeds = cfg.tuning.n_seeds
    final_epochs = cfg.tuning.final_epochs
    seeds = plan_run_seeds(cfg, n_seeds)
    accelerator = cfg.trainer.get("accelerator", "auto")
    objective = objective_of(cfg)

    # Um "arm" é um lado da comparação. As MESMAS seeds nos dois → amostras
    # pareadas (um par baseline/tunada por seed), que é o que cancela o ruído
    # comum e habilita os testes pareados.
    def run_arm(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Any]]:
        runs: list[dict[str, Any]] = []
        models: list[Any] = []
        for rs in seeds:
            run = _train_single_run(config, rs, datamodule, final_epochs, accelerator=accelerator, objective=objective)
            runs.append(
                {
                    "seeds": rs,
                    "val_acc": run.val_acc,
                    "test_acc": run.test_acc,
                    "val_score": run.val_score,
                    "per_split": run.per_split,
                    "epochs": run.epoch_rows,
                }
            )
            models.append(run.model)
        return runs, models

    baseline_runs, baseline_models = run_arm(baseline_cfg)
    best_runs, best_models = run_arm(best_cfg)

    baseline_val_arr = np.array([r["val_acc"] for r in baseline_runs])
    best_val_arr = np.array([r["val_acc"] for r in best_runs])
    baseline_test_arr = np.array([r["test_acc"] for r in baseline_runs])
    best_test_arr = np.array([r["test_acc"] for r in best_runs])

    # As acurácias acima são as REPORTADAS (micro, iguais às do export). Os
    # scores abaixo são os que DECIDEM. Com `objective=accuracy` os dois
    # coincidem; com `balanced_accuracy` não, e é por isso que são arrays
    # distintos em vez de um só reaproveitado.
    baseline_score_arr = np.array([r["val_score"] for r in baseline_runs])
    best_score_arr = np.array([r["val_score"] for r in best_runs])

    baseline_val_mean = float(baseline_val_arr.mean())
    best_val_mean = float(best_val_arr.mean())
    baseline_test_mean = float(baseline_test_arr.mean())
    best_test_mean = float(best_test_arr.mean())

    baseline_val_std = float(baseline_val_arr.std(ddof=1)) if n_seeds > 1 else 0.0
    best_val_std = float(best_val_arr.std(ddof=1)) if n_seeds > 1 else 0.0
    baseline_test_std = float(baseline_test_arr.std(ddof=1)) if n_seeds > 1 else 0.0
    best_test_std = float(best_test_arr.std(ddof=1)) if n_seeds > 1 else 0.0

    # Diferença média das acurácias de teste: descritivo, para o print e o W&B.
    # Teste t, Cohen's d, IC95% e McNemar NÃO são calculados aqui — são
    # inferência, e inferência é responsabilidade do `analysis/` (R), a partir
    # dos artefatos exportados. Ver `cvlab.export`.
    diff_mean = float((best_test_arr - baseline_test_arr).mean())

    # Índice da seed de melhor VALIDAÇÃO de cada lado (nunca a de melhor teste →
    # seria vazamento). Serve para dois fins: escolher o modelo entregue, e
    # registrar no manifest qual par o McNemar deve usar do lado do R. Usa o
    # score de objetivo, o mesmo critério da seleção — senão o modelo entregue
    # seria o melhor numa métrica e a config, a melhor em outra.
    best_baseline_idx = int(np.argmax(baseline_score_arr))
    best_tuned_idx = int(np.argmax(best_score_arr))

    # Seleção NA VALIDAÇÃO (nunca no teste): a tunada só vence se empatar ou superar
    # a baseline na validação média. Senão, fallback p/ baseline — o guard-rail que
    # impede reportar algo pior que a baseline mesmo quando a busca "achou" outra coisa.
    tuned_ge_baseline = bool(float(best_score_arr.mean()) >= float(baseline_score_arr.mean()))
    if tuned_ge_baseline:
        selected_name = "melhor config (Optuna)"
        selected_config = best_cfg
        selected_model = best_models[best_tuned_idx]
    else:
        selected_name = "baseline (fallback)"
        selected_config = baseline_cfg
        selected_model = baseline_models[best_baseline_idx]

    return {
        "n_seeds": n_seeds,
        # O relatório precisa dizer em que métrica a seleção foi feita: com
        # `balanced_accuracy` a config vencedora pode ter acurácia REPORTADA
        # menor que a perdedora, e sem esta linha isso pareceria um erro.
        "objective": objective,
        "baseline_val_score_mean": float(baseline_score_arr.mean()),
        "best_val_score_mean": float(best_score_arr.mean()),
        "seed_rows": _seed_rows(baseline_runs, best_runs),
        "epoch_rows": _epoch_rows(baseline_runs, best_runs),
        "predictions": _prediction_rows(baseline_runs, best_runs),
        "mcnemar_baseline_seed": baseline_runs[best_baseline_idx]["seeds"].init,
        "mcnemar_tuned_seed": best_runs[best_tuned_idx]["seeds"].init,
        "selected_arm": "tuned" if tuned_ge_baseline else "baseline",
        "diff_test_mean": diff_mean,
        "uses_official_split": bool(datamodule.uses_official_split),
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
        "tuned_ge_baseline": tuned_ge_baseline,
        "selected_name": selected_name,
        "selected_config": selected_config,
        "selected_model": selected_model,
    }
