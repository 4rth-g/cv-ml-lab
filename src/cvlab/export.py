"""Export dos artefatos de um run: a fronteira entre o Python e a análise em R.

O cvlab é um pipeline de duas linguagens. O Python treina, decide e **reduz tudo
a tabela**; o R lê essas tabelas e faz a inferência estatística e o relatório
(ver ``analysis/``). O contrato entre os dois é este módulo, e nada mais: o lado
R nunca importa código Python nem toca em uma imagem.

Layout por run, sob ``output_dir``::

    results/runs/<run_id>/manifest.json        metadados e configs
    results/runs/<run_id>/seed_metrics.csv     uma linha por (arm, seed, split)
    results/runs/<run_id>/predictions.csv.gz   uma linha por (arm, seed, split, exemplo)
    results/runs/<run_id>/trials.csv           histórico da busca Optuna
    results/runs/index.csv                     uma linha por run (append-only)

Por que **predições por exemplo** e não só a acurácia: é o que torna qualquer
outra métrica (F1, AUC, kappa, matriz de confusão) calculável DEPOIS, sem
retreinar, e é o único insumo possível para McNemar e bootstrap por exemplo.
``example_idx`` casa com o EDA do dataset (`cvlab.eda`), o que permite cruzar
erro do modelo com propriedade da imagem.

Escrita atômica em toda parte: um run interrompido não pode deixar um CSV
truncado que o R leia como se estivesse completo.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from cvlab.tracking import config_group_choice, git_commit

#: Versão do contrato. Incrementar ao mudar coluna, e o leitor R recusa o que
#: não reconhece em vez de interpretar errado em silêncio.
SCHEMA_VERSION = 1

SEED_METRICS_COLUMNS = (
    "run_id",
    "model",
    "dataset",
    "subset",
    "budget",
    "arm",
    "seed_init",
    "seed_data",
    "seed_split",
    "split",
    "acc",
)

PREDICTION_COLUMNS = (
    "run_id",
    "arm",
    "seed_init",
    "split",
    "example_idx",
    "y_true",
    "y_pred",
)

INDEX_COLUMNS = (
    "run_id",
    "timestamp_utc",
    "commit",
    "model",
    "dataset",
    "subset",
    "budget",
    "n_seeds",
    "selected_arm",
    "path",
)

TRIAL_COLUMNS = ("trial_number", "state", "value", "duration_s", "params_json")


def _atomic_write(path: Path, write_body: Any, *, gzipped: bool = False) -> None:
    """Escreve via ``.tmp`` + ``os.replace`` (atômico no mesmo filesystem)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    opener = gzip.open if gzipped else open
    try:
        with opener(tmp, "wt", newline="", encoding="utf-8") as fh:  # type: ignore[operator]
            write_body(fh)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[dict[str, Any]], *, gzipped: bool = False) -> None:
    def body(fh: Any) -> None:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

    _atomic_write(path, body, gzipped=gzipped)


def _validate_predictions(predictions: Sequence[dict[str, Any]]) -> dict[str, list[int]]:
    """Confere as invariantes de que a análise em R depende, e devolve os rótulos por split.

    Duas checagens, ambas fatais:

    1. **Rótulo escalar.** O schema assume classificação de rótulo único. Um alvo
       multi-label (ex.: ChestMNIST) precisaria de outro schema; gravar assim
       produziria um CSV silenciosamente errado.
    2. **Mesmo ``y_true`` em todos os (arm, seed).** É o que garante que
       ``example_idx`` indexa o mesmo exemplo em toda linha do arquivo. Se
       divergir, o McNemar em R estaria comparando conjuntos diferentes e o
       join com o EDA apontaria para a imagem errada.
    """
    labels_by_split: dict[str, list[int]] = {}
    for entry in predictions:
        split = str(entry["split"])
        preds = entry["preds"]

        # A checagem de escalar vem ANTES do int(): com alvo multi-label a
        # conversão estouraria com um TypeError cru, sem dizer o que houve.
        for t in entry["targets"]:
            if isinstance(t, (list, tuple)) or hasattr(t, "__len__"):
                raise ValueError(
                    "y_true não é escalar: o export assume classificação de rótulo único. "
                    "Alvos multi-label exigem um schema próprio (SCHEMA_VERSION)."
                )

        targets = [int(t) for t in entry["targets"]]
        if len(preds) != len(targets):
            raise ValueError(f"predições e rótulos com tamanhos diferentes em {entry['arm']}/{split}.")

        known = labels_by_split.setdefault(split, targets)
        if known != targets:
            raise ValueError(
                f"y_true diverge entre execuções no split '{split}'. O example_idx deixaria de "
                "identificar o mesmo exemplo, invalidando McNemar e o join com o EDA."
            )
    return labels_by_split


def export_run_artifacts(
    run_id: str,
    res: dict[str, Any],
    cfg: Any,
    datamodule: Any,
    output_dir: Path | str,
    study: Any = None,
) -> Path:
    """Grava os artefatos do run e devolve o diretório criado.

    Args:
        run_id: Nome canônico do run (o mesmo do W&B e do checkpoint).
        res: Retorno de :func:`cvlab.tuning.rigorous.rigorous_compare`.
        cfg: Config Hydra resolvida.
        datamodule: DataModule já com ``setup`` feito (para tamanhos e classes).
        output_dir: Raiz dos resultados (``cfg.output_dir``).
        study: Study do Optuna, opcional; sem ele o ``trials.csv`` não é gerado.
    """
    root = Path(output_dir) / "runs"
    run_dir = root / run_id

    model = config_group_choice("model", cfg.model.get("_target_", "model"))
    dataset = config_group_choice("dataset", cfg.dataset.get("_target_", "dataset"))
    subset = str(cfg.dataset.get("subset", "") or "")
    budget = config_group_choice("tuning/budget", "default")

    labels_by_split = _validate_predictions(res["predictions"])

    # 1. seed_metrics.csv
    seed_rows = [
        {
            "run_id": run_id,
            "model": model,
            "dataset": dataset,
            "subset": subset,
            "budget": budget,
            **row,
            # Split oficial não tem seed: gravar um número aqui sugeriria uma
            # fonte de variância que não existe neste dataset.
            "seed_split": "official" if res.get("uses_official_split") else row["seed_split"],
        }
        for row in res["seed_rows"]
    ]
    _write_csv(run_dir / "seed_metrics.csv", SEED_METRICS_COLUMNS, seed_rows)

    # 2. predictions.csv.gz
    def prediction_rows() -> Iterable[dict[str, Any]]:
        for entry in res["predictions"]:
            for idx, (y_true, y_pred) in enumerate(zip(entry["targets"], entry["preds"])):
                yield {
                    "run_id": run_id,
                    "arm": entry["arm"],
                    "seed_init": entry["seed_init"],
                    "split": entry["split"],
                    "example_idx": idx,
                    "y_true": int(y_true),
                    "y_pred": int(y_pred),
                }

    _write_csv(run_dir / "predictions.csv.gz", PREDICTION_COLUMNS, prediction_rows(), gzipped=True)

    # 3. trials.csv (histórico da busca; insumo da curva de performance esperada
    #    por orçamento — reportar o máximo de uma busca é estimador enviesado)
    if study is not None:
        _write_csv(run_dir / "trials.csv", TRIAL_COLUMNS, _trial_rows(study))

    # 4. manifest.json
    timestamp = datetime.now(UTC).isoformat()
    commit = git_commit()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "commit": commit,
        "model": model,
        "dataset": dataset,
        "subset": subset,
        "dataset_id": datamodule.dataset_id,
        "budget": budget,
        "n_seeds": res["n_seeds"],
        "final_epochs": int(cfg.tuning.final_epochs),
        "variance": _variance_block(cfg, res),
        "in_channels": int(datamodule.in_channels),
        "num_classes": int(datamodule.num_classes),
        "class_names": list(datamodule.class_names),
        "n_val": len(labels_by_split.get("val", [])),
        "n_test": len(labels_by_split.get("test", [])),
        "accelerator": str(cfg.trainer.get("accelerator", "auto")),
        "baseline_config": res["baseline_config"],
        "tuned_config": res["best_config"],
        "search_best_val_acc": float(study.best_value) if study is not None else None,
        "selected_arm": res["selected_arm"],
        "selected_label": res["selected_name"],
        "tuned_ge_baseline": bool(res["tuned_ge_baseline"]),
        # A seed de melhor VALIDAÇÃO de cada lado. O R reproduz com isto o mesmo
        # par que o McNemar histórico usava (nunca a melhor de teste: seria
        # vazamento).
        "mcnemar_baseline_seed": res["mcnemar_baseline_seed"],
        "mcnemar_tuned_seed": res["mcnemar_tuned_seed"],
    }

    def manifest_body(fh: Any) -> None:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, default=str)

    _atomic_write(run_dir / "manifest.json", manifest_body)

    # 5. index.csv (append-only; é o que viabiliza análise ENTRE runs — Friedman
    #    sobre datasets — sem abrir N manifests)
    _append_index(
        root / "index.csv",
        {
            "run_id": run_id,
            "timestamp_utc": timestamp,
            "commit": commit,
            "model": model,
            "dataset": dataset,
            "subset": subset,
            "budget": budget,
            "n_seeds": res["n_seeds"],
            "selected_arm": res["selected_arm"],
            "path": str(Path("runs") / run_id),
        },
    )

    return run_dir


def _variance_block(cfg: Any, res: dict[str, Any]) -> dict[str, Any]:
    """Quais fontes variaram de fato — o relatório precisa dizer qual variância mediu."""
    variance = cfg.get("variance", {}) if hasattr(cfg, "get") else {}
    return {
        "vary_init": bool(variance.get("vary_init", True)),
        "vary_data_order": bool(variance.get("vary_data_order", False)),
        "vary_split": bool(variance.get("vary_split", False)) and not res.get("uses_official_split"),
        "uses_official_split": bool(res.get("uses_official_split")),
    }


def _trial_rows(study: Any) -> list[dict[str, Any]]:
    """Histórico da busca, um dict por trial, sem depender de pandas."""
    rows = []
    for t in study.trials:
        duration = None
        if t.datetime_start is not None and t.datetime_complete is not None:
            duration = (t.datetime_complete - t.datetime_start).total_seconds()
        rows.append(
            {
                "trial_number": t.number,
                "state": t.state.name,
                "value": t.value if t.value is not None else "",
                "duration_s": duration if duration is not None else "",
                "params_json": json.dumps(t.params, sort_keys=True, default=str),
            }
        )
    return rows


def _append_index(path: Path, row: dict[str, Any]) -> None:
    """Acrescenta uma linha ao índice, escrevendo o cabeçalho só na criação."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(INDEX_COLUMNS), extrasaction="raise")
        if is_new:
            writer.writeheader()
        writer.writerow(row)
