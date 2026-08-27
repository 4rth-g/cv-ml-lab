"""Testes do contrato de artefatos (`cvlab.export`).

Não treinam nada: montam `res` sintético e verificam o schema. O ponto é que
esses arquivos são a ÚNICA interface com a análise em R, então mudar uma coluna
sem perceber quebraria o outro lado em silêncio.
"""

import csv
import gzip
import json
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from cvlab.export import (
    PREDICTION_COLUMNS,
    SCHEMA_VERSION,
    SEED_METRICS_COLUMNS,
    export_run_artifacts,
)

N_SEEDS = 2
N_VAL = 4
N_TEST = 6


class FakeDataModule:
    """DataModule mínimo: o export só lê metadados, nunca as imagens."""

    in_channels = 1
    num_classes = 3
    class_names = ["a", "b", "c"]
    dataset_id = "fake-ds-28"


def make_cfg() -> Any:
    return OmegaConf.create(
        {
            "seed": 42,
            "model": {"_target_": "cvlab.models.cnn.ConfigurableCNN"},
            "dataset": {"_target_": "cvlab.data.mnist.MNISTDataModule", "subset": ""},
            "tuning": {"final_epochs": 3},
            "trainer": {"accelerator": "cpu"},
            "variance": {"vary_init": True, "vary_data_order": False, "vary_split": False},
        }
    )


def make_res(uses_official_split: bool = False, targets_override: dict | None = None) -> dict[str, Any]:
    seed_rows = []
    predictions = []
    for arm in ("baseline", "tuned"):
        for i in range(N_SEEDS):
            seed = 42 + i
            for split in ("val", "test"):
                seed_rows.append(
                    {
                        "arm": arm,
                        "seed_init": seed,
                        "seed_data": 42,
                        "seed_split": 42,
                        "split": split,
                        "acc": 0.1 + 0.01 * i,
                    }
                )
            for split, n in (("val", N_VAL), ("test", N_TEST)):
                targets = [j % 3 for j in range(n)]
                if targets_override and (arm, seed, split) in targets_override:
                    targets = targets_override[(arm, seed, split)]
                predictions.append(
                    {
                        "arm": arm,
                        "seed_init": seed,
                        "split": split,
                        "preds": [(j + 1) % 3 for j in range(n)],
                        "targets": targets,
                    }
                )

    return {
        "n_seeds": N_SEEDS,
        "seed_rows": seed_rows,
        "predictions": predictions,
        "baseline_config": {"_target_": "x", "lr": 0.1},
        "best_config": {"_target_": "x", "lr": 0.2},
        "selected_name": "melhor config (Optuna)",
        "selected_arm": "tuned",
        "tuned_ge_baseline": True,
        "mcnemar_baseline_seed": 42,
        "mcnemar_tuned_seed": 43,
        "diff_test_mean": 0.01,
        "uses_official_split": uses_official_split,
    }


def export(tmp_path: Path, **kw: Any) -> Path:
    return export_run_artifacts(
        run_id="run-teste",
        res=make_res(**kw),
        cfg=make_cfg(),
        datamodule=FakeDataModule(),
        output_dir=tmp_path,
    )


def read_csv(path: Path, gzipped: bool = False) -> list[dict[str, str]]:
    opener = gzip.open if gzipped else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        return list(csv.DictReader(fh))


def test_seed_metrics_header_is_exact_and_ordered(tmp_path: Path) -> None:
    """O R lê por posição e por nome: cabeçalho e ordem são parte do contrato."""
    run_dir = export(tmp_path)
    with open(run_dir / "seed_metrics.csv", encoding="utf-8") as fh:
        header = fh.readline().strip().split(",")
    assert tuple(header) == SEED_METRICS_COLUMNS


def test_seed_metrics_row_count(tmp_path: Path) -> None:
    """n_seeds x 2 arms x 2 splits, sem linha faltando nem duplicada."""
    rows = read_csv(export(tmp_path) / "seed_metrics.csv")
    assert len(rows) == N_SEEDS * 2 * 2
    assert {r["arm"] for r in rows} == {"baseline", "tuned"}
    assert {r["split"] for r in rows} == {"val", "test"}


def test_accuracy_survives_float_roundtrip(tmp_path: Path) -> None:
    """A acurácia vai com precisão total: formatar aqui jogaria fora dígitos que o R usa."""
    rows = read_csv(export(tmp_path) / "seed_metrics.csv")
    assert float(rows[0]["acc"]) == 0.1


def test_three_seed_columns_are_recorded(tmp_path: Path) -> None:
    """As três fontes de variância aparecem separadas, mesmo quando só uma varia.

    É justamente o caso em que elas coincidem que precisa ficar registrado: o
    relatório tem que poder dizer que split e ordem de dados NÃO variaram.
    """
    rows = read_csv(export(tmp_path) / "seed_metrics.csv")
    assert {r["seed_init"] for r in rows} == {"42", "43"}
    assert {r["seed_data"] for r in rows} == {"42"}
    assert {r["seed_split"] for r in rows} == {"42"}


def test_official_split_is_marked_not_numbered(tmp_path: Path) -> None:
    """Dataset com split oficial grava 'official', não um número que não existe."""
    rows = read_csv(export(tmp_path, uses_official_split=True) / "seed_metrics.csv")
    assert {r["seed_split"] for r in rows} == {"official"}


def test_predictions_schema_and_row_count(tmp_path: Path) -> None:
    """Val E test são exportados: sem val, escolher métrica pós-hoc olharia o teste."""
    run_dir = export(tmp_path)
    rows = read_csv(run_dir / "predictions.csv.gz", gzipped=True)

    assert len(rows) == N_SEEDS * 2 * (N_VAL + N_TEST)
    assert tuple(rows[0].keys()) == PREDICTION_COLUMNS
    assert {r["split"] for r in rows} == {"val", "test"}

    val_rows = [r for r in rows if r["split"] == "val" and r["arm"] == "baseline" and r["seed_init"] == "42"]
    assert [int(r["example_idx"]) for r in val_rows] == list(range(N_VAL))


def test_manifest_has_required_keys(tmp_path: Path) -> None:
    manifest = json.loads((export(tmp_path) / "manifest.json").read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "run_id",
        "timestamp_utc",
        "commit",
        "model",
        "dataset",
        "subset",
        "dataset_id",
        "budget",
        "n_seeds",
        "variance",
        "in_channels",
        "num_classes",
        "class_names",
        "n_val",
        "n_test",
        "baseline_config",
        "tuned_config",
        "selected_arm",
        "tuned_ge_baseline",
        "mcnemar_baseline_seed",
        "mcnemar_tuned_seed",
    }
    assert required <= set(manifest)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["n_val"] == N_VAL
    assert manifest["n_test"] == N_TEST


def test_index_is_append_only(tmp_path: Path) -> None:
    """Dois exports -> duas linhas e UM cabeçalho. O índice é o que habilita análise entre runs."""
    export(tmp_path)
    export(tmp_path)
    rows = read_csv(tmp_path / "runs" / "index.csv")
    assert len(rows) == 2
    assert all(r["run_id"] == "run-teste" for r in rows)


def test_divergent_labels_are_rejected(tmp_path: Path) -> None:
    """y_true diferente entre arms invalidaria o McNemar e o join com o EDA."""
    override = {("tuned", 43, "test"): [2] * N_TEST}
    with pytest.raises(ValueError, match="y_true diverge"):
        export(tmp_path, targets_override=override)


def test_multilabel_target_is_rejected(tmp_path: Path) -> None:
    """Alvo multi-label não cabe no schema: melhor falhar que gravar CSV errado."""
    override = {("baseline", 42, "val"): [[1, 0, 1]] * N_VAL}
    with pytest.raises(ValueError, match="rótulo único"):
        export(tmp_path, targets_override=override)


def test_no_tmp_files_left_behind(tmp_path: Path) -> None:
    """Escrita atômica: nenhum .tmp sobrevive a um export bem-sucedido."""
    run_dir = export(tmp_path)
    assert list(run_dir.glob("*.tmp")) == []


def test_failed_export_leaves_no_partial_file(tmp_path: Path) -> None:
    """Export abortado não pode deixar CSV truncado que o R leia como completo."""
    with pytest.raises(ValueError):
        export(tmp_path, targets_override={("tuned", 43, "test"): [2] * N_TEST})
    run_dir = tmp_path / "runs" / "run-teste"
    assert not (run_dir / "predictions.csv.gz").exists()
    assert list(tmp_path.rglob("*.tmp")) == []


def test_dataset_id_comes_from_the_datamodule() -> None:
    """export e eda precisam derivar `dataset_id` do MESMO lugar.

    Se cada lado calculasse o nome por conta própria, o EDA iria para um
    diretório e o manifest apontaria para outro. A falha seria silenciosa: o
    relatório apenas diria "EDA ausente", sem nunca revelar que o par existia.
    """
    from cvlab.data.cifar10 import CIFAR10DataModule
    from cvlab.data.med_mnist import MedMNISTDataModule
    from cvlab.data.mnist import MNISTDataModule

    assert MNISTDataModule(data_dir="./data").dataset_id == "mnist"
    assert CIFAR10DataModule(data_dir="./data").dataset_id == "cifar10"
    # MedMNIST sobrescreve: subset e resolução são variantes do mesmo módulo.
    assert MedMNISTDataModule(subset="bloodmnist", size=64).dataset_id == "medmnist-bloodmnist-64"
