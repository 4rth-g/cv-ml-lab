"""EDA de dataset: reduz as imagens a uma tabela que o R consegue analisar.

Contrapartida de `cvlab.export`. Enquanto aquele descreve o *resultado* de um
run, este descreve o *dado de entrada*, e os dois compartilham a chave
``(split, example_idx)``::

    results/datasets/<dataset_id>/summary.json      agregados por split e classe
    results/datasets/<dataset_id>/examples.csv.gz   uma linha por exemplo

É essa chave que faz a análise valer a pena. Sozinho, o EDA do MNIST é resultado
conhecido; cruzado com ``predictions.csv.gz`` ele responde perguntas que nenhum
notebook isolado respondia: em que classes o erro se concentra, se os exemplos
que baseline e tunada erram juntos diferem dos que só uma erra, se o erro
correlaciona com intensidade da imagem, e quais exemplos TODAS as seeds erram —
que é o que separa dificuldade intrínseca de instabilidade de treino.

Roda desacoplado do treino::

    just eda medmnist dataset.subset=dermamnist
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from cvlab.data.base import BaseImageClfDataModule
from cvlab.data.registry import get_datamodule

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs")

EXAMPLE_COLUMNS = (
    "dataset_id",
    "split",
    "example_idx",
    "y_true",
    "class_name",
    "mean_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "nonzero_frac",
)

SCHEMA_VERSION = 1


def _denormalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Desfaz o ``Normalize`` para recuperar os pixels em [0, 1].

    As estatísticas precisam descrever a IMAGEM, não a normalização escolhida na
    config: sem isto, trocar ``mean``/``std`` mudaria o EDA de um dataset que não
    mudou. Inverter é exato e vale para qualquer DataModule, o que evita código
    de leitura por dataset.
    """
    return x * std + mean


def _split_datasets(dm: BaseImageClfDataModule) -> dict[str, Any]:
    """Materializa os splits SEM augmentation.

    O ``train_dataset`` normal carrega augmentation aleatória, então as
    estatísticas por exemplo não seriam reproduzíveis entre execuções. O EDA
    descreve o dado, não a política de augmentation.
    """
    original = dm._augment_ops
    dm._augment_ops = lambda: []  # type: ignore[method-assign]
    try:
        dm.setup("fit")
        return {"train": dm.train_dataset, "val": dm.val_dataset, "test": dm.test_dataset}
    finally:
        dm._augment_ops = original  # type: ignore[method-assign]


def export_dataset_artifacts(
    dm: BaseImageClfDataModule,
    output_dir: Path | str,
    dataset_id: str,
    batch_size: int = 512,
) -> Path:
    """Grava ``summary.json`` e ``examples.csv.gz`` do dataset. Devolve o diretório."""
    out_dir = Path(output_dir) / "datasets" / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = _split_datasets(dm)
    class_names = list(dm.class_names)
    mean = torch.tensor(dm.mean).view(-1, 1, 1)
    std = torch.tensor(dm.std).view(-1, 1, 1)

    counts: dict[str, dict[int, int]] = {}
    channel_sums = torch.zeros(dm.in_channels)
    channel_sq_sums = torch.zeros(dm.in_channels)
    n_pixels = 0
    resolution: tuple[int, int] | None = None
    rows_written = 0

    tmp = out_dir / "examples.csv.gz.tmp"
    final = out_dir / "examples.csv.gz"
    try:
        with gzip.open(tmp, "wt", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(EXAMPLE_COLUMNS), extrasaction="raise")
            writer.writeheader()

            for split, dataset in splits.items():
                if dataset is None:
                    continue
                counts[split] = {}
                # shuffle=False: o example_idx precisa casar com o de
                # predictions.csv.gz, que vem dos *_dataloader() do DataModule.
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                idx = 0
                for xb, yb in loader:
                    raw = _denormalize(xb, mean, std).clamp(0.0, 1.0)
                    if resolution is None:
                        resolution = (int(raw.shape[-2]), int(raw.shape[-1]))

                    per_example = raw.flatten(1)
                    means = per_example.mean(dim=1)
                    stds = per_example.std(dim=1)
                    mins = per_example.min(dim=1).values
                    maxs = per_example.max(dim=1).values
                    nonzero = (per_example > 0).float().mean(dim=1)

                    if split == "train":
                        channel_sums += raw.sum(dim=(0, 2, 3))
                        channel_sq_sums += (raw**2).sum(dim=(0, 2, 3))
                        n_pixels += raw.shape[0] * raw.shape[2] * raw.shape[3]

                    labels = [int(v) for v in yb.tolist()] if torch.is_tensor(yb) else [int(v) for v in yb]
                    for i, label in enumerate(labels):
                        counts[split][label] = counts[split].get(label, 0) + 1
                        writer.writerow(
                            {
                                "dataset_id": dataset_id,
                                "split": split,
                                "example_idx": idx,
                                "y_true": label,
                                "class_name": class_names[label],
                                "mean_intensity": float(means[i]),
                                "std_intensity": float(stds[i]),
                                "min_intensity": float(mins[i]),
                                "max_intensity": float(maxs[i]),
                                "nonzero_frac": float(nonzero[i]),
                            }
                        )
                        idx += 1
                        rows_written += 1
        os.replace(tmp, final)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "in_channels": int(dm.in_channels),
        "num_classes": int(dm.num_classes),
        "class_names": class_names,
        "resolution": list(resolution) if resolution else None,
        "n_examples": {split: sum(c.values()) for split, c in counts.items()},
        "class_counts": {split: {class_names[k]: v for k, v in sorted(c.items())} for split, c in counts.items()},
        "class_proportions": {
            split: {class_names[k]: v / max(1, sum(c.values())) for k, v in sorted(c.items())}
            for split, c in counts.items()
        },
        # Razão entre a classe mais e a menos frequente no treino. É o número que
        # decide se acurácia é métrica honesta neste dataset.
        "train_imbalance_ratio": _imbalance_ratio(counts.get("train", {})),
        "channel_mean": _channel_stat(channel_sums, n_pixels),
        "channel_std": _channel_std(channel_sums, channel_sq_sums, n_pixels),
        "normalization_used": {"mean": list(dm.mean), "std": list(dm.std)},
        "total_rows": rows_written,
    }

    tmp_summary = out_dir / "summary.json.tmp"
    try:
        with open(tmp_summary, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_summary, out_dir / "summary.json")
    except BaseException:
        tmp_summary.unlink(missing_ok=True)
        raise

    return out_dir


def _imbalance_ratio(counts: dict[int, int]) -> float | None:
    if not counts:
        return None
    values = [v for v in counts.values() if v > 0]
    return float(max(values) / min(values)) if values else None


def _channel_stat(sums: torch.Tensor, n_pixels: int) -> list[float] | None:
    if n_pixels == 0:
        return None
    return [float(v) for v in (sums / n_pixels)]


def _channel_std(sums: torch.Tensor, sq_sums: torch.Tensor, n_pixels: int) -> list[float] | None:
    if n_pixels == 0:
        return None
    mean = sums / n_pixels
    var = (sq_sums / n_pixels) - mean**2
    return [float(v) for v in var.clamp(min=0).sqrt()]


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="train")
def eda(cfg: DictConfig) -> None:
    """Entry point do EDA. Usa a MESMA config de dataset do treino, então o
    ``dataset_id`` e o ``example_idx`` batem com os do run por construção."""
    dm = get_datamodule(cfg)
    dm.prepare_data()

    # dataset_id vem do DataModule, nunca derivado aqui: `cvlab.export` grava o
    # MESMO valor no manifest, e é por ele que o relatório encontra este EDA.
    dataset_id = dm.dataset_id
    output_dir = Path(cfg.get("output_dir", "results"))

    print(f"=== cv-ml-lab: EDA de dataset ({dataset_id}) ===")
    out = export_dataset_artifacts(dm, output_dir=output_dir, dataset_id=dataset_id)

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    print(f"Exemplos por split: {summary['n_examples']}")
    ratio = summary["train_imbalance_ratio"]
    if ratio is not None:
        print(f"Desbalanço no treino (maior/menor classe): {ratio:.2f}x")
        if ratio > 3:
            print("  ⚠ acurácia é métrica enganosa com este desbalanço; prefira F1/balanced accuracy no relatório.")
    print(f"Artefatos: {out}/")


def main() -> None:
    eda()


if __name__ == "__main__":
    main()
