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

import copy
import csv
import gzip
import json
import os
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, open_dict
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


#: Como reduzir um volume 28x28x28 a uma imagem 2D para o EDA visual.
#:
#: - ``mean``: projeção média ao longo da profundidade. A leitura "textura" que
#:   não satura: cada pixel é a média das 28 fatias.
#: - ``std``: desvio ao longo da profundidade. Realça ONDE o volume muda entre
#:   fatias, que é a informação que uma fatia sozinha não tem.
#: - ``mip``: projeção de intensidade MÁXIMA. Padrão em angiografia, onde o vaso
#:   é claro sobre fundo escuro. Em TC satura: o osso é claro em quase toda
#:   fatia e a projeção sai lavada.
#: - ``triplanar``: três projeções médias ortogonais (axial, coronal, sagital)
#:   lado a lado. Mostra que o volume não é isotrópico.
#: - ``mosaic``: as 28 fatias numa grade 7x4. Preserva tudo, mas cada fatia fica
#:   minúscula na montagem por classe.
#: - ``middle``: só a fatia central. O corte que um radiologista olha primeiro.
VOLUME_VIEWS = ("mosaic", "mean", "std", "mip", "triplanar", "middle")


def _volume_to_view(vol: torch.Tensor, view: str = "triplanar") -> torch.Tensor:
    """Volume ``(D, H, W)`` -> imagem ``(1, H', W')`` para o EDA visual.

    Um tensor de 28 canais não é uma imagem: `save_image` só aceita 1 ou 3. Cada
    modo aqui é uma escolha diferente sobre o que preservar. Ver :data:`VOLUME_VIEWS`.
    """
    if view == "middle":
        return vol[vol.shape[0] // 2].unsqueeze(0)

    if view == "mean":
        return vol.mean(dim=0).unsqueeze(0)

    if view == "std":
        # Normaliza para o range visível: o desvio entre fatias é pequeno em
        # valor absoluto e sairia quase preto sem reescalar.
        d = vol.std(dim=0)
        rng = float(d.max() - d.min())
        return ((d - d.min()) / rng if rng > 0 else d).unsqueeze(0)

    if view == "mip":
        return vol.max(dim=0).values.unsqueeze(0)

    if view == "triplanar":
        # Uma MIP por eixo. Juntas dizem se a estrutura é a mesma em toda
        # direção, o que uma fatia sozinha nunca mostra.
        planos = [vol.mean(dim=d) for d in (0, 1, 2)]
        alvo = max(p.shape[0] for p in planos)
        planos = [torch.nn.functional.pad(p, (0, 0, 0, alvo - p.shape[0])) for p in planos]
        return torch.cat(planos, dim=1).unsqueeze(0)

    if view == "mosaic":
        d = vol.shape[0]
        cols = int(d**0.5 + 0.999)
        rows = (d + cols - 1) // cols
        pad = rows * cols - d
        fatias = torch.cat([vol, torch.zeros(pad, *vol.shape[1:])]) if pad else vol
        grade = fatias.reshape(rows, cols, *vol.shape[1:])
        return grade.permute(0, 2, 1, 3).reshape(rows * vol.shape[1], cols * vol.shape[2]).unsqueeze(0)

    raise ValueError(f"volume_view={view!r} desconhecido. Use um de: {list(VOLUME_VIEWS)}.")


def _labels_of_batch(yb: Any, start_idx: int) -> tuple[list[int], list[list[int]]]:
    """Rótulos de um batch, tolerando alvo escalar e vetorial.

    Devolve `(labels, multilabel)`. Em rótulo único, `labels` é a classe de cada
    exemplo e `multilabel` vem vazio. Em multi-label não existe "a classe" do
    exemplo, então `labels` traz a CONTAGEM de achados positivos (o que permite
    um histograma útil) e `multilabel` traz a matriz 0/1 por rótulo.
    """
    arr = yb.tolist() if torch.is_tensor(yb) else list(yb)
    if arr and isinstance(arr[0], (list, tuple)):
        rows = [[int(v) for v in row] for row in arr]
        return [sum(r) for r in rows], rows
    return [int(v) for v in arr], []


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


def export_class_montage(
    dm: BaseImageClfDataModule,
    out_dir: Path,
    split: str = "train",
    per_class: int = 16,
    volume_view: str = "triplanar",
) -> dict[str, Any]:
    """Grava uma grade PNG por classe em ``montage/`` e devolve os metadados.

    Existe para resolver a única parte do EDA que a fronteira entre as
    linguagens bloqueia. O R não decodifica `.npz` nem formato do torchvision,
    então não teria como *olhar* o dataset. Aqui o Python, que já sabe
    decodificar, reduz cada classe a um PNG comum, e o R apenas lê PNG.

    O fluxo continua unidirecional: o R nunca pede nada ao Python, só consome o
    que já está no disco.

    Args:
        split: De qual split tirar os exemplos.
        per_class: Quantos exemplos por classe (vira uma grade quadrada).
    """
    from torchvision.utils import save_image

    splits = _split_datasets(dm)
    dataset = splits.get(split)
    if dataset is None:
        return {}

    is_volume = bool(getattr(dm, "is_3d", False))
    montage_dir = out_dir / "montage"
    montage_dir.mkdir(parents=True, exist_ok=True)
    for stale in montage_dir.glob("*.png"):
        stale.unlink()

    mean = torch.tensor(dm.mean).view(-1, 1, 1)
    std = torch.tensor(dm.std).view(-1, 1, 1)
    class_names = list(dm.class_names)

    # Uma passada só, parando quando todas as classes estiverem cheias: em
    # TissueMNIST (165k exemplos) varrer o dataset inteiro para pegar 16 de cada
    # seria desperdício puro.
    buckets: dict[int, list[torch.Tensor]] = {k: [] for k in range(dm.num_classes)}
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    for xb, yb in loader:
        labels, multilabel_rows = _labels_of_batch(yb, 0)
        for i, img in enumerate(xb):
            # Em multi-label o exemplo pertence a VÁRIOS buckets, um por achado
            # presente. Bucket por "a classe" não existe: a radiografia com
            # cardiomegalia e derrame ilustra as duas.
            alvos = [k for k, v in enumerate(multilabel_rows[i]) if v] if multilabel_rows else [labels[i]]
            raw = None
            for label in alvos:
                if len(buckets[label]) >= per_class:
                    continue
                if raw is None:
                    raw = _denormalize(img, mean, std).clamp(0.0, 1.0)
                    # Volume não vira PNG direto: 28 canais não são uma imagem.
                    if raw.shape[0] not in (1, 3):
                        raw = _volume_to_view(raw, volume_view)
                buckets[label].append(raw)
        if all(len(v) >= per_class for v in buckets.values()):
            break

    nrow = max(1, int(per_class**0.5))
    files: dict[str, str] = {}
    for k, imgs in buckets.items():
        if not imgs:
            continue
        slug = _slug(class_names[k])
        name = f"{k:02d}_{slug}.png"
        save_image(torch.stack(imgs), montage_dir / name, nrow=nrow, padding=1, pad_value=0.35)
        files[class_names[k]] = f"montage/{name}"

    return {
        "split": split,
        "per_class": per_class,
        "nrow": nrow,
        "volume_view": volume_view if any(b and b[0].shape[0] == 1 for b in buckets.values()) and is_volume else None,
        # Resolução das montagens, que pode diferir da usada nas estatísticas.
        # Fica registrada para que o relatório não sugira que as imagens exibidas
        # são exatamente o que o modelo recebeu. `None` em datasets sem o
        # parâmetro (torchvision), onde não há escolha de resolução.
        "size": getattr(dm, "size", None),
        "files": files,
    }


def _slug(name: str) -> str:
    """Nome de classe -> componente de caminho seguro (acento e barra viram '-')."""
    import re
    import unicodedata

    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-") or "classe"


def export_dataset_artifacts(
    dm: BaseImageClfDataModule,
    output_dir: Path | str,
    dataset_id: str,
    batch_size: int = 512,
    per_class_montage: int = 16,
    montage_dm: BaseImageClfDataModule | None = None,
    volume_view: str = "triplanar",
) -> Path:
    """Grava ``summary.json`` e ``examples.csv.gz`` do dataset. Devolve o diretório.

    Args:
        dm: DataModule na resolução do TREINO. Define as estatísticas por exemplo.
        montage_dm: DataModule opcional em OUTRA resolução, usado só para as
            montagens PNG. Existe porque as duas metades do EDA têm exigências
            opostas: a tabela precisa descrever o que o modelo viu, e a imagem
            precisa ser legível para o olho humano. Ver `eda`.
    """
    out_dir = Path(output_dir) / "datasets" / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = _split_datasets(dm)
    class_names = list(dm.class_names)
    mean = torch.tensor(dm.mean).view(-1, 1, 1)
    std = torch.tensor(dm.std).view(-1, 1, 1)

    counts: dict[str, dict[int, int]] = {}
    label_positives: dict[str, list[int]] = {}
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
                    labels, multilabel_rows = _labels_of_batch(yb, idx)
                    raw = _denormalize(xb, mean, std).clamp(0.0, 1.0)
                    if resolution is None:
                        resolution = (int(raw.shape[-2]), int(raw.shape[-1]))

                    per_example = raw.flatten(1)
                    means = per_example.mean(dim=1)
                    stds = per_example.std(dim=1)
                    mins = per_example.min(dim=1).values
                    maxs = per_example.max(dim=1).values
                    nonzero = (per_example > 0).float().mean(dim=1)

                    if multilabel_rows:
                        acc = label_positives.setdefault(split, [0] * dm.num_classes)
                        for row in multilabel_rows:
                            for k, v in enumerate(row):
                                acc[k] += v

                    if split == "train":
                        channel_sums += raw.sum(dim=(0, 2, 3))
                        channel_sq_sums += (raw**2).sum(dim=(0, 2, 3))
                        n_pixels += raw.shape[0] * raw.shape[2] * raw.shape[3]

                    for i, label in enumerate(labels):
                        counts[split][label] = counts[split].get(label, 0) + 1
                        writer.writerow(
                            {
                                "dataset_id": dataset_id,
                                "split": split,
                                "example_idx": idx,
                                "y_true": label,
                                "class_name": class_names[label] if not multilabel_rows else "",
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
        # Em multi-label o "rótulo" agregado é a CONTAGEM de achados do exemplo,
        # não uma classe. Publicar isso como class_counts produziria um
        # "desbalanço" de milhares de vezes que não descreve classe nenhuma.
        "class_counts": None
        if label_positives
        else {split: {class_names[k]: v for k, v in sorted(c.items())} for split, c in counts.items()},
        "findings_per_example": {split: dict(sorted(c.items())) for split, c in counts.items()}
        if label_positives
        else None,
        "class_proportions": None
        if label_positives
        else {
            split: {class_names[k]: v / max(1, sum(c.values())) for k, v in sorted(c.items())}
            for split, c in counts.items()
        },
        # Razão entre a classe mais e a menos frequente no treino. É o número que
        # decide se acurácia é métrica honesta neste dataset.
        "train_imbalance_ratio": None if label_positives else _imbalance_ratio(counts.get("train", {})),
        # Multi-label não tem "distribuição de classes": tem prevalência por
        # achado, e os achados coexistem. A soma passa de 100%.
        "multilabel": bool(label_positives),
        "label_prevalence": {
            split: {class_names[k]: v / max(1, sum(counts[split].values())) for k, v in enumerate(acc)}
            for split, acc in label_positives.items()
        }
        if label_positives
        else None,
        "channel_mean": _channel_stat(channel_sums, n_pixels),
        "channel_std": _channel_std(channel_sums, channel_sq_sums, n_pixels),
        "normalization_used": {"mean": list(dm.mean), "std": list(dm.std)},
        "total_rows": rows_written,
        "montage": export_class_montage(
            montage_dm or dm, out_dir, per_class=per_class_montage, volume_view=volume_view
        ),
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


def _montage_datamodule(cfg: DictConfig) -> BaseImageClfDataModule | None:
    """DataModule extra só para as montagens, quando ``eda.montage_size`` pedir outra resolução.

    Devolve ``None`` quando não há o que fazer: nenhuma resolução pedida, dataset
    sem o parâmetro ``size`` (torchvision), ou resolução igual à do treino.

    O custo é baixo: a montagem lê algumas dezenas de imagens por classe, não o
    split inteiro. Mas o download da outra resolução acontece aqui, então a
    primeira execução é mais lenta.
    """
    requested = cfg.get("eda", {}).get("montage_size", None)
    if requested is None:
        return None
    if "size" not in cfg.dataset:
        print(f"  ⚠ {cfg.dataset._target_} não aceita 'size'; eda.montage_size ignorado.")
        return None
    if int(requested) == int(cfg.dataset.size):
        return None

    alt = copy.deepcopy(cfg)
    with open_dict(alt):
        alt.dataset.size = int(requested)
    montage_dm = get_datamodule(alt)
    montage_dm.prepare_data()
    return montage_dm


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

    # Resolução separada para as montagens. As duas metades do EDA têm exigências
    # opostas: `examples.csv.gz` precisa descrever a imagem que o MODELO recebeu,
    # senão o cruzamento com o erro fica sem sentido; a montagem PNG só precisa
    # ser legível para o olho, e 28x28 é pequeno demais para julgar uma lesão.
    #
    # O `dataset_id` continua vindo da resolução de treino, então o join com as
    # predições segue funcionando.
    montage_dm = _montage_datamodule(cfg)
    if montage_dm is not None:
        print(f"Montagens em {montage_dm.size}px (estatísticas em {getattr(dm, 'size', '?')}px)")

    out = export_dataset_artifacts(
        dm,
        output_dir=output_dir,
        dataset_id=dataset_id,
        per_class_montage=int(cfg.get("eda", {}).get("per_class", 16)),
        montage_dm=montage_dm,
        volume_view=str(cfg.get("eda", {}).get("volume_view", "triplanar")),
    )

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    print(f"Exemplos por split: {summary['n_examples']}")

    if summary.get("multilabel"):
        pv = summary["label_prevalence"]["train"]
        top = sorted(pv.items(), key=lambda kv: -kv[1])[:3]
        print("Multi-label: prevalência por achado (os achados coexistem, a soma passa de 100%)")
        for name, v in top:
            print(f"  {name}: {100 * v:.2f}%")
        print(f"  achado mais raro: {min(pv.values()) * 100:.2f}%")
        print("  ⚠ prevalência baixa por achado: use AUC por rótulo, não acurácia.")
    else:
        ratio = summary["train_imbalance_ratio"]
        if ratio is not None:
            print(f"Desbalanço no treino (maior/menor classe): {ratio:.2f}x")
            if ratio > 3:
                print("  ⚠ acurácia é métrica enganosa com este desbalanço; prefira F1/balanced accuracy.")
    print(f"Artefatos: {out}/")


def main() -> None:
    eda()


if __name__ == "__main__":
    main()
