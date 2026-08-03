"""Módulo de experiment tracking e testes estatísticos do cvlab."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binom, chi2


def mcnemar_test(
    y_true: np.ndarray | list,
    preds_a: np.ndarray | list,
    preds_b: np.ndarray | list,
) -> dict[str, Any]:
    """McNemar pareado entre dois classificadores no MESMO conjunto de teste.

    Semântica idêntica ao utilitário de referência (fashion-mnist-fundamentos-ia/src/utils.py).
    Retorna {'n01', 'n10', 'statistic', 'p_value', 'method'}.
    Usa binomial exata quando discordantes (n01+n10) < 25; senão chi2 com correção de continuidade.
    """
    yt = np.asarray(y_true)
    pa = np.asarray(preds_a)
    pb = np.asarray(preds_b)

    correct_a = yt == pa
    correct_b = yt == pb

    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    disc = n01 + n10

    if disc < 25:
        method = "binom_exact"
        statistic = float(min(n01, n10))
        if disc == 0:
            p_val = 1.0
        else:
            p_val = min(1.0, float(2 * binom.cdf(statistic, disc, 0.5)))
    else:
        method = "chi2_continuity"
        statistic = float((abs(n01 - n10) - 1) ** 2 / disc)
        p_val = float(chi2.sf(statistic, 1))

    return {
        "n01": n01,
        "n10": n10,
        "statistic": statistic,
        "p_value": p_val,
        "method": method,
    }


def log_experiment(
    model_name: str,
    config: dict,
    metrics: dict,
    seed: int = 42,
    db_path: Path | str | None = None,
) -> None:
    """Registra metadados de treino num SQLite append-only (results/experiment_tracker.db).

    Salva timestamp, commit_hash (com sufixo -dirty se houver alterações não commitadas),
    configuração e métricas finais.
    """
    if db_path is None:
        db_path = Path("results") / "experiment_tracker.db"
    else:
        db_path = Path(db_path)

    db_path.parent.mkdir(exist_ok=True, parents=True)

    commit_hash = os.environ.get("CVLAB_GIT_SHA")
    if not commit_hash:
        try:
            sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
            dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
            commit_hash = f"{sha}{'-dirty' if dirty else ''}"
        except Exception:
            commit_hash = "unknown"

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            commit_hash TEXT,
            model_name TEXT,
            seed INTEGER,
            config JSON,
            metrics JSON
        )
    """)
    c.execute(
        "INSERT INTO experiments (timestamp, commit_hash, model_name, seed, config, metrics) VALUES (?, ?, ?, ?, ?, ?)",
        (
            datetime.now(UTC).isoformat(),
            commit_hash,
            model_name,
            seed,
            json.dumps(config),
            json.dumps(metrics),
        ),
    )
    conn.commit()
    conn.close()


def init_wandb(cfg: Any) -> Any:
    """Helper para inicializar o WandbLogger do PyTorch Lightning se disponível."""
    try:
        from lightning.pytorch.loggers import WandbLogger

        mode = cfg.logger.get("mode", "offline")
        project = cfg.logger.get("project", "cv-ml-lab")
        entity = cfg.logger.get("entity", None)
        save_dir = cfg.logger.get("save_dir", ".")
        name = cfg.logger.get("name", None)
        if not name:
            # Deriva do dataset + data/hora: "mnist-2026-08-02_22-05-30"
            target = str(cfg.dataset.get("_target_", "run"))
            ds = target.split(".")[-1].replace("DataModule", "").lower() or "run"
            name = f"{ds}-{datetime.now():%Y-%m-%d_%H-%M-%S}"
        return WandbLogger(name=name, project=project, mode=mode, entity=entity, save_dir=save_dir)
    except Exception:
        return None
