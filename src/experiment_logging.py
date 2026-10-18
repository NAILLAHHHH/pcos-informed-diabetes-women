"""Weights & Biases offline logging helpers (rubric: traceable experiment logs)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Force offline mode before wandb import side effects in callers
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")

import wandb

from src.config import PROJECT_ROOT, RANDOM_STATE

WANDB_DIR = PROJECT_ROOT / "wandb"
EXPERIMENT_TABLE_PATH = PROJECT_ROOT / "report" / "experiment_results" / "experiment_table.csv"


def init_run(
    name: str,
    config: dict[str, Any],
    group: str = "summative",
    tags: list[str] | None = None,
):
    WANDB_DIR.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project="pcos-informed-diabetes-women",
        name=name,
        group=group,
        tags=tags or [],
        config=config,
        dir=str(PROJECT_ROOT),
        reinit=True,
        settings=wandb.Settings(start_method="thread"),
    )


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    if step is None:
        wandb.log(metrics)
    else:
        wandb.log(metrics, step=step)


def finish_run() -> None:
    if wandb.run is not None:
        wandb.finish()


def append_experiment_row(row: dict[str, Any], path: Path | None = None) -> Path:
    import pandas as pd

    out = path or EXPERIMENT_TABLE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if out.exists():
        prev = pd.read_csv(out)
        frame = pd.concat([prev, frame], ignore_index=True)
    frame.to_csv(out, index=False)
    return out


def base_split_config(n_train: int, n_val: int, n_test: int) -> dict[str, Any]:
    return {
        "random_state": RANDOM_STATE,
        "split": "stratified 70/15/15 (train/val/test)",
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "primary_metric": "macro_f1",
        "wandb_mode": "offline",
    }
