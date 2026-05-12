"""Replot training curves directly from ``training_log.csv`` (CSVLogger output).

Why this exists
---------------
The Keras ``History`` object captured at the end of ``model.fit`` and the
``training_history.json`` we save alongside the model both vanish if a long run
crashes. ``training_log.csv`` (written by ``keras.callbacks.CSVLogger`` per
epoch) survives crashes, so it is the safer source of truth when you need to
diagnose convergence.

This script produces:
  * ``overall_loss_csv.png`` — train vs val loss
  * ``{target}_loss_csv.png`` — per-target train vs val loss
  * ``learning_rate_csv.png`` — LR over epochs (only if a 'lr' column exists)
  * ``zoom_first_50_loss_csv.png`` — first-50-epoch zoom on overall loss

Usage
-----
    python replot_training_log_csv.py path/to/artifact_dir
    python replot_training_log_csv.py  # auto-detects via WEATHER_ARTIFACT_DIR
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGETS = ("temperature_k", "pressure_pa", "wind_u", "wind_v")


def _read_csv(path: Path) -> Dict[str, List[float]]:
    columns: Dict[str, List[float]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for name in header:
            columns[name] = []
        for row in reader:
            if len(row) != len(header):
                continue
            for name, value in zip(header, row):
                if value == "":
                    columns[name].append(float("nan"))
                    continue
                try:
                    columns[name].append(float(value))
                except ValueError:
                    columns[name].append(float("nan"))
    return columns


def _ensure_column(cols: Dict[str, List[float]], key: str) -> np.ndarray:
    if key not in cols:
        return np.array([], dtype=np.float64)
    return np.asarray(cols[key], dtype=np.float64)


def _plot_pair(out_path: Path, x: np.ndarray, train: np.ndarray, val: np.ndarray, title: str, ylabel: str) -> None:
    plt.figure(figsize=(11, 6))
    if train.size:
        plt.plot(x, train, linewidth=2.0, linestyle="--", label="Train", zorder=3)
    if val.size:
        plt.plot(x, val, linewidth=2.0, alpha=0.85, marker="o",
                 markersize=2.5, markevery=max(1, len(x) // 60), label="Validation", zorder=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_lr(out_path: Path, x: np.ndarray, lr: np.ndarray) -> None:
    plt.figure(figsize=(11, 4))
    plt.plot(x, lr, linewidth=2.0, color="tab:purple")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.title("Learning rate schedule (per epoch)")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(argv: List[str]) -> int:
    if len(argv) > 1:
        artifact_dir = Path(argv[1]).expanduser().resolve()
    else:
        env_dir = os.getenv("WEATHER_ARTIFACT_DIR")
        if env_dir:
            artifact_dir = Path(env_dir).expanduser().resolve()
        else:
            here = Path(__file__).resolve().parent
            artifact_dir = (here / "artifacts" / "multi_head_mlp_weather").resolve()

    csv_path = artifact_dir / "training_log.csv"
    if not csv_path.exists():
        print(f"ERROR: training_log.csv not found at {csv_path}", file=sys.stderr)
        return 1

    out_dir = artifact_dir / "training_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = _read_csv(csv_path)
    epochs = _ensure_column(cols, "epoch")
    if epochs.size == 0:
        # Fall back to 1..N if no epoch column.
        any_col = next(iter(cols.values()))
        epochs = np.arange(1, len(any_col) + 1, dtype=np.float64)
    else:
        # CSVLogger writes 0-based epochs; shift to 1-based for human reading.
        epochs = epochs + 1.0

    train_loss = _ensure_column(cols, "loss")
    val_loss = _ensure_column(cols, "val_loss")
    if train_loss.size or val_loss.size:
        _plot_pair(
            out_dir / "overall_loss_csv.png",
            epochs, train_loss, val_loss,
            "Overall training/validation loss (from CSV)", "Loss",
        )
        if epochs.size > 50:
            _plot_pair(
                out_dir / "zoom_first_50_loss_csv.png",
                epochs[:50], train_loss[:50] if train_loss.size else train_loss,
                val_loss[:50] if val_loss.size else val_loss,
                "Loss — first 50 epochs (from CSV)", "Loss",
            )

    for target in TARGETS:
        train_key = f"{target}_loss"
        val_key = f"val_{target}_loss"
        train = _ensure_column(cols, train_key)
        val = _ensure_column(cols, val_key)
        if train.size == 0 and val.size == 0:
            continue
        _plot_pair(
            out_dir / f"{target}_loss_csv.png",
            epochs, train, val,
            f"{target} — training/validation loss (from CSV)", "Huber on normalized target",
        )

    for lr_key in ("lr", "learning_rate"):
        lr = _ensure_column(cols, lr_key)
        if lr.size:
            _plot_lr(out_dir / "learning_rate_csv.png", epochs, lr)
            break

    print(f"Saved CSV-based plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
