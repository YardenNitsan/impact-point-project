"""
Replot training curves for the NumPy MLP weather model.

Reads the saved training history from the .npz artefact and generates
two PNG variants per target:

  - {target}_training_validation_loss_visible.png  (full curve, polished)
  - {target}_training_validation_loss_zoom.png     (first N epochs zoomed in)

Output goes to: artifacts/replotted_numpy_mlp_curves/

Usage:
    python replot_numpy_mlp_curves.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from project root
THIS_FILE = Path(__file__).resolve()
ML_SERVICE_DIR = THIS_FILE.parent
if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))

from era5_gam_weather.numpy_mlp_model import NumpyMLPWeatherModel

ARTIFACT_DIR = ML_SERVICE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "numpy_mlp_weather_model.npz"
OUT_DIR = ARTIFACT_DIR / "replotted_numpy_mlp_curves"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_NAMES = ["T", "P", "U", "V"]


def save_visible_curve(target: str, train: np.ndarray, val: np.ndarray) -> None:
    n = len(train)
    epochs = np.arange(1, n + 1)
    plt.figure(figsize=(11, 6))
    plt.plot(epochs, val, linewidth=2, alpha=0.75, marker="o",
             markersize=2.5, markevery=max(1, n // 60),
             label="Validation loss", zorder=2)
    plt.plot(epochs, train, linewidth=2.2, linestyle="--", alpha=0.95,
             label="Training loss", zorder=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (normalised MSE)")
    plt.title(f"{target} - Training and Validation Loss (NumPy MLP)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{target.lower()}_training_validation_loss_visible.png", dpi=300)
    plt.close()


def save_zoom_curve(target: str, train: np.ndarray, val: np.ndarray, first_n: int) -> None:
    n = min(len(train), first_n)
    epochs = np.arange(1, n + 1)
    plt.figure(figsize=(11, 6))
    plt.plot(epochs, val[:n], linewidth=2, alpha=0.75, marker="o",
             markersize=3, markevery=max(1, n // 20),
             label="Validation loss", zorder=2)
    plt.plot(epochs, train[:n], linewidth=2.2, linestyle="--", alpha=0.95,
             label="Training loss", zorder=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (normalised MSE)")
    plt.title(f"{target} - Training and Validation Loss - Zoom (NumPy MLP)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{target.lower()}_training_validation_loss_zoom.png", dpi=300)
    plt.close()


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"ERROR: NumPy MLP model not found: {MODEL_PATH}")
        print("Train the model first:  python train_numpy_mlp.py")
        sys.exit(1)

    model = NumpyMLPWeatherModel.load(str(MODEL_PATH))
    print(f"Loaded model from: {MODEL_PATH}")
    print(f"Saving plots to:   {OUT_DIR}")
    print()

    for tgt in TARGET_NAMES:
        history = model.histories[tgt]
        train = np.asarray(history.train_loss, dtype=np.float64)
        val = np.asarray(history.val_loss, dtype=np.float64)

        if train.size == 0:
            print(f"[{tgt}] no training history found, skipping.")
            continue

        if val.size == 0:
            val = np.full_like(train, np.nan)
        else:
            n = min(len(train), len(val))
            train = train[:n]
            val = val[:n]

        if np.all(np.isfinite(val)):
            max_diff = float(np.max(np.abs(train - val)))
            mean_diff = float(np.mean(np.abs(train - val)))
            print(f"[{tgt}] epochs={len(train)} | "
                  f"max_abs_diff={max_diff:.10f} | mean_abs_diff={mean_diff:.10f}")
        else:
            print(f"[{tgt}] epochs={len(train)} (no validation curve)")

        save_visible_curve(tgt, train, val)
        zoom_n = 50 if tgt in ("T", "P") else 100
        save_zoom_curve(tgt, train, val, first_n=zoom_n)

    print("\nDone.")


if __name__ == "__main__":
    main()
