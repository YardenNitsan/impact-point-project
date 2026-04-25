from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

THIS_FILE = Path(__file__).resolve()
PACKAGE_DIR = THIS_FILE.parent                  # era5_gam_weather
ML_SERVICE_DIR = PACKAGE_DIR.parent             # ML_service

if str(ML_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(ML_SERVICE_DIR))

from era5_gam_weather.tree_model import WeatherTreeBundle


ARTIFACT_DIR = ML_SERVICE_DIR / "artifacts"
OUT_DIR = ARTIFACT_DIR / "replotted_training_curves"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_model_path() -> Path:
    candidates = sorted(ARTIFACT_DIR.glob("weather_tree_bundle_*.joblib"))
    if not candidates:
        raise FileNotFoundError(
            f"לא נמצא שום קובץ model בתיקייה:\n{ARTIFACT_DIR}\n"
            "חיפשתי קבצים בשם weather_tree_bundle_*.joblib"
        )
    return candidates[-1]


def as_positive_loss_curve(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr

    # אם sklearn שמר score שלילי
    if np.nanmean(arr) < 0:
        arr = -arr

    # לפעמים האיבר הראשון הוא baseline
    if arr.size > 1:
        arr = arr[1:]

    return arr


def save_visible_curve(target_name: str, train_curve: np.ndarray, val_curve: np.ndarray) -> None:
    n = len(train_curve)
    iterations = np.arange(1, n + 1)

    plt.figure(figsize=(11, 6))

    plt.plot(
        iterations,
        val_curve,
        linewidth=2,
        alpha=0.75,
        marker="o",
        markersize=2.5,
        markevery=max(1, n // 60),
        label="Validation loss",
        zorder=2,
    )

    plt.plot(
        iterations,
        train_curve,
        linewidth=2.2,
        linestyle="--",
        alpha=0.95,
        label="Training loss",
        zorder=3,
    )

    plt.xlabel("Boosting iteration")
    plt.ylabel("Loss")
    plt.title(f"{target_name} - Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{target_name.lower()}_training_validation_loss_visible.png", dpi=300)
    plt.close()


def save_zoom_curve(target_name: str, train_curve: np.ndarray, val_curve: np.ndarray, first_n: int) -> None:
    n = min(len(train_curve), first_n)
    iterations = np.arange(1, n + 1)

    plt.figure(figsize=(11, 6))

    plt.plot(
        iterations,
        val_curve[:n],
        linewidth=2,
        alpha=0.75,
        marker="o",
        markersize=3,
        markevery=max(1, n // 20),
        label="Validation loss",
        zorder=2,
    )

    plt.plot(
        iterations,
        train_curve[:n],
        linewidth=2.2,
        linestyle="--",
        alpha=0.95,
        label="Training loss",
        zorder=3,
    )

    plt.xlabel("Boosting iteration")
    plt.ylabel("Loss")
    plt.title(f"{target_name} - Training and Validation Loss (Zoom)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{target_name.lower()}_training_validation_loss_zoom.png", dpi=300)
    plt.close()


def main() -> None:
    model_path = find_model_path()
    bundle = WeatherTreeBundle.load(str(model_path))

    print(f"Loaded model from: {model_path}")
    print(f"Saving plots to:   {OUT_DIR}")
    print()

    for target_name, model in bundle.models.items():
        train_curve = as_positive_loss_curve(getattr(model, "train_score_", []))
        val_curve = as_positive_loss_curve(getattr(model, "validation_score_", []))

        if train_curve.size == 0:
            print(f"[{target_name}] no train curve found, skipping.")
            continue

        if val_curve.size > 0:
            n = min(len(train_curve), len(val_curve))
            train_curve = train_curve[:n]
            val_curve = val_curve[:n]
        else:
            val_curve = np.full_like(train_curve, np.nan)

        if np.all(np.isfinite(val_curve)):
            max_abs_diff = float(np.max(np.abs(train_curve - val_curve)))
            mean_abs_diff = float(np.mean(np.abs(train_curve - val_curve)))
            print(
                f"[{target_name}] "
                f"iters={len(train_curve)} | "
                f"max_abs_diff={max_abs_diff:.10f} | "
                f"mean_abs_diff={mean_abs_diff:.10f}"
            )
        else:
            print(f"[{target_name}] validation curve missing")

        save_visible_curve(target_name, train_curve, val_curve)
        zoom_n = 80 if target_name in ("T", "P") else 200
        save_zoom_curve(target_name, train_curve, val_curve, first_n=zoom_n)

    print("\nDone.")


if __name__ == "__main__":
    main()