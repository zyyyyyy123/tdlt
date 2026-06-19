from pathlib import Path
import os
import sys

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = PROJECT_DIR / "outputs" / ".matplotlib"
XDG_CACHE_DIR = PROJECT_DIR / "outputs" / ".cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib.pyplot as plt

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data import load_loss_curves
from src.feature_fit import FeatureRidgeConfig, FeatureRidgeLossPredictor
from src.metrics import regression_metrics


OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    curves = load_loss_curves()
    train_curve = curves[curves["schedule"] == "cosine"]

    model = FeatureRidgeLossPredictor(FeatureRidgeConfig(alpha=1e-3)).fit(train_curve)

    metric_rows = []
    prediction_frames = []

    for schedule in ["cosine", "wsd", "811"]:
        curve = curves[curves["schedule"] == schedule].sort_values("step").copy()
        pred = model.predict(curve)
        metrics = regression_metrics(curve["loss"], pred)
        metric_rows.append({"schedule": schedule, **metrics})

        pred_frame = curve[["schedule", "step", "loss", "lr"]].copy()
        pred_frame["pred_loss"] = pred
        prediction_frames.append(pred_frame)

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    metrics_df.to_csv(OUTPUT_DIR / "feature_fit_metrics.csv", index=False)
    predictions_df.to_csv(OUTPUT_DIR / "feature_fit_predictions.csv", index=False)

    plot_predictions(predictions_df)
    print(metrics_df.to_string(index=False))


def plot_predictions(predictions: pd.DataFrame) -> None:
    colors = {
        "cosine": "#009E73",
        "wsd": "#D55E00",
        "811": "#0072B2",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    for schedule, frame in predictions.groupby("schedule", sort=False):
        color = colors.get(schedule, "#666666")
        ax.plot(frame["step"], frame["loss"], color=color, alpha=0.25, linewidth=0.8)
        ax.plot(
            frame["step"],
            frame["pred_loss"],
            color=color,
            linewidth=1.8,
            label=f"{schedule} prediction",
        )

    ax.set_title("Feature Ridge Fit: fit cosine, predict other schedules")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "feature_fit_predictions.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
