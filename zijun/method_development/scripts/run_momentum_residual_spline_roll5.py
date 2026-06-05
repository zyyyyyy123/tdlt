from pathlib import Path
import os
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = PROJECT_DIR / "outputs" / ".matplotlib"
XDG_CACHE_DIR = PROJECT_DIR / "outputs" / ".cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.baseline_io import load_reproduced_momentum_predictions
from src.metrics import regression_metrics
from src.targets import add_trailing_rolling_loss


OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"
TARGET_COLUMN = "loss_roll5"
ROLLING_WINDOW = 5
KEY_TARGET = "roll5"
KEY_SCHEDULE = "wsd"
KEY_WINDOW = "full"
SMOOTHING_GRID = [0.01, 0.05, 0.1, 0.5]
SHRINK_GRID = [0.5, 0.75, 1.0]
RESIDUAL_CLIP = 0.25


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_reproduced_momentum_predictions()
    baseline = add_trailing_rolling_loss(
        baseline,
        window=ROLLING_WINDOW,
        target_col=TARGET_COLUMN,
    )

    predictions = build_roll5_predictions(baseline)
    metrics = evaluate_predictions_by_target(predictions)
    key_feature_set = select_key_feature_set(metrics)

    metrics_path = OUTPUT_DIR / "momentum_residual_spline_roll5_metrics.csv"
    predictions_path = OUTPUT_DIR / "key_momentum_residual_roll5_predictions.csv"

    metrics.to_csv(metrics_path, index=False)
    key_predictions = predictions[
        (predictions["correction"] == "smooth_residual_roll5")
        & (predictions["feature_set"] == key_feature_set)
    ].copy()
    key_predictions.to_csv(predictions_path, index=False)

    plot_key_model(
        predictions,
        key_feature_set,
        FIGURE_DIR / "momentum_residual_spline_roll5_full.png",
    )
    plot_key_model(
        predictions,
        key_feature_set,
        FIGURE_DIR / "momentum_residual_spline_roll5_20000_30000.png",
        start_step=20000,
        end_step=30000,
    )

    report = metrics[
        (metrics["schedule"] == "wsd")
        & (metrics["window"].isin(["full", "steps_20000_30000"]))
    ].sort_values(["target", "window", "mae", "correction", "feature_set"])
    print(f"Selected key feature set: {key_feature_set}")
    print(report.to_string(index=False))
    print(f"\nSaved metrics to: {metrics_path.relative_to(PROJECT_DIR)}")
    print(f"Saved key predictions to: {predictions_path.relative_to(PROJECT_DIR)}")


def build_roll5_predictions(baseline: pd.DataFrame) -> pd.DataFrame:
    frames = [build_none_prediction(baseline)]
    for smoothing in SMOOTHING_GRID:
        for shrink in SHRINK_GRID:
            frames.append(
                build_spline_prediction(
                    baseline,
                    smoothing=smoothing,
                    shrink=shrink,
                    residual_clip=RESIDUAL_CLIP,
                )
            )
    return pd.concat(frames, ignore_index=True)


def build_none_prediction(baseline: pd.DataFrame) -> pd.DataFrame:
    frame = baseline.copy()
    frame["correction"] = "none"
    frame["feature_set"] = "none"
    frame["residual_pred"] = 0.0
    frame["pred_loss"] = frame["base_pred_loss"]
    return frame


def build_spline_prediction(
    baseline: pd.DataFrame,
    *,
    smoothing: float,
    shrink: float,
    residual_clip: float,
    train_schedule: str = "cosine",
) -> pd.DataFrame:
    train = baseline[baseline["schedule"] == train_schedule].sort_values("step")
    residual = (
        np.log(train[TARGET_COLUMN])
        - np.log(np.maximum(train["base_pred_loss"], 1e-12))
    )
    spline = UnivariateSpline(
        train["step"].to_numpy(dtype=float),
        residual.to_numpy(dtype=float),
        s=smoothing,
    )

    frame = baseline.copy()
    frame["correction"] = "smooth_residual_roll5"
    frame["feature_set"] = f"roll5_spline_s{smoothing:g}_shrink{shrink:g}"
    frame["residual_pred"] = spline(frame["step"].to_numpy(dtype=float)) * shrink
    frame["residual_pred"] = np.clip(
        frame["residual_pred"],
        -residual_clip,
        residual_clip,
    )
    frame["pred_loss"] = frame["base_pred_loss"] * np.exp(frame["residual_pred"])
    return frame


def evaluate_predictions_by_target(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    windows = {
        "full": (None, None),
        "steps_20000_30000": (20000, 30000),
    }
    targets = {
        "roll5": TARGET_COLUMN,
        "raw": "loss",
    }
    group_cols = ["method", "correction", "feature_set", "schedule"]

    for group_values, group in predictions.groupby(group_cols, sort=False):
        for target_name, target_col in targets.items():
            for window_name, (start_step, end_step) in windows.items():
                window = group
                if start_step is not None:
                    window = window[window["step"] >= start_step]
                if end_step is not None:
                    window = window[window["step"] <= end_step]
                metrics = regression_metrics(window[target_col], window["pred_loss"])
                rows.append(
                    {
                        **dict(zip(group_cols, group_values)),
                        "target": target_name,
                        "window": window_name,
                        "n": len(window),
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def select_key_feature_set(metrics: pd.DataFrame) -> str:
    candidates = metrics[
        (metrics["target"] == KEY_TARGET)
        & (metrics["schedule"] == KEY_SCHEDULE)
        & (metrics["window"] == KEY_WINDOW)
        & (metrics["correction"] == "smooth_residual_roll5")
    ].sort_values("mae")
    if candidates.empty:
        raise RuntimeError("No roll5 smooth residual candidates found.")
    return str(candidates.iloc[0]["feature_set"])


def plot_key_model(
    predictions: pd.DataFrame,
    key_feature_set: str,
    output_path: Path,
    start_step: int | None = None,
    end_step: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schedules = ["cosine", "wsd"]
    fig, axes = plt.subplots(len(schedules), 1, figsize=(10, 6.5), sharex=True)

    for ax, schedule in zip(axes, schedules):
        base = predictions[
            (predictions["schedule"] == schedule)
            & (predictions["correction"] == "none")
        ].sort_values("step")
        corrected = predictions[
            (predictions["schedule"] == schedule)
            & (predictions["correction"] == "smooth_residual_roll5")
            & (predictions["feature_set"] == key_feature_set)
        ].sort_values("step")

        if start_step is not None:
            base = base[base["step"] >= start_step]
            corrected = corrected[corrected["step"] >= start_step]
        if end_step is not None:
            base = base[base["step"] <= end_step]
            corrected = corrected[corrected["step"] <= end_step]

        ax.plot(
            base["step"],
            base["loss"],
            color="#111111",
            linewidth=0.8,
            alpha=0.35,
            label="raw actual",
        )
        ax.plot(
            base["step"],
            base[TARGET_COLUMN],
            color="#111111",
            linewidth=1.2,
            label="roll5 actual",
        )
        ax.plot(
            base["step"],
            base["base_pred_loss"],
            color="#D55E00",
            linestyle="--",
            linewidth=1.4,
            label="momentum baseline",
        )
        ax.plot(
            corrected["step"],
            corrected["pred_loss"],
            color="#0072B2",
            linewidth=1.6,
            label="roll5 residual",
        )
        ax.set_title(schedule)
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if start_step is not None or end_step is not None:
        axes[-1].set_xlim(start_step, end_step)
        fig.suptitle(f"Roll5 residual correction, steps {start_step}-{end_step}")
    else:
        fig.suptitle("Roll5 residual correction, sampled trajectory")

    axes[0].legend(frameon=False, ncol=4)
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
