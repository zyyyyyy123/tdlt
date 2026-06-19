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

from src.metrics import regression_metrics


BASELINE_PATH = (
    PROJECT_DIR
    / "baseline_results"
    / "momentum_wsd_multi_end_check"
    / "predictions.csv"
)
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"
SCHEDULE = "wsd"
MIN_TRAIN_STEP = 1000
KEY_SAMPLE_END = 10000
SMOOTHING = 0.1
SHRINK = 1.0
RESIDUAL_CLIP = 0.25
FEATURE_SET = f"prefix_spline_s{SMOOTHING:g}_shrink{SHRINK:g}_clip{RESIDUAL_CLIP:g}"


def load_momentum_baseline(path: Path = BASELINE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing baseline predictions: {path}. "
            "Run scripts/reproduce_momentum.py with --fit-runs wsd --eval-runs wsd first."
        )

    frame = pd.read_csv(path)
    frame = frame.rename(columns={"run": "schedule", "prediction": "base_pred_loss"})
    required = {
        "sample_end",
        "schedule",
        "step",
        "lr",
        "loss",
        "base_pred_loss",
        "is_sampled",
        "is_unseen_eval",
        "s1",
        "s2",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Baseline file misses columns: {sorted(missing)}")

    frame = frame[frame["schedule"] == SCHEDULE].copy()
    frame["method"] = "momentum_wsd_partial"
    return frame.sort_values(["sample_end", "step"]).reset_index(drop=True)


def build_baseline_prediction(baseline: pd.DataFrame) -> pd.DataFrame:
    frame = baseline.copy()
    frame["correction"] = "none"
    frame["feature_set"] = "none"
    frame["residual_pred"] = 0.0
    frame["pred_loss"] = frame["base_pred_loss"]
    frame["is_train_observed"] = frame["is_sampled"] & (frame["step"] <= frame["sample_end"])
    frame["is_future_eval"] = frame["step"] > frame["sample_end"]
    return frame


def build_smooth_residual_prediction(baseline: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for sample_end, group in baseline.groupby("sample_end", sort=True):
        group = group.sort_values("step").copy()
        observed = group[
            (group["is_sampled"])
            & (group["step"] >= MIN_TRAIN_STEP)
            & (group["step"] <= sample_end)
        ].copy()
        if len(observed) < 4:
            raise ValueError(f"sample_end={sample_end} has too few observed points")

        residual = np.log(observed["loss"]) - np.log(
            np.maximum(observed["base_pred_loss"], 1e-12)
        )
        spline = UnivariateSpline(
            observed["step"].to_numpy(dtype=float),
            residual.to_numpy(dtype=float),
            s=SMOOTHING,
        )

        pred_residual = spline(group["step"].to_numpy(dtype=float)) * SHRINK
        pred_residual = np.clip(pred_residual, -RESIDUAL_CLIP, RESIDUAL_CLIP)

        group["correction"] = "smooth_residual_prefix"
        group["feature_set"] = FEATURE_SET
        group["residual_pred"] = pred_residual
        group["pred_loss"] = group["base_pred_loss"] * np.exp(group["residual_pred"])
        group["is_train_observed"] = (
            group["is_sampled"]
            & (group["step"] >= MIN_TRAIN_STEP)
            & (group["step"] <= sample_end)
        )
        group["is_future_eval"] = group["step"] > sample_end
        parts.append(group)

    return pd.concat(parts, ignore_index=True)


def evaluate_future(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["sample_end", "method", "correction", "feature_set", "schedule"]
    for group_values, group in predictions.groupby(group_cols, sort=False):
        sample_end = int(group_values[0])
        future = group[group["step"] > sample_end].copy()
        if future.empty:
            continue

        metrics = regression_metrics(future["loss"], future["pred_loss"])
        endpoint_true = float(future["loss"].iloc[-1])
        endpoint_pred = float(future["pred_loss"].iloc[-1])
        rows.append(
            {
                **dict(zip(group_cols, group_values)),
                "eval_scope": "future_after_sample_end",
                "n_train_points": int(group["is_train_observed"].sum()),
                "n_eval_points": len(future),
                "eval_step_start": int(future["step"].iloc[0]),
                "eval_step_end": int(future["step"].iloc[-1]),
                "endpoint_true": endpoint_true,
                "endpoint_pred": endpoint_pred,
                "endpoint_diff": endpoint_pred - endpoint_true,
                "endpoint_abs_diff": abs(endpoint_pred - endpoint_true),
                **metrics,
            }
        )
    return pd.DataFrame(rows).sort_values(["sample_end", "correction"]).reset_index(drop=True)


def plot_key_forecast(predictions: pd.DataFrame, output_path: Path) -> None:
    key = predictions[predictions["sample_end"] == KEY_SAMPLE_END].copy()
    baseline = key[key["correction"] == "none"].sort_values("step")
    corrected = key[key["correction"] == "smooth_residual_prefix"].sort_values("step")

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(baseline["step"], baseline["loss"], color="#111111", linewidth=1.0, label="actual WSD")
    ax.plot(
        baseline["step"],
        baseline["base_pred_loss"],
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
        label="prefix residual forecast",
    )
    ax.axvline(KEY_SAMPLE_END, color="#555555", linewidth=1.0, alpha=0.8)
    ax.set_title(f"WSD future forecast after observing steps {MIN_TRAIN_STEP}-{KEY_SAMPLE_END}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metric_curve(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for correction, color, label in [
        ("none", "#D55E00", "momentum baseline"),
        ("smooth_residual_prefix", "#0072B2", "prefix residual forecast"),
    ]:
        group = metrics[metrics["correction"] == correction].sort_values("sample_end")
        ax.plot(
            group["sample_end"],
            group["mae"],
            marker="o",
            linewidth=1.8,
            color=color,
            label=label,
        )
    ax.set_title("Future WSD MAE by observed prefix length")
    ax.set_xlabel("Observed prefix end step")
    ax.set_ylabel("Future MAE")
    ax.grid(alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_momentum_baseline()
    predictions = pd.concat(
        [
            build_baseline_prediction(baseline),
            build_smooth_residual_prediction(baseline),
        ],
        ignore_index=True,
    )
    metrics = evaluate_future(predictions)

    metrics_path = OUTPUT_DIR / "wsd_partial_residual_forecast_metrics.csv"
    predictions_path = OUTPUT_DIR / "wsd_partial_residual_forecast_predictions.csv"
    key_predictions_path = OUTPUT_DIR / "key_wsd_partial_residual_forecast_10000_predictions.csv"

    metrics.to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    predictions[predictions["sample_end"] == KEY_SAMPLE_END].to_csv(
        key_predictions_path,
        index=False,
    )

    plot_key_forecast(
        predictions,
        FIGURE_DIR / "wsd_partial_residual_forecast_10000.png",
    )
    plot_metric_curve(
        metrics,
        FIGURE_DIR / "wsd_partial_residual_forecast_mae.png",
    )

    display_cols = [
        "sample_end",
        "correction",
        "n_train_points",
        "n_eval_points",
        "mae",
        "rmse",
        "mape",
        "r2",
        "endpoint_abs_diff",
    ]
    print(metrics[display_cols].to_string(index=False))
    print(f"\nSaved metrics to: {metrics_path.relative_to(PROJECT_DIR)}")
    print(f"Saved predictions to: {predictions_path.relative_to(PROJECT_DIR)}")
    print(f"Saved key predictions to: {key_predictions_path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
