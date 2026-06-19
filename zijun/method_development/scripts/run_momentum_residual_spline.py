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
import pandas as pd

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.baseline_io import load_reproduced_momentum_predictions
from src.evaluation import evaluate_predictions
from src.smooth_residual import (
    SplineResidualConfig,
    build_mean_shift_prediction,
    build_none_prediction,
    build_spline_prediction,
)


OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"
KEY_FEATURE_SET = "spline_s0.1_shrink1"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_reproduced_momentum_predictions()
    attempts = [
        build_none_prediction(baseline),
        build_mean_shift_prediction(baseline),
        build_spline_prediction(baseline, SplineResidualConfig(smoothing=0.5, shrink=1.0)),
        build_spline_prediction(baseline, SplineResidualConfig(smoothing=0.1, shrink=0.75)),
        build_spline_prediction(baseline, SplineResidualConfig(smoothing=0.1, shrink=1.0)),
    ]
    predictions = pd.concat(attempts, ignore_index=True)
    metrics = evaluate_predictions(predictions)

    metrics.to_csv(OUTPUT_DIR / "momentum_residual_spline_metrics.csv", index=False)

    key_predictions = predictions[
        (predictions["correction"] == "smooth_residual")
        & (predictions["feature_set"] == KEY_FEATURE_SET)
    ].copy()
    key_predictions.to_csv(OUTPUT_DIR / "key_momentum_residual_predictions.csv", index=False)

    for filename in [
        "momentum_residual_spline_full.png",
        "momentum_residual_spline_full_smoothed.png",
    ]:
        plot_key_model(
            predictions,
            FIGURE_DIR / filename,
        )
    for filename in [
        "momentum_residual_spline_20000_30000.png",
        "momentum_residual_spline_20000_30000_smoothed.png",
    ]:
        plot_key_model(
            predictions,
            FIGURE_DIR / filename,
            start_step=20000,
            end_step=30000,
        )
    plot_contrast_model(
        predictions,
        FIGURE_DIR / "momentum_residual_spline_contrast.png",
    )

    report = metrics[
        (metrics["schedule"] == "wsd")
        & (metrics["window"].isin(["full", "steps_20000_30000"]))
    ].sort_values(["window", "correction", "feature_set"])
    print(report.to_string(index=False))


def plot_key_model(
    predictions: pd.DataFrame,
    output_path: Path,
    start_step: int | None = None,
    end_step: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schedules = ["cosine", "wsd"]
    fig, axes = plt.subplots(len(schedules), 1, figsize=(7.0, 4.6), sharex=True)
    display_window = 251 if start_step is not None or end_step is not None else 501

    for ax, schedule in zip(axes, schedules):
        base = predictions[
            (predictions["schedule"] == schedule)
            & (predictions["correction"] == "none")
        ].sort_values("step")
        corrected = predictions[
            (predictions["schedule"] == schedule)
            & (predictions["correction"] == "smooth_residual")
            & (predictions["feature_set"] == KEY_FEATURE_SET)
        ].sort_values("step")

        if start_step is not None:
            base = base[base["step"] >= start_step]
            corrected = corrected[corrected["step"] >= start_step]
        if end_step is not None:
            base = base[base["step"] <= end_step]
            corrected = corrected[corrected["step"] <= end_step]

        actual_smooth = smooth_for_display(base["loss"], display_window)
        baseline_smooth = smooth_for_display(base["base_pred_loss"], max(51, display_window // 2))
        corrected_smooth = smooth_for_display(corrected["pred_loss"], max(51, display_window // 2))

        raw_base = downsample_for_display(base)
        ax.plot(
            raw_base["step"],
            raw_base["loss"],
            color="#4D4D4D",
            alpha=0.16,
            linewidth=0.45,
            label="actual raw",
            rasterized=True,
            zorder=1,
        )
        ax.plot(
            base["step"],
            actual_smooth,
            color="#111111",
            linewidth=1.4,
            label="actual smoothed",
            zorder=4,
        )
        ax.plot(
            base["step"],
            baseline_smooth,
            color="#D55E00",
            linestyle="--",
            linewidth=1.35,
            label="momentum baseline",
            zorder=3,
        )
        ax.plot(
            corrected["step"],
            corrected_smooth,
            color="#0072B2",
            linewidth=1.55,
            label="spline correction",
            zorder=5,
        )
        title = "cosine fit" if schedule == "cosine" else "WSD transfer"
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.18, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    if start_step is not None or end_step is not None:
        axes[-1].set_xlim(start_step, end_step)

    axes[0].legend(
        frameon=True,
        framealpha=0.88,
        edgecolor="#DDDDDD",
        fontsize=7.5,
        ncol=2,
        loc="upper right",
        borderpad=0.45,
        handlelength=2.1,
    )
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_contrast_model(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display_window = 501

    base = predictions[
        (predictions["schedule"] == "wsd")
        & (predictions["correction"] == "none")
    ].sort_values("step")
    corrected = predictions[
        (predictions["schedule"] == "wsd")
        & (predictions["correction"] == "smooth_residual")
        & (predictions["feature_set"] == KEY_FEATURE_SET)
    ].sort_values("step")

    step = base["step"].to_numpy()
    actual_smooth = smooth_for_display(base["loss"], display_window).to_numpy()
    baseline_smooth = smooth_for_display(base["base_pred_loss"], 251).to_numpy()
    corrected_smooth = smooth_for_display(corrected["pred_loss"], 251).to_numpy()

    baseline_abs_error = smooth_for_display(
        (base["base_pred_loss"].reset_index(drop=True) - base["loss"].reset_index(drop=True)).abs(),
        display_window,
    ).to_numpy()
    corrected_abs_error = smooth_for_display(
        (corrected["pred_loss"].reset_index(drop=True) - corrected["loss"].reset_index(drop=True)).abs(),
        display_window,
    ).to_numpy()
    raw_baseline_mae = float((base["base_pred_loss"] - base["loss"]).abs().mean())
    raw_corrected_mae = float((corrected["pred_loss"] - corrected["loss"]).abs().mean())
    mae_reduction = 100.0 * (1.0 - raw_corrected_mae / raw_baseline_mae)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.7, 1.0], "hspace": 0.12},
    )
    loss_ax, error_ax = axes

    raw_base = downsample_for_display(base)
    loss_ax.plot(
        raw_base["step"],
        raw_base["loss"],
        color="#4D4D4D",
        alpha=0.12,
        linewidth=0.4,
        label="actual raw",
        rasterized=True,
        zorder=1,
    )
    loss_ax.fill_between(
        step,
        baseline_smooth,
        corrected_smooth,
        color="#0072B2",
        alpha=0.16,
        linewidth=0,
        label="baseline-to-spline gap",
        zorder=2,
    )
    loss_ax.plot(
        step,
        actual_smooth,
        color="#111111",
        linewidth=1.45,
        label="actual smoothed",
        zorder=5,
    )
    loss_ax.plot(
        step,
        baseline_smooth,
        color="#D55E00",
        linestyle="--",
        linewidth=1.45,
        label="momentum baseline",
        zorder=4,
    )
    loss_ax.plot(
        step,
        corrected_smooth,
        color="#0072B2",
        linewidth=1.8,
        label="spline correction",
        zorder=6,
    )
    loss_ax.text(
        0.015,
        0.08,
        f"WSD MAE: {raw_baseline_mae:.4f} -> {raw_corrected_mae:.4f} ({mae_reduction:.0f}% lower)",
        transform=loss_ax.transAxes,
        fontsize=8.2,
        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.9, "pad": 3.0},
    )
    loss_ax.set_title("WSD transfer: loss trajectory", fontsize=10, pad=4)
    loss_ax.set_ylabel("Loss")
    loss_ax.legend(
        frameon=True,
        framealpha=0.9,
        edgecolor="#DDDDDD",
        fontsize=7.2,
        ncol=2,
        loc="upper right",
        borderpad=0.4,
        handlelength=2.0,
    )

    error_reduction = baseline_abs_error - corrected_abs_error
    improved = error_reduction >= 0
    error_ax.fill_between(
        step,
        0,
        error_reduction,
        where=improved,
        color="#0072B2",
        alpha=0.26,
        interpolate=True,
        linewidth=0,
        label="spline better",
    )
    error_ax.fill_between(
        step,
        0,
        error_reduction,
        where=~improved,
        color="#D55E00",
        alpha=0.22,
        interpolate=True,
        linewidth=0,
        label="spline worse",
    )
    error_ax.axhline(0, color="#111111", linewidth=0.8, alpha=0.8)
    error_ax.plot(
        step,
        error_reduction,
        color="#0072B2",
        linewidth=1.55,
        label="error reduction",
        zorder=4,
    )
    error_ax.set_title("Absolute-error reduction (positive = spline better)", fontsize=9, pad=3)
    error_ax.set_ylabel("reduction")
    error_ax.set_xlabel("Step")
    error_ax.set_ylim(error_reduction.min() * 1.15, error_reduction.max() * 1.18)
    error_ax.legend(
        frameon=False,
        fontsize=7.0,
        ncol=3,
        loc="upper right",
        handlelength=1.8,
    )

    for ax in axes:
        ax.grid(alpha=0.18, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.12, hspace=0.14)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def smooth_for_display(values: pd.Series, window: int) -> pd.Series:
    series = values.reset_index(drop=True).astype(float)
    if len(series) < 5:
        return series

    window = min(window, len(series))
    if window % 2 == 0:
        window -= 1
    if window < 5:
        return series

    min_periods = max(5, window // 8)
    smoothed = series.rolling(window, center=True, min_periods=min_periods).median()

    mean_window = max(5, window // 5)
    mean_window = min(mean_window, len(series))
    if mean_window % 2 == 0:
        mean_window -= 1
    smoothed = smoothed.rolling(
        mean_window,
        center=True,
        min_periods=max(3, mean_window // 2),
    ).mean()
    return smoothed.bfill().ffill()


def downsample_for_display(frame: pd.DataFrame, max_points: int = 6000) -> pd.DataFrame:
    stride = max(1, len(frame) // max_points)
    return frame.iloc[::stride]


if __name__ == "__main__":
    main()
