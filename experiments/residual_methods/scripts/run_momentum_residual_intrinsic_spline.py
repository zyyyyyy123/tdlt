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
from src.evaluation import evaluate_predictions
from src.smooth_residual import (
    SplineResidualConfig,
    build_none_prediction,
    build_spline_prediction,
)


OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"

# Keep this grid small: very low smoothing is slow for 16k sampled points and
# did not change the qualitative conclusion in the first diagnostic run.
SMOOTHING_GRID = [0.1, 0.5]
SHRINK_GRID = [0.5, 0.75, 1.0]
RESIDUAL_CLIP = 0.25
STEP_BASELINE_CONFIGS = [
    SplineResidualConfig(smoothing=0.5, shrink=1.0),
    SplineResidualConfig(smoothing=0.1, shrink=0.75),
    SplineResidualConfig(smoothing=0.1, shrink=1.0),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_reproduced_momentum_predictions()
    predictions = build_predictions(baseline)
    metrics = evaluate_predictions(predictions)

    metrics_path = OUTPUT_DIR / "momentum_residual_intrinsic_spline_metrics.csv"
    predictions_path = OUTPUT_DIR / "key_momentum_residual_intrinsic_spline_predictions.csv"
    metrics.to_csv(metrics_path, index=False)

    key_feature_sets = select_key_feature_sets(metrics)
    key_predictions = predictions[
        predictions["feature_set"].isin(key_feature_sets.values())
    ].copy()
    key_predictions.to_csv(predictions_path, index=False)

    plot_key_models(
        predictions,
        key_feature_sets,
        FIGURE_DIR / "momentum_residual_intrinsic_spline_full.png",
    )
    plot_key_models(
        predictions,
        key_feature_sets,
        FIGURE_DIR / "momentum_residual_intrinsic_spline_20000_30000.png",
        start_step=20000,
        end_step=30000,
    )

    report = metrics[
        (metrics["schedule"] == "wsd")
        & (metrics["window"].isin(["full", "steps_20000_30000"]))
        & (
            metrics["feature_set"].isin(key_feature_sets.values())
            | (
                (metrics["correction"] == "none")
                & (metrics["feature_set"] == "none")
            )
        )
    ].sort_values(["window", "mae", "correction", "feature_set"])

    print("Selected key feature sets:")
    for family, feature_set in key_feature_sets.items():
        print(f"  {family}: {feature_set}")
    print()
    print(report.to_string(index=False))
    print(f"\nSaved metrics to: {metrics_path.relative_to(PROJECT_DIR)}")
    print(f"Saved key predictions to: {predictions_path.relative_to(PROJECT_DIR)}")


def build_predictions(baseline: pd.DataFrame) -> pd.DataFrame:
    frames = [build_none_prediction(baseline)]

    for config in STEP_BASELINE_CONFIGS:
        step_frame = build_spline_prediction(baseline, config)
        step_frame["feature_set"] = "step_" + step_frame["feature_set"]
        step_frame["spline_x"] = step_frame["step"].astype(float)
        frames.append(step_frame)

    for mode in ["raw", "clamp", "ratio"]:
        for smoothing in SMOOTHING_GRID:
            for shrink in SHRINK_GRID:
                frames.append(
                    build_intrinsic_spline_prediction(
                        baseline,
                        smoothing=smoothing,
                        shrink=shrink,
                        mode=mode,
                    )
                )

    return pd.concat(frames, ignore_index=True)


def build_intrinsic_spline_prediction(
    baseline: pd.DataFrame,
    *,
    smoothing: float,
    shrink: float,
    mode: str,
    train_schedule: str = "cosine",
) -> pd.DataFrame:
    if mode not in {"raw", "clamp", "ratio"}:
        raise ValueError(f"Unknown intrinsic-time mode: {mode}")

    train = baseline[baseline["schedule"] == train_schedule].sort_values("s1")
    if mode == "ratio":
        train_x = intrinsic_progress_ratio(train).to_numpy(dtype=float)
    else:
        train_x = train["s1"].to_numpy(dtype=float)
    train_residual = (
        np.log(train["loss"].to_numpy(dtype=float))
        - np.log(np.maximum(train["base_pred_loss"].to_numpy(dtype=float), 1e-12))
    )
    spline = UnivariateSpline(train_x, train_residual, s=smoothing)

    frame = baseline.copy()
    if mode == "ratio":
        x = intrinsic_progress_ratio(frame).to_numpy(dtype=float)
    else:
        x = frame["s1"].to_numpy(dtype=float)
    if mode == "clamp":
        x = np.clip(x, float(train_x.min()), float(train_x.max()))

    frame["correction"] = "smooth_residual_intrinsic_time"
    frame["feature_set"] = f"intrinsic_s1_{mode}_s{smoothing:g}_shrink{shrink:g}"
    frame["intrinsic_time"] = frame["s1"]
    frame["spline_x"] = x
    frame["residual_pred"] = spline(x) * shrink
    frame["residual_pred"] = np.clip(frame["residual_pred"], -RESIDUAL_CLIP, RESIDUAL_CLIP)
    frame["pred_loss"] = frame["base_pred_loss"] * np.exp(frame["residual_pred"])
    return frame


def intrinsic_progress_ratio(frame: pd.DataFrame) -> pd.Series:
    max_s1 = frame.groupby("schedule")["s1"].transform("max")
    return frame["s1"] / max_s1


def select_key_feature_sets(metrics: pd.DataFrame) -> dict[str, str]:
    keys = {
        "step": metrics["feature_set"].str.startswith("step_spline"),
        "intrinsic_raw": metrics["feature_set"].str.startswith("intrinsic_s1_raw"),
        "intrinsic_clamp": metrics["feature_set"].str.startswith("intrinsic_s1_clamp"),
        "intrinsic_ratio": metrics["feature_set"].str.startswith("intrinsic_s1_ratio"),
    }

    selected: dict[str, str] = {}
    for family, mask in keys.items():
        candidates = metrics[
            mask
            & (metrics["schedule"] == "wsd")
            & (metrics["window"] == "full")
        ].sort_values(["mae", "rmse", "feature_set"])
        if candidates.empty:
            raise RuntimeError(f"No candidates found for {family}")
        selected[family] = str(candidates.iloc[0]["feature_set"])
    return selected


def plot_key_models(
    predictions: pd.DataFrame,
    key_feature_sets: dict[str, str],
    output_path: Path,
    start_step: int | None = None,
    end_step: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schedules = ["cosine", "wsd"]
    fig, axes = plt.subplots(len(schedules), 1, figsize=(10, 6.5), sharex=True)

    styles = {
        "step": ("#0072B2", "step spline"),
        "intrinsic_raw": ("#009E73", "intrinsic raw"),
        "intrinsic_clamp": ("#CC79A7", "intrinsic clamp"),
        "intrinsic_ratio": ("#56B4E9", "intrinsic ratio"),
    }

    for ax, schedule in zip(axes, schedules):
        base = predictions[
            (predictions["schedule"] == schedule)
            & (predictions["correction"] == "none")
        ].sort_values("step")
        if start_step is not None:
            base = base[base["step"] >= start_step]
        if end_step is not None:
            base = base[base["step"] <= end_step]

        ax.plot(base["step"], base["loss"], color="#111111", linewidth=1.0, label="actual")
        ax.plot(
            base["step"],
            base["base_pred_loss"],
            color="#D55E00",
            linestyle="--",
            linewidth=1.4,
            label="momentum baseline",
        )

        for family, feature_set in key_feature_sets.items():
            corrected = predictions[
                (predictions["schedule"] == schedule)
                & (predictions["feature_set"] == feature_set)
            ].sort_values("step")
            if start_step is not None:
                corrected = corrected[corrected["step"] >= start_step]
            if end_step is not None:
                corrected = corrected[corrected["step"] <= end_step]

            color, label = styles[family]
            ax.plot(
                corrected["step"],
                corrected["pred_loss"],
                color=color,
                linewidth=1.4,
                label=label,
            )

        ax.set_title(schedule)
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if start_step is not None or end_step is not None:
        axes[-1].set_xlim(start_step, end_step)
        fig.suptitle(f"Step vs intrinsic-time residual spline, steps {start_step}-{end_step}")
    else:
        fig.suptitle("Step vs intrinsic-time residual spline, sampled trajectory")

    axes[0].legend(frameon=False, ncol=5)
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
