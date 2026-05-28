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

    plot_key_model(
        predictions,
        FIGURE_DIR / "momentum_residual_spline_full.png",
    )
    plot_key_model(
        predictions,
        FIGURE_DIR / "momentum_residual_spline_20000_30000.png",
        start_step=20000,
        end_step=30000,
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
    fig, axes = plt.subplots(len(schedules), 1, figsize=(10, 6.5), sharex=True)

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

        ax.plot(base["step"], base["loss"], color="#111111", linewidth=1.0, label="actual")
        ax.plot(base["step"], base["base_pred_loss"], color="#D55E00", linestyle="--", linewidth=1.4, label="momentum baseline")
        ax.plot(corrected["step"], corrected["pred_loss"], color="#0072B2", linewidth=1.6, label="smooth residual")
        ax.set_title(schedule)
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if start_step is not None or end_step is not None:
        axes[-1].set_xlim(start_step, end_step)
        fig.suptitle(f"Momentum residual correction, steps {start_step}-{end_step}")
    else:
        fig.suptitle("Momentum residual correction, sampled trajectory")

    axes[0].legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
