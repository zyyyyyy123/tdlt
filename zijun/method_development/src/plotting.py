from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCHEDULE_ORDER = ["cosine", "wsd", "811"]


def plot_residual_predictions(
    predictions: pd.DataFrame,
    output_path: Path,
    method: str = "mpl",
    feature_set: str = "drop_momentum",
    start_step: int | None = None,
    end_step: int | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(SCHEDULE_ORDER), 1, figsize=(10, 9), sharex=True)
    for ax, schedule in zip(axes, SCHEDULE_ORDER):
        base = predictions[
            (predictions["method"] == method)
            & (predictions["schedule"] == schedule)
            & (predictions["correction"] == "none")
        ].sort_values("step")
        corrected = predictions[
            (predictions["method"] == method)
            & (predictions["schedule"] == schedule)
            & (predictions["correction"] == "residual")
            & (predictions["feature_set"] == feature_set)
        ].sort_values("step")

        if start_step is not None:
            base = base[base["step"] >= start_step]
            corrected = corrected[corrected["step"] >= start_step]
        if end_step is not None:
            base = base[base["step"] <= end_step]
            corrected = corrected[corrected["step"] <= end_step]

        ax.plot(base["step"], base["loss"], color="#111111", linewidth=1.0, label="actual")
        ax.plot(base["step"], base["base_pred_loss"], color="#D55E00", linewidth=1.4, linestyle="--", label=f"{method} base")
        ax.plot(corrected["step"], corrected["pred_loss"], color="#0072B2", linewidth=1.6, label="residual corrected")
        ax.set_title(schedule)
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Step")
    if start_step is not None or end_step is not None:
        axes[-1].set_xlim(start_step, end_step)
        title = f"Residual-over-{method.upper()} ({feature_set}, steps {start_step}-{end_step})"
    else:
        title = f"Residual-over-{method.upper()} ({feature_set}, full curve)"
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
