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

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.baseline_io import load_reproduced_momentum_predictions
from src.data import load_loss_curves


OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"
SCHEDULE_ORDER = ["cosine", "wsd", "811"]
LAGS = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
MIN_STEP = 1000
DETREND_WINDOW = 501
DETREND_MIN_PERIODS = 251
RESIDUAL_ROLLING_WINDOW = 101
RESIDUAL_ROLLING_MIN_PERIODS = 51
DROP_WINDOW_RADIUS = 1000

WINDOW_COLORS = {
    "full_ge1000": "#111111",
    "steps_20000_30000": "#0072B2",
    "wsd_decay": "#D55E00",
    "811_drop80": "#009E73",
    "811_drop90": "#CC79A7",
}

WINDOW_LABELS = {
    "full_ge1000": "full, step >= 1000",
    "steps_20000_30000": "steps 20000-30000",
    "wsd_decay": "WSD decay",
    "811_drop80": "8-1-1 drop 80%",
    "811_drop90": "8-1-1 drop 90%",
}

LOSS_SERIES_LABELS = {
    "log_loss": "log(loss)",
    "diff_log_loss": "diff log(loss)",
    "detrended_log_loss": "detrended log(loss)",
}

RESIDUAL_SERIES_LABELS = {
    "momentum_log_residual": "raw residual",
    "rolling_momentum_residual": "rolling residual",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    curves = prepare_loss_series(load_loss_curves())
    momentum_residuals = prepare_momentum_residual_series()
    windows = build_windows(curves)

    loss_metrics = collect_acf_metrics(
        curves,
        windows,
        analysis="loss",
        series_names=list(LOSS_SERIES_LABELS),
    )
    residual_metrics = collect_acf_metrics(
        momentum_residuals,
        windows,
        analysis="momentum_residual",
        series_names=list(RESIDUAL_SERIES_LABELS),
    )
    metrics = pd.concat([loss_metrics, residual_metrics], ignore_index=True)

    metrics_path = OUTPUT_DIR / "loss_autocorrelation_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    for series_name, label in LOSS_SERIES_LABELS.items():
        plot_loss_acf(
            metrics,
            series_name=series_name,
            title=f"Loss autocorrelation: {label}",
            output_path=FIGURE_DIR / f"loss_acf_{series_name}.png",
        )

    plot_residual_acf(
        metrics,
        output_path=FIGURE_DIR / "loss_acf_momentum_residual.png",
    )

    report = build_summary_report(metrics)
    print(report.to_string(index=False))
    print(f"\nSaved metrics to: {metrics_path.relative_to(PROJECT_DIR)}")
    print(f"Saved figures to: {FIGURE_DIR.relative_to(PROJECT_DIR)}")


def prepare_loss_series(curves: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in curves.groupby("schedule", sort=False):
        frame = group.sort_values("step").copy()
        frame["log_loss"] = np.log(frame["loss"].clip(lower=1e-12))
        frame["diff_log_loss"] = frame["log_loss"].diff()
        trend = frame["log_loss"].rolling(
            window=DETREND_WINDOW,
            center=True,
            min_periods=DETREND_MIN_PERIODS,
        ).mean()
        frame["detrended_log_loss"] = frame["log_loss"] - trend
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


def prepare_momentum_residual_series() -> pd.DataFrame:
    reproduced = load_reproduced_momentum_predictions(
        sampled_only=False,
        min_step=0,
        max_step=33907,
    )
    reproduced = reproduced.copy()
    reproduced["residual_source"] = "reproduced_momentum"

    fallback = load_momentum_fallback_predictions()
    if not fallback.empty:
        missing_schedules = sorted(set(SCHEDULE_ORDER) - set(reproduced["schedule"]))
        fallback = fallback[fallback["schedule"].isin(missing_schedules)].copy()

    if fallback.empty:
        frame = reproduced
    else:
        frame = pd.concat([reproduced, fallback], ignore_index=True)

    pieces = []
    for _, group in frame.groupby("schedule", sort=False):
        group = group.sort_values("step").copy()
        group["momentum_log_residual"] = (
            np.log(group["loss"].clip(lower=1e-12))
            - np.log(group["base_pred_loss"].clip(lower=1e-12))
        )
        group["rolling_momentum_residual"] = group["momentum_log_residual"].rolling(
            window=RESIDUAL_ROLLING_WINDOW,
            center=True,
            min_periods=RESIDUAL_ROLLING_MIN_PERIODS,
        ).mean()
        pieces.append(group)

    return pd.concat(pieces, ignore_index=True)


def load_momentum_fallback_predictions() -> pd.DataFrame:
    path = PROJECT_DIR / "baseline_results" / "baseline_predictions.csv"
    if not path.exists():
        return pd.DataFrame()

    frame = pd.read_csv(path)
    frame = frame[frame["method"] == "momentum"].copy()
    if frame.empty:
        return frame

    frame = frame.rename(columns={"pred_loss": "base_pred_loss"})
    frame["is_sampled"] = True
    frame["residual_source"] = "method_development_baseline"
    return frame[
        [
            "method",
            "schedule",
            "step",
            "lr",
            "loss",
            "base_pred_loss",
            "is_sampled",
            "residual_source",
        ]
    ]


def build_windows(curves: pd.DataFrame) -> list[dict[str, object]]:
    max_step = int(curves["step"].max())
    wsd_decay_start = infer_first_lr_decay_step(curves[curves["schedule"] == "wsd"])
    drop80_center = int(round(max_step * 0.8))
    drop90_center = int(round(max_step * 0.9))

    windows: list[dict[str, object]] = [
        {
            "window": "full_ge1000",
            "schedules": set(SCHEDULE_ORDER),
            "start_step": MIN_STEP,
            "end_step": None,
        },
        {
            "window": "steps_20000_30000",
            "schedules": set(SCHEDULE_ORDER),
            "start_step": 20000,
            "end_step": 30000,
        },
        {
            "window": "wsd_decay",
            "schedules": {"wsd"},
            "start_step": wsd_decay_start,
            "end_step": None,
        },
        {
            "window": "811_drop80",
            "schedules": {"811"},
            "start_step": drop80_center - DROP_WINDOW_RADIUS,
            "end_step": drop80_center + DROP_WINDOW_RADIUS,
        },
        {
            "window": "811_drop90",
            "schedules": {"811"},
            "start_step": drop90_center - DROP_WINDOW_RADIUS,
            "end_step": drop90_center + DROP_WINDOW_RADIUS,
        },
    ]
    return windows


def infer_first_lr_decay_step(frame: pd.DataFrame) -> int:
    if frame.empty:
        return MIN_STEP

    sorted_frame = frame.sort_values("step")
    lr = sorted_frame["lr"].to_numpy(dtype=float)
    steps = sorted_frame["step"].to_numpy(dtype=int)
    tolerance = max(float(np.nanmax(lr)) * 1e-8, 1e-12)
    decays = np.flatnonzero(np.diff(lr) < -tolerance)
    if len(decays) == 0:
        return MIN_STEP
    return int(max(steps[decays[0] + 1], MIN_STEP))


def collect_acf_metrics(
    frame: pd.DataFrame,
    windows: list[dict[str, object]],
    *,
    analysis: str,
    series_names: list[str],
) -> pd.DataFrame:
    rows = []
    for schedule in SCHEDULE_ORDER:
        schedule_frame = frame[frame["schedule"] == schedule].sort_values("step")
        if schedule_frame.empty:
            continue

        for window in windows:
            if schedule not in window["schedules"]:
                continue

            window_frame = select_window(
                schedule_frame,
                start_step=window["start_step"],
                end_step=window["end_step"],
            )
            if window_frame.empty:
                continue
            source = infer_source_label(window_frame)

            for series_name in series_names:
                if series_name not in window_frame:
                    continue

                values = window_frame[series_name].to_numpy(dtype=float)
                acf_values, n_valid = autocorrelation(values, LAGS)
                for lag, acf_value in zip(LAGS, acf_values):
                    rows.append(
                        {
                            "analysis": analysis,
                            "series": series_name,
                            "source": source,
                            "schedule": schedule,
                            "window": window["window"],
                            "step_start": int(window_frame["step"].min()),
                            "step_end": int(window_frame["step"].max()),
                            "n": n_valid,
                            "lag": int(lag),
                            "acf": acf_value,
                        }
                    )

    return pd.DataFrame(rows)


def infer_source_label(frame: pd.DataFrame) -> str:
    if "residual_source" not in frame:
        return "loss_curves"

    values = sorted(str(value) for value in frame["residual_source"].dropna().unique())
    if not values:
        return "unknown"
    return ",".join(values)


def select_window(
    frame: pd.DataFrame,
    *,
    start_step: object,
    end_step: object,
) -> pd.DataFrame:
    selected = frame
    if start_step is not None:
        selected = selected[selected["step"] >= int(start_step)]
    if end_step is not None:
        selected = selected[selected["step"] <= int(end_step)]
    return selected


def autocorrelation(values: np.ndarray, lags: np.ndarray) -> tuple[np.ndarray, int]:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return np.full(len(lags), np.nan), 0

    centered = clean - float(np.mean(clean))
    denominator = float(np.sum(centered * centered))
    if denominator <= 0:
        return np.full(len(lags), np.nan), len(clean)

    result = []
    for lag in lags:
        if lag >= len(centered):
            result.append(np.nan)
            continue
        numerator = float(np.sum(centered[:-lag] * centered[lag:]))
        result.append(numerator / denominator)
    return np.array(result), len(clean)


def plot_loss_acf(
    metrics: pd.DataFrame,
    *,
    series_name: str,
    title: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(SCHEDULE_ORDER), 1, figsize=(10, 8), sharex=True)

    for ax, schedule in zip(axes, SCHEDULE_ORDER):
        subset = metrics[
            (metrics["analysis"] == "loss")
            & (metrics["series"] == series_name)
            & (metrics["schedule"] == schedule)
        ]
        plot_acf_window_lines(ax, subset)
        ax.set_title(schedule)
        ax.set_ylabel("ACF")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Lag")
    add_global_legend(fig, axes)
    fig.suptitle(title, y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_residual_acf(metrics: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(SCHEDULE_ORDER),
        len(RESIDUAL_SERIES_LABELS),
        figsize=(12, 8.5),
        sharex=True,
        sharey=True,
    )

    for row_idx, schedule in enumerate(SCHEDULE_ORDER):
        for col_idx, (series_name, label) in enumerate(RESIDUAL_SERIES_LABELS.items()):
            ax = axes[row_idx, col_idx]
            subset = metrics[
                (metrics["analysis"] == "momentum_residual")
                & (metrics["series"] == series_name)
                & (metrics["schedule"] == schedule)
            ]
            plot_acf_window_lines(ax, subset)
            if row_idx == 0:
                ax.set_title(label)
            if col_idx == 0:
                ax.set_ylabel(f"{schedule}\nACF")
            ax.grid(alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Lag")

    add_global_legend(fig, axes.ravel(), ncol=3)
    fig.suptitle("Momentum residual autocorrelation", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_acf_window_lines(ax: plt.Axes, subset: pd.DataFrame) -> None:
    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle=":")
    if subset.empty:
        ax.text(
            0.5,
            0.5,
            "not available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color="#666666",
        )
        configure_lag_axis(ax)
        return

    for window, group in subset.groupby("window", sort=False):
        group = group.sort_values("lag")
        ax.plot(
            group["lag"],
            group["acf"],
            marker="o",
            markersize=3,
            linewidth=1.4,
            color=WINDOW_COLORS.get(window, "#666666"),
            label=WINDOW_LABELS.get(window, window),
        )

    configure_lag_axis(ax)
    ax.set_ylim(-0.65, 1.05)


def add_global_legend(
    fig: plt.Figure,
    axes: np.ndarray | list[plt.Axes],
    *,
    ncol: int = 3,
) -> None:
    handles_by_label = {}
    for ax in np.ravel(axes):
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            handles_by_label.setdefault(label, handle)

    if not handles_by_label:
        return

    fig.legend(
        handles_by_label.values(),
        handles_by_label.keys(),
        frameon=False,
        ncol=ncol,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )


def configure_lag_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(LAGS)
    ax.set_xticklabels([str(lag) for lag in LAGS])


def build_summary_report(metrics: pd.DataFrame) -> pd.DataFrame:
    key_lags = {1, 16, 64, 128, 512, 2048}
    key_windows = {"full_ge1000", "steps_20000_30000", "wsd_decay", "811_drop80", "811_drop90"}
    report = metrics[
        (metrics["lag"].isin(key_lags))
        & (metrics["window"].isin(key_windows))
        & (
            metrics["series"].isin(
                [
                    "log_loss",
                    "diff_log_loss",
                    "detrended_log_loss",
                    "rolling_momentum_residual",
                ]
            )
        )
    ].copy()
    report["acf"] = report["acf"].round(3)
    return report[
        [
            "analysis",
            "series",
            "source",
            "schedule",
            "window",
            "n",
            "lag",
            "acf",
        ]
    ].sort_values(["analysis", "series", "schedule", "window", "lag"])


if __name__ == "__main__":
    main()
