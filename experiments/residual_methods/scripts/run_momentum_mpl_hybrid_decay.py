from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

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


OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"
PROJECT_ROOT = PROJECT_DIR.parents[1]
MOMENTUM_PREDICTIONS = PROJECT_ROOT / "results" / "intermediates" / "three_schedule_momentum" / "predictions.csv"

TRAIN_SCHEDULE = "cosine"
VALIDATION_SCHEDULE = "811"
TEST_SCHEDULE = "wsd"
START_STEP = 1000
END_STEP = 33906
SAMPLE_INTERVAL = 2
RESIDUAL_CLIP = 0.25
KEY_METHOD = "hybrid_decay"


@dataclass(frozen=True)
class Candidate:
    feature_set: str
    scale: float
    beta: float
    ridge_alpha: float
    shrink: float


def load_three_schedule_momentum_predictions() -> pd.DataFrame:
    frame = pd.read_csv(MOMENTUM_PREDICTIONS)
    frame = frame.rename(columns={"run": "schedule", "momentum_s2": "base_pred_loss"})
    frame = frame[
        ["schedule", "step", "loss", "lr", "s1", "s2", "base_pred_loss"]
    ].copy()
    frame = frame[
        (frame["step"] >= START_STEP)
        & (frame["step"] <= END_STEP)
        & (((frame["step"] - START_STEP) % SAMPLE_INTERVAL) == 0)
    ].copy()
    frame["is_sampled"] = True
    frame["method"] = "three_schedule_momentum_baseline"
    return frame.sort_values(["schedule", "step"]).reset_index(drop=True)


def add_decay_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    out = []
    for schedule, group in frame.groupby("schedule", sort=False):
        group = group.sort_values("step").copy()
        lr = group["lr"].to_numpy(dtype=np.float64)
        drop = np.maximum(np.r_[0.0, lr[:-1] - lr[1:]], 0.0)
        drop_mass = np.cumsum(drop, dtype=np.float64)

        first_drop_idx = int(np.argmax(drop > 0.0)) if np.any(drop > 0.0) else len(group)
        s1 = group["s1"].to_numpy(dtype=np.float64)
        if first_drop_idx < len(group):
            s1_after_first_drop = np.maximum(s1 - s1[first_drop_idx], 0.0)
        else:
            s1_after_first_drop = np.zeros_like(s1)

        group["lr_drop"] = drop
        group["drop_mass"] = drop_mass
        group["s1_after_first_drop"] = s1_after_first_drop
        group["schedule_progress"] = (
            (group["step"].to_numpy(dtype=np.float64) - START_STEP)
            / max(float(END_STEP - START_STEP), 1.0)
        )
        group["first_drop_step"] = int(group["step"].iloc[first_drop_idx]) if first_drop_idx < len(group) else np.nan
        out.append(group)
    return pd.concat(out, ignore_index=True)


def train_scales(frame: pd.DataFrame) -> dict[str, float]:
    train = frame[frame["schedule"] == TRAIN_SCHEDULE]
    return {
        "s2": max(float(train["s2"].max()), 1e-12),
        "drop_mass": max(float(train["drop_mass"].max()), 1e-12),
        "s1_after_first_drop": max(float(train["s1_after_first_drop"].max()), 1e-12),
    }


def saturating_power(x: np.ndarray, scale: float, beta: float) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    return 1.0 - np.power(1.0 + scale * x, -beta)


def build_feature_matrix(
    frame: pd.DataFrame,
    feature_set: str,
    scale: float,
    beta: float,
    scales: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    s2_norm = frame["s2"].to_numpy(dtype=np.float64) / scales["s2"]
    drop_norm = frame["drop_mass"].to_numpy(dtype=np.float64) / scales["drop_mass"]
    tail_norm = frame["s1_after_first_drop"].to_numpy(dtype=np.float64) / scales["s1_after_first_drop"]

    sat_s2 = saturating_power(s2_norm, scale, beta)
    sat_drop = saturating_power(drop_norm, scale, beta)
    sat_tail = saturating_power(tail_norm, scale, beta)
    progress = frame["schedule_progress"].to_numpy(dtype=np.float64)

    columns: list[np.ndarray] = []
    names: list[str] = []

    if feature_set == "sat_s2":
        columns = [sat_s2]
        names = ["sat_s2"]
    elif feature_set == "sat_s2_drop":
        columns = [sat_s2, sat_drop]
        names = ["sat_s2", "sat_drop"]
    elif feature_set == "sat_s2_tail":
        columns = [sat_s2, sat_tail]
        names = ["sat_s2", "sat_tail"]
    elif feature_set == "mpl_proxy":
        columns = [sat_s2, sat_drop * sat_tail]
        names = ["sat_s2", "sat_drop_times_sat_tail"]
    elif feature_set == "hybrid_decay_progress":
        columns = [sat_s2, sat_drop * sat_tail, sat_s2 * progress]
        names = ["sat_s2", "sat_drop_times_sat_tail", "sat_s2_times_progress"]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    x = np.stack(columns, axis=1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, names


def fit_ridge_residual(x: np.ndarray, y: np.ndarray, ridge_alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def predict_ridge_residual(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    return design @ coef


def apply_log_residual(frame: pd.DataFrame, residual: np.ndarray, correction: str, feature_set: str) -> pd.DataFrame:
    out = frame.copy()
    out["correction"] = correction
    out["feature_set"] = feature_set
    out["residual_pred"] = np.clip(residual, -RESIDUAL_CLIP, RESIDUAL_CLIP)
    out["pred_loss"] = out["base_pred_loss"] * np.exp(out["residual_pred"])
    out["method"] = "momentum_mpl_hybrid_decay"
    return out


def build_none_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["correction"] = "none"
    out["feature_set"] = "none"
    out["residual_pred"] = 0.0
    out["pred_loss"] = out["base_pred_loss"]
    out["method"] = "momentum_mpl_hybrid_decay"
    return out


def build_mean_shift_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    train = frame[frame["schedule"] == TRAIN_SCHEDULE]
    residual = np.log(train["loss"]) - np.log(np.maximum(train["base_pred_loss"], 1e-12))
    return apply_log_residual(
        frame,
        np.full(len(frame), float(residual.mean()), dtype=np.float64),
        "mean_shift",
        "constant_residual",
    )


def build_step_spline_reference(frame: pd.DataFrame) -> pd.DataFrame:
    train = frame[frame["schedule"] == TRAIN_SCHEDULE].sort_values("step")
    residual = np.log(train["loss"]) - np.log(np.maximum(train["base_pred_loss"], 1e-12))
    spline = UnivariateSpline(train["step"], residual, s=0.1)
    residual_pred = spline(frame["step"].to_numpy(dtype=np.float64))
    return apply_log_residual(frame, residual_pred, "step_spline_reference", "spline_s0.1_shrink1")


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["method", "correction", "feature_set", "schedule"]
    windows = {
        "full": (None, None),
        "steps_20000_30000": (20000, 30000),
    }
    for group_values, group in predictions.groupby(group_cols, sort=False):
        group = group.sort_values("step")
        for window_name, (start, end) in windows.items():
            window = group
            if start is not None:
                window = window[window["step"] >= start]
            if end is not None:
                window = window[window["step"] <= end]
            actual = window["loss"].to_numpy(dtype=np.float64)
            pred = window["pred_loss"].to_numpy(dtype=np.float64)
            residual = actual - pred
            ss_res = float(np.sum(residual * residual))
            ss_tot = float(np.sum((actual - actual.mean()) ** 2))
            rows.append(
                {
                    **dict(zip(group_cols, group_values)),
                    "window": window_name,
                    "n": len(window),
                    "mae": float(np.mean(np.abs(residual))),
                    "rmse": float(np.sqrt(np.mean(residual * residual))),
                    "mape": float(np.mean(np.abs(residual / actual))),
                    "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
                    "max_abs_error": float(np.max(np.abs(residual))),
                    "endpoint_abs_diff": float(abs(pred[-1] - actual[-1])),
                    "residual_mean": float(np.mean(residual)),
                }
            )
    return pd.DataFrame(rows)


def run_trial(frame: pd.DataFrame, candidate: Candidate, scales: dict[str, float]) -> tuple[pd.DataFrame, dict[str, object]]:
    train_mask = frame["schedule"] == TRAIN_SCHEDULE
    x_all, names = build_feature_matrix(frame, candidate.feature_set, candidate.scale, candidate.beta, scales)
    x_train = x_all[train_mask.to_numpy()]
    y_train = (
        np.log(frame.loc[train_mask, "loss"].to_numpy(dtype=np.float64))
        - np.log(np.maximum(frame.loc[train_mask, "base_pred_loss"].to_numpy(dtype=np.float64), 1e-12))
    )
    coef = fit_ridge_residual(x_train, y_train, candidate.ridge_alpha)
    residual = predict_ridge_residual(x_all, coef) * candidate.shrink
    pred = apply_log_residual(
        frame,
        residual,
        "hybrid_decay",
        (
            f"{candidate.feature_set}_scale{candidate.scale:g}_beta{candidate.beta:g}"
            f"_ridge{candidate.ridge_alpha:g}_shrink{candidate.shrink:g}"
        ),
    )
    trial_info = {
        "feature_set_family": candidate.feature_set,
        "feature_columns": ";".join(names),
        "scale": candidate.scale,
        "beta": candidate.beta,
        "ridge_alpha": candidate.ridge_alpha,
        "shrink": candidate.shrink,
        "coef": ";".join(f"{value:.12g}" for value in coef),
    }
    return pred, trial_info


def select_best_trial(metrics: pd.DataFrame) -> pd.Series:
    validation = metrics[
        (metrics["correction"] == "hybrid_decay")
        & (metrics["schedule"] == VALIDATION_SCHEDULE)
        & (metrics["window"] == "full")
    ].copy()
    if validation.empty:
        raise RuntimeError("No validation rows found for hybrid trials")
    validation = validation.sort_values(["mae", "endpoint_abs_diff", "feature_set"])
    return validation.iloc[0]


def plot_key_model(predictions: pd.DataFrame, output_path: Path, key_feature_set: str, start: int | None = None, end: int | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schedules = [TRAIN_SCHEDULE, VALIDATION_SCHEDULE, TEST_SCHEDULE]
    fig, axes = plt.subplots(len(schedules), 1, figsize=(10.5, 8.5), sharex=True)
    for ax, schedule in zip(axes, schedules):
        base = predictions[(predictions["schedule"] == schedule) & (predictions["correction"] == "none")].sort_values("step")
        hybrid = predictions[
            (predictions["schedule"] == schedule)
            & (predictions["correction"] == "hybrid_decay")
            & (predictions["feature_set"] == key_feature_set)
        ].sort_values("step")
        spline = predictions[
            (predictions["schedule"] == schedule)
            & (predictions["correction"] == "step_spline_reference")
        ].sort_values("step")
        if start is not None:
            base = base[base["step"] >= start]
            hybrid = hybrid[hybrid["step"] >= start]
            spline = spline[spline["step"] >= start]
        if end is not None:
            base = base[base["step"] <= end]
            hybrid = hybrid[hybrid["step"] <= end]
            spline = spline[spline["step"] <= end]

        ax.plot(base["step"], base["loss"], color="#111111", linewidth=1.0, label="actual")
        ax.plot(base["step"], base["base_pred_loss"], color="#D55E00", linestyle="--", linewidth=1.2, label="momentum")
        ax.plot(hybrid["step"], hybrid["pred_loss"], color="#0072B2", linewidth=1.4, label="hybrid decay")
        ax.plot(spline["step"], spline["pred_loss"], color="#009E73", linewidth=1.1, alpha=0.8, label="step spline ref")
        ax.set_title(schedule)
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if start is not None or end is not None:
        axes[-1].set_xlim(start, end)
        fig.suptitle(f"Momentum-MPL hybrid decay, steps {start}-{end}")
    else:
        fig.suptitle("Momentum-MPL hybrid decay, sampled trajectory")
    axes[0].legend(frameon=False, ncol=4, fontsize=8)
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_decay_diagnostics(frame: pd.DataFrame, predictions: pd.DataFrame, output_path: Path, key_feature_set: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8), sharex=True)
    for schedule, group in frame.groupby("schedule", sort=False):
        group = group.sort_values("step")
        axes[0].plot(group["step"], group["s2"], label=schedule, linewidth=1.1)
        axes[1].plot(group["step"], group["drop_mass"], label=schedule, linewidth=1.1)

        key = predictions[
            (predictions["schedule"] == schedule)
            & (predictions["correction"] == "hybrid_decay")
            & (predictions["feature_set"] == key_feature_set)
        ].sort_values("step")
        axes[2].plot(key["step"], key["residual_pred"], label=schedule, linewidth=1.1)

    axes[0].set_ylabel("S2")
    axes[1].set_ylabel("drop mass")
    axes[2].set_ylabel("log residual correction")
    axes[2].set_xlabel("Step")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=3, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Hybrid decay geometry and selected residual correction")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    base = add_decay_geometry(load_three_schedule_momentum_predictions())
    scales = train_scales(base)

    candidates = [
        Candidate(feature_set, scale, beta, ridge_alpha, shrink)
        for feature_set in ["sat_s2", "sat_s2_drop", "sat_s2_tail", "mpl_proxy", "hybrid_decay_progress"]
        for scale in [0.5, 1.0, 2.0, 5.0]
        for beta in [0.5, 1.0, 2.0]
        for ridge_alpha in [1e-6, 1e-4, 1e-2, 1.0]
        for shrink in [0.25, 0.5, 0.75, 1.0, 1.25]
    ]

    reference_predictions = [
        build_none_prediction(base),
        build_mean_shift_prediction(base),
        build_step_spline_reference(base),
    ]

    trial_metrics_parts = []
    trial_rows = []
    best_pred: pd.DataFrame | None = None
    best_val_key: tuple[float, float, str] | None = None
    for candidate in candidates:
        pred, info = run_trial(base, candidate, scales)
        trial_metrics = evaluate_predictions(pred)
        trial_metrics_parts.append(trial_metrics)
        val_row = trial_metrics[
            (trial_metrics["schedule"] == VALIDATION_SCHEDULE)
            & (trial_metrics["window"] == "full")
        ].iloc[0]
        test_row = trial_metrics[
            (trial_metrics["schedule"] == TEST_SCHEDULE)
            & (trial_metrics["window"] == "full")
        ].iloc[0]
        trial_rows.append(
            {
                **info,
                "feature_set": pred["feature_set"].iloc[0],
                "validation_mae": float(val_row["mae"]),
                "validation_rmse": float(val_row["rmse"]),
                "validation_r2": float(val_row["r2"]),
                "validation_endpoint_abs_diff": float(val_row["endpoint_abs_diff"]),
                "test_mae": float(test_row["mae"]),
                "test_rmse": float(test_row["rmse"]),
                "test_r2": float(test_row["r2"]),
                "test_endpoint_abs_diff": float(test_row["endpoint_abs_diff"]),
            }
        )
        val_key = (float(val_row["mae"]), float(val_row["endpoint_abs_diff"]), str(pred["feature_set"].iloc[0]))
        if best_val_key is None or val_key < best_val_key:
            best_val_key = val_key
            best_pred = pred.copy()

    if best_pred is None:
        raise RuntimeError("No hybrid candidate produced predictions")

    reference_metrics = [evaluate_predictions(pred) for pred in reference_predictions]
    all_metrics = pd.concat(reference_metrics + trial_metrics_parts, ignore_index=True)
    trials = pd.DataFrame(trial_rows).sort_values(["validation_mae", "validation_endpoint_abs_diff", "feature_set"])
    best = select_best_trial(all_metrics)
    key_feature_set = str(best["feature_set"])

    if str(best_pred["feature_set"].iloc[0]) != key_feature_set:
        raise RuntimeError("Internal selection mismatch between metrics and cached prediction")

    compact_predictions = pd.concat(reference_predictions + [best_pred], ignore_index=True)
    compact_metrics = evaluate_predictions(compact_predictions)

    all_metrics.to_csv(OUTPUT_DIR / "momentum_mpl_hybrid_decay_all_metrics.csv", index=False)
    compact_metrics.to_csv(OUTPUT_DIR / "momentum_mpl_hybrid_decay_metrics.csv", index=False)
    trials.to_csv(OUTPUT_DIR / "momentum_mpl_hybrid_decay_trials.csv", index=False)
    compact_predictions[
        (compact_predictions["correction"] == "hybrid_decay")
        & (compact_predictions["feature_set"] == key_feature_set)
    ].to_csv(OUTPUT_DIR / "key_momentum_mpl_hybrid_decay_predictions.csv", index=False)

    plot_key_model(
        compact_predictions,
        FIGURE_DIR / "momentum_mpl_hybrid_decay_full.png",
        key_feature_set,
    )
    plot_key_model(
        compact_predictions,
        FIGURE_DIR / "momentum_mpl_hybrid_decay_20000_30000.png",
        key_feature_set,
        start=20000,
        end=30000,
    )
    plot_decay_diagnostics(
        base,
        compact_predictions,
        FIGURE_DIR / "momentum_mpl_hybrid_decay_diagnostics.png",
        key_feature_set,
    )

    report = compact_metrics[
        (compact_metrics["schedule"].isin([VALIDATION_SCHEDULE, TEST_SCHEDULE]))
        & (compact_metrics["window"].isin(["full", "steps_20000_30000"]))
    ].sort_values(["schedule", "window", "correction", "feature_set"])
    print(f"Selected hybrid feature_set: {key_feature_set}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
