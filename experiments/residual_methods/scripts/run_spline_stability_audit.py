from __future__ import annotations

import argparse
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

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


OUTPUT_DIR = PROJECT_DIR / "outputs"
PROJECT_ROOT = PROJECT_DIR.parents[1]
THREE_RUN_PREDICTIONS = PROJECT_ROOT / "results" / "intermediates" / "three_schedule_momentum" / "predictions.csv"

SCHEDULES = ["cosine", "811", "wsd"]
TRAIN_SCHEDULE = "cosine"
VALIDATION_SCHEDULE = "811"
TEST_SCHEDULE = "wsd"
START_STEP = 1000
END_STEP = 33906
SAMPLE_INTERVAL = 2
RNG_SEED = 20260608
SELECTED_CONFIG = None


@dataclass(frozen=True)
class SplineConfig:
    smoothing: float
    shrink: float
    residual_clip: float

    @property
    def name(self) -> str:
        return f"spline_s{self.smoothing:g}_shrink{self.shrink:g}_clip{self.residual_clip:g}"


SELECTED_CONFIG = SplineConfig(smoothing=0.01, shrink=1.0, residual_clip=0.15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit step-aligned residual spline transfer.")
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help=(
            "Regenerate only the 811-selected spline metrics used by the report "
            "verification. The full grid, LOSO, bootstrap, and placebo audit "
            "remain available by running this script without the flag."
        ),
    )
    return parser.parse_args()


def load_predictions() -> pd.DataFrame:
    frame = pd.read_csv(THREE_RUN_PREDICTIONS)
    frame = frame.rename(columns={"run": "schedule", "momentum_s2": "base_pred_loss"})
    frame = frame[
        ["schedule", "step", "loss", "lr", "s1", "s2", "base_pred_loss"]
    ].copy()
    frame = frame[
        (frame["step"] >= START_STEP)
        & (frame["step"] <= END_STEP)
        & (((frame["step"] - START_STEP) % SAMPLE_INTERVAL) == 0)
    ].copy()
    frame["log_residual"] = np.log(frame["loss"]) - np.log(np.maximum(frame["base_pred_loss"], 1e-12))
    return frame.sort_values(["schedule", "step"]).reset_index(drop=True)


def fit_step_spline(train: pd.DataFrame, config: SplineConfig) -> UnivariateSpline:
    # For multi-schedule training, average duplicated step labels into one target
    # so the spline remains a one-dimensional step template.
    grouped = (
        train.groupby("step", as_index=False)["log_residual"]
        .mean()
        .sort_values("step")
    )
    return UnivariateSpline(grouped["step"], grouped["log_residual"], s=config.smoothing)


def predict_from_spline(frame: pd.DataFrame, spline: UnivariateSpline, config: SplineConfig) -> pd.DataFrame:
    out = frame.copy()
    residual = spline(out["step"].to_numpy(dtype=np.float64)) * config.shrink
    out["residual_pred"] = np.clip(residual, -config.residual_clip, config.residual_clip)
    out["pred_loss"] = out["base_pred_loss"] * np.exp(out["residual_pred"])
    return out


def predict_mean_shift(train: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["residual_pred"] = float(train["log_residual"].mean())
    out["pred_loss"] = out["base_pred_loss"] * np.exp(out["residual_pred"])
    return out


def predict_none(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["residual_pred"] = 0.0
    out["pred_loss"] = out["base_pred_loss"]
    return out


def window_slices(group: pd.DataFrame) -> dict[str, pd.DataFrame]:
    group = group.sort_values("step")
    windows = {
        "full": group,
        "steps_1000_10000": group[(group["step"] >= 1000) & (group["step"] <= 10000)],
        "steps_10000_20000": group[(group["step"] >= 10000) & (group["step"] <= 20000)],
        "steps_20000_30000": group[(group["step"] >= 20000) & (group["step"] <= 30000)],
        "steps_20000_27125": group[(group["step"] >= 20000) & (group["step"] <= 27125)],
        "steps_27126_30000": group[(group["step"] >= 27126) & (group["step"] <= 30000)],
        "steps_27126_33906": group[(group["step"] >= 27126) & (group["step"] <= 33906)],
        "steps_30000_33906": group[(group["step"] >= 30000) & (group["step"] <= 33906)],
        "last_512_sampled": group.tail(512),
        "last_2048_sampled": group.tail(2048),
    }
    return {name: window for name, window in windows.items() if len(window) > 0}


def metric_dict(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = actual - pred
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "mape": float(np.mean(np.abs(residual / actual))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "signed_bias": float(np.mean(pred - actual)),
        "max_abs_error": float(np.max(np.abs(residual))),
        "endpoint_abs_diff": float(abs(pred[-1] - actual[-1])),
    }


def evaluate_prediction(
    pred: pd.DataFrame,
    model: str,
    train_schedule: str,
    target_schedule: str,
    config_name: str,
) -> pd.DataFrame:
    rows = []
    target = pred[pred["schedule"] == target_schedule].sort_values("step")
    for window_name, window in window_slices(target).items():
        rows.append(
            {
                "model": model,
                "train_schedule": train_schedule,
                "target_schedule": target_schedule,
                "config": config_name,
                "window": window_name,
                "n": len(window),
                **metric_dict(
                    window["loss"].to_numpy(dtype=np.float64),
                    window["pred_loss"].to_numpy(dtype=np.float64),
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_pair(
    base: pd.DataFrame,
    corrected: pd.DataFrame,
    model: str,
    train_schedule: str,
    target_schedule: str,
    config_name: str,
) -> pd.DataFrame:
    base_metrics = evaluate_prediction(base, "momentum_baseline", train_schedule, target_schedule, "none")
    corr_metrics = evaluate_prediction(corrected, model, train_schedule, target_schedule, config_name)
    joined = corr_metrics.merge(
        base_metrics,
        on=["train_schedule", "target_schedule", "window"],
        suffixes=("", "_baseline"),
    )
    joined["mae_improvement"] = joined["mae_baseline"] - joined["mae"]
    joined["rmse_improvement"] = joined["rmse_baseline"] - joined["rmse"]
    joined["relative_mae_improvement_pct"] = 100.0 * joined["mae_improvement"] / joined["mae_baseline"]
    return pd.concat([base_metrics, corr_metrics], ignore_index=True), joined


def frozen_parameter_grid(frame: pd.DataFrame) -> tuple[pd.DataFrame, SplineConfig]:
    # Keep this grid compact: it is an audit for the existing spline claim, not
    # a fresh hyperparameter search. The original key setting is included.
    configs = [
        SplineConfig(smoothing=s, shrink=shrink, residual_clip=clip)
        for s in [0.01, 0.05, 0.1, 0.25, 0.5]
        for shrink in [0.75, 1.0, 1.25]
        for clip in [0.15, 0.25]
    ]
    train = frame[frame["schedule"] == TRAIN_SCHEDULE]
    rows = []
    for config in configs:
        spline = fit_step_spline(train, config)
        pred = predict_from_spline(frame, spline, config)
        for schedule, split in [(VALIDATION_SCHEDULE, "validation"), (TEST_SCHEDULE, "test")]:
            metrics = evaluate_prediction(
                pred,
                "step_spline",
                TRAIN_SCHEDULE,
                schedule,
                config.name,
            )
            full = metrics[metrics["window"] == "full"].iloc[0]
            hard = metrics[metrics["window"] == "steps_20000_30000"].iloc[0]
            rows.append(
                {
                    "split": split,
                    "schedule": schedule,
                    "config": config.name,
                    "smoothing": config.smoothing,
                    "shrink": config.shrink,
                    "residual_clip": config.residual_clip,
                    "full_mae": float(full["mae"]),
                    "full_rmse": float(full["rmse"]),
                    "full_r2": float(full["r2"]),
                    "full_endpoint_abs_diff": float(full["endpoint_abs_diff"]),
                    "hard_mae": float(hard["mae"]),
                    "hard_rmse": float(hard["rmse"]),
                    "hard_r2": float(hard["r2"]),
                    "hard_endpoint_abs_diff": float(hard["endpoint_abs_diff"]),
                }
            )
    grid = pd.DataFrame(rows)
    validation = grid[grid["split"] == "validation"].sort_values(
        ["full_mae", "hard_mae", "full_endpoint_abs_diff", "config"]
    )
    best_row = validation.iloc[0]
    best = SplineConfig(
        smoothing=float(best_row["smoothing"]),
        shrink=float(best_row["shrink"]),
        residual_clip=float(best_row["residual_clip"]),
    )
    return grid, best


def selected_model_metrics(frame: pd.DataFrame, config: SplineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame[frame["schedule"] == TRAIN_SCHEDULE]
    spline = fit_step_spline(train, config)
    base = predict_none(frame)
    mean = predict_mean_shift(train, frame)
    corrected = predict_from_spline(frame, spline, config)

    metrics = []
    improvements = []
    for schedule in [VALIDATION_SCHEDULE, TEST_SCHEDULE]:
        for pred, model_name, cfg in [
            (base, "momentum_baseline", "none"),
            (mean, "mean_shift", "constant_residual"),
            (corrected, "step_spline_811_selected", config.name),
        ]:
            metrics.append(evaluate_prediction(pred, model_name, TRAIN_SCHEDULE, schedule, cfg))
        _, joined = evaluate_pair(base, corrected, "step_spline_811_selected", TRAIN_SCHEDULE, schedule, config.name)
        improvements.append(joined)
    return pd.concat(metrics, ignore_index=True), pd.concat(improvements, ignore_index=True)


def loso_transfer_matrix(frame: pd.DataFrame, config: SplineConfig) -> pd.DataFrame:
    rows = []
    base = predict_none(frame)
    for train_schedule in SCHEDULES:
        train = frame[frame["schedule"] == train_schedule]
        spline = fit_step_spline(train, config)
        corrected = predict_from_spline(frame, spline, config)
        for target in SCHEDULES:
            if target == train_schedule:
                continue
            _, joined = evaluate_pair(
                base,
                corrected,
                "single_source_step_spline",
                train_schedule,
                target,
                config.name,
            )
            rows.append(joined)

    for held_out in SCHEDULES:
        train_schedules = [schedule for schedule in SCHEDULES if schedule != held_out]
        train = frame[frame["schedule"].isin(train_schedules)]
        spline = fit_step_spline(train, config)
        corrected = predict_from_spline(frame, spline, config)
        _, joined = evaluate_pair(
            base,
            corrected,
            "two_source_loso_step_spline",
            "+".join(train_schedules),
            held_out,
            config.name,
        )
        rows.append(joined)
    return pd.concat(rows, ignore_index=True)


def block_bootstrap_summary(
    frame: pd.DataFrame,
    config: SplineConfig,
    n_replicates: int = 200,
) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    train = frame[frame["schedule"] == TRAIN_SCHEDULE]
    spline = fit_step_spline(train, config)
    pred = predict_from_spline(frame, spline, config)
    target = pred[pred["schedule"] == TEST_SCHEDULE].sort_values("step").reset_index(drop=True)
    base_abs = np.abs(target["loss"].to_numpy(dtype=np.float64) - target["base_pred_loss"].to_numpy(dtype=np.float64))
    corr_abs = np.abs(target["loss"].to_numpy(dtype=np.float64) - target["pred_loss"].to_numpy(dtype=np.float64))
    diff = base_abs - corr_abs

    rows = []
    audit_windows = {"full", "steps_20000_30000", "steps_27126_33906", "last_2048_sampled"}
    for window_name, window in window_slices(target).items():
        if window_name not in audit_windows:
            continue
        indices = window.index.to_numpy(dtype=np.int64)
        window_diff = diff[indices]
        if len(window_diff) == 0:
            continue
        for block_size in [128, 512, 2048]:
            block_size = min(block_size, len(window_diff))
            boot = []
            for _ in range(n_replicates):
                pieces = []
                while sum(len(piece) for piece in pieces) < len(window_diff):
                    start = int(rng.integers(0, max(len(window_diff) - block_size + 1, 1)))
                    pieces.append(window_diff[start : start + block_size])
                sample = np.concatenate(pieces)[: len(window_diff)]
                boot.append(float(np.mean(sample)))
            boot_arr = np.asarray(boot, dtype=np.float64)
            rows.append(
                {
                    "target_schedule": TEST_SCHEDULE,
                    "window": window_name,
                    "n": len(window_diff),
                    "block_size": block_size,
                    "n_replicates": n_replicates,
                    "mean_abs_error_improvement": float(np.mean(window_diff)),
                    "q05": float(np.quantile(boot_arr, 0.05)),
                    "q50": float(np.quantile(boot_arr, 0.50)),
                    "q95": float(np.quantile(boot_arr, 0.95)),
                    "prob_positive": float(np.mean(boot_arr > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def residual_controls(frame: pd.DataFrame, config: SplineConfig) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    train = frame[frame["schedule"] == TRAIN_SCHEDULE].sort_values("step")
    spline = fit_step_spline(train, config)
    base_residual = np.clip(
        spline(train["step"].to_numpy(dtype=np.float64)) * config.shrink,
        -config.residual_clip,
        config.residual_clip,
    )

    def block_permute(values: np.ndarray, block_size: int) -> np.ndarray:
        blocks = [values[start : start + block_size] for start in range(0, len(values), block_size)]
        order = rng.permutation(len(blocks))
        return np.concatenate([blocks[idx] for idx in order])[: len(values)]

    controls = {
        "true_step_spline": base_residual,
        "circular_shift_25pct": np.roll(base_residual, len(base_residual) // 4),
        "reverse_time": base_residual[::-1],
        "sign_flip": -base_residual,
        "block_permute_512": block_permute(base_residual, 512),
    }

    rows = []
    target = frame[frame["schedule"] == TEST_SCHEDULE].sort_values("step").copy()
    train_steps = train["step"].to_numpy(dtype=np.float64)
    target_steps = target["step"].to_numpy(dtype=np.float64)
    base = predict_none(frame)
    for name, residual_values in controls.items():
        target_pred = target.copy()
        residual = np.interp(target_steps, train_steps, residual_values)
        target_pred["residual_pred"] = residual
        target_pred["pred_loss"] = target_pred["base_pred_loss"] * np.exp(target_pred["residual_pred"])
        pseudo = pd.concat(
            [
                frame[frame["schedule"] != TEST_SCHEDULE].assign(
                    residual_pred=0.0,
                    pred_loss=lambda x: x["base_pred_loss"],
                ),
                target_pred,
            ],
            ignore_index=True,
        )
        _, joined = evaluate_pair(base, pseudo, name, TRAIN_SCHEDULE, TEST_SCHEDULE, config.name)
        rows.append(joined)
    return pd.concat(rows, ignore_index=True)


def run_selected_only(frame: pd.DataFrame) -> None:
    selected_metrics, selected_improvements = selected_model_metrics(frame, SELECTED_CONFIG)
    selected_metrics.to_csv(OUTPUT_DIR / "spline_stability_selected_metrics.csv", index=False)
    selected_improvements.to_csv(OUTPUT_DIR / "spline_stability_selected_improvements.csv", index=False)

    print(f"Fixed 811-selected config: {SELECTED_CONFIG.name}")
    print("\nSelected WSD metrics:")
    report = selected_metrics[
        (selected_metrics["target_schedule"] == TEST_SCHEDULE)
        & (selected_metrics["window"].isin(["full", "steps_20000_30000", "steps_27126_33906", "last_2048_sampled"]))
    ].sort_values(["window", "model"])
    print(report.to_string(index=False))


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_predictions()

    if args.selected_only:
        run_selected_only(frame)
        return

    param_grid, selected = frozen_parameter_grid(frame)
    selected_metrics, selected_improvements = selected_model_metrics(frame, selected)
    loso = loso_transfer_matrix(frame, selected)
    bootstrap = block_bootstrap_summary(frame, selected)
    controls = residual_controls(frame, selected)

    param_grid.to_csv(OUTPUT_DIR / "spline_stability_parameter_grid.csv", index=False)
    selected_metrics.to_csv(OUTPUT_DIR / "spline_stability_selected_metrics.csv", index=False)
    selected_improvements.to_csv(OUTPUT_DIR / "spline_stability_selected_improvements.csv", index=False)
    loso.to_csv(OUTPUT_DIR / "spline_stability_loso_transfer.csv", index=False)
    bootstrap.to_csv(OUTPUT_DIR / "spline_stability_block_bootstrap.csv", index=False)
    controls.to_csv(OUTPUT_DIR / "spline_stability_negative_controls.csv", index=False)

    print(f"811-selected config: {selected.name}")
    print("\nSelected WSD metrics:")
    report = selected_metrics[
        (selected_metrics["target_schedule"] == TEST_SCHEDULE)
        & (selected_metrics["window"].isin(["full", "steps_20000_30000", "steps_27126_33906", "last_2048_sampled"]))
    ].sort_values(["window", "model"])
    print(report.to_string(index=False))
    print("\nWSD block bootstrap summary:")
    boot_report = bootstrap[
        bootstrap["window"].isin(["full", "steps_20000_30000", "steps_27126_33906", "last_2048_sampled"])
    ].sort_values(["window", "block_size"])
    print(boot_report.to_string(index=False))
    print("\nNegative controls, WSD full:")
    control_report = controls[controls["window"] == "full"].sort_values("mae")
    print(control_report[["model", "mae_baseline", "mae", "mae_improvement", "relative_mae_improvement_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
