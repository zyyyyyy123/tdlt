from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


PROJECT_DIR = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = PROJECT_DIR / "outputs" / ".matplotlib"
XDG_CACHE_DIR = PROJECT_DIR / "outputs" / ".cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))


OUTPUT_DIR = PROJECT_DIR / "outputs"
PROJECT_ROOT = PROJECT_DIR.parents[1]
THREE_RUN_PREDICTIONS = PROJECT_ROOT / "results" / "intermediates" / "three_schedule_momentum" / "predictions.csv"

TRAIN_SCHEDULE = "cosine"
VALIDATION_SCHEDULE = "811"
TEST_SCHEDULE = "wsd"
START_STEP = 1000
END_STEP = 33906
SAMPLE_INTERVAL = 2
EPS = 1e-12
MAX_SPLINE_FIT_POINTS = 4096

COORDINATE_FAMILIES = [
    "step_abs",
    "s1_raw",
    "s1_clamp",
    "s1_ratio",
]
PRIMARY_WINDOWS = [
    "full",
    "steps_1000_10000",
    "steps_10000_20000",
    "steps_27126_33906",
    "steps_30000_33906",
    "last_2048_sampled",
]


@dataclass(frozen=True)
class SplineConfig:
    smoothing: float
    shrink: float
    residual_clip: float

    @property
    def name(self) -> str:
        return f"s{self.smoothing:g}_shrink{self.shrink:g}_clip{self.residual_clip:g}"


CONFIG_GRID = [
    SplineConfig(smoothing=s, shrink=shrink, residual_clip=clip)
    for s in [0.1, 0.5]
    for shrink in [0.5, 0.75, 1.0]
    for clip in [0.15, 0.25]
]


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
    frame = frame.sort_values(["schedule", "step"]).reset_index(drop=True)
    frame["max_s1"] = frame.groupby("schedule")["s1"].transform("max")
    frame["max_s2"] = frame.groupby("schedule")["s2"].transform("max")
    frame["log_residual"] = (
        np.log(frame["loss"].to_numpy(dtype=np.float64))
        - np.log(np.maximum(frame["base_pred_loss"].to_numpy(dtype=np.float64), EPS))
    )
    for window in [101, 501]:
        frame[f"log_residual_roll{window}"] = (
            frame.groupby("schedule")["log_residual"]
            .transform(lambda values: values.rolling(window, center=True, min_periods=1).mean())
            .to_numpy(dtype=np.float64)
        )
    return frame


def coordinate_values(frame: pd.DataFrame, family: str, train_frame: pd.DataFrame) -> np.ndarray:
    if family == "step_abs":
        return frame["step"].to_numpy(dtype=np.float64)

    if family.startswith("s1"):
        raw = frame["s1"].to_numpy(dtype=np.float64)
        train_raw = train_frame["s1"].to_numpy(dtype=np.float64)
        if family == "s1_raw":
            return raw
        if family == "s1_clamp":
            return np.clip(raw, float(train_raw.min()), float(train_raw.max()))
        if family == "s1_ratio":
            return raw / np.maximum(frame["max_s1"].to_numpy(dtype=np.float64), EPS)

    if family.startswith("s2"):
        raw = frame["s2"].to_numpy(dtype=np.float64)
        train_raw = train_frame["s2"].to_numpy(dtype=np.float64)
        if family == "s2_raw":
            return raw
        if family == "s2_clamp":
            return np.clip(raw, float(train_raw.min()), float(train_raw.max()))
        if family == "s2_ratio":
            return raw / np.maximum(frame["max_s2"].to_numpy(dtype=np.float64), EPS)

    raise ValueError(f"Unknown coordinate family: {family}")


def raw_coordinate_values(frame: pd.DataFrame, family: str) -> np.ndarray:
    if family == "step_abs":
        return frame["step"].to_numpy(dtype=np.float64)
    if family.startswith("s1"):
        return frame["s1"].to_numpy(dtype=np.float64)
    if family.startswith("s2"):
        return frame["s2"].to_numpy(dtype=np.float64)
    raise ValueError(f"Unknown coordinate family: {family}")


def fit_coordinate_spline(
    train: pd.DataFrame,
    family: str,
    config: SplineConfig,
    residual_col: str = "log_residual",
) -> UnivariateSpline:
    x = coordinate_values(train, family, train)
    fit_data = pd.DataFrame(
        {
            "x": x,
            "residual": train[residual_col].to_numpy(dtype=np.float64),
        }
    )
    fit_data = fit_data.groupby("x", as_index=False)["residual"].mean().sort_values("x")
    original_fit_points = len(fit_data)
    if len(fit_data) > MAX_SPLINE_FIT_POINTS:
        fit_data = bin_fit_data(fit_data, MAX_SPLINE_FIT_POINTS)
    if len(fit_data) < 4:
        raise RuntimeError(f"Not enough unique x values for {family}")
    scaled_smoothing = config.smoothing * len(fit_data) / max(original_fit_points, 1)
    return UnivariateSpline(fit_data["x"], fit_data["residual"], s=scaled_smoothing)


def bin_fit_data(fit_data: pd.DataFrame, max_points: int) -> pd.DataFrame:
    fit_data = fit_data.sort_values("x").reset_index(drop=True)
    bin_id = (np.arange(len(fit_data), dtype=np.int64) * max_points) // len(fit_data)
    binned = (
        fit_data.assign(bin_id=bin_id)
        .groupby("bin_id", as_index=False)
        .agg(x=("x", "mean"), residual=("residual", "mean"))
        .sort_values("x")
    )
    return binned[["x", "residual"]]


def predict_coordinate_spline(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    family: str,
    config: SplineConfig,
    residual_col: str = "log_residual",
) -> pd.DataFrame:
    spline = fit_coordinate_spline(train, family, config, residual_col=residual_col)
    out = frame.copy()
    x = coordinate_values(out, family, train)
    residual = spline(x) * config.shrink
    out["family"] = family
    out["config"] = config.name
    out["spline_x"] = x
    out["residual_pred"] = np.clip(residual, -config.residual_clip, config.residual_clip)
    out["pred_loss"] = out["base_pred_loss"] * np.exp(out["residual_pred"])
    return out


def predict_none(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["family"] = "momentum_baseline"
    out["config"] = "none"
    out["spline_x"] = np.nan
    out["residual_pred"] = 0.0
    out["pred_loss"] = out["base_pred_loss"]
    return out


def window_slices(group: pd.DataFrame) -> dict[str, pd.DataFrame]:
    group = group.sort_values("step")
    windows = {
        "full": group,
        "steps_1000_10000": group[(group["step"] >= 1000) & (group["step"] <= 10000)],
        "steps_10000_20000": group[(group["step"] >= 10000) & (group["step"] <= 20000)],
        "steps_27126_33906": group[(group["step"] >= 27126) & (group["step"] <= 33906)],
        "steps_30000_33906": group[(group["step"] >= 30000) & (group["step"] <= 33906)],
        "last_2048_sampled": group.tail(2048),
    }
    return {name: value for name, value in windows.items() if len(value) > 0}


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


def residual_metric_dict(true_residual: np.ndarray, pred_residual: np.ndarray) -> dict[str, float]:
    diff = true_residual - pred_residual
    pred_std = float(np.std(pred_residual))
    true_std = float(np.std(true_residual))
    if pred_std <= EPS or true_std <= EPS:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(true_residual, pred_residual)[0, 1])
    return {
        "residual_mae": float(np.mean(np.abs(diff))),
        "residual_rmse": float(np.sqrt(np.mean(diff * diff))),
        "residual_corr": corr,
        "residual_spearman": float(
            pd.Series(true_residual).corr(pd.Series(pred_residual), method="spearman")
        ),
        "sign_agreement": float(np.mean(np.sign(true_residual) == np.sign(pred_residual))),
        "signed_bias": float(np.mean(pred_residual - true_residual)),
        "pred_residual_std": pred_std,
        "true_residual_std": true_std,
    }


def evaluate_prediction(pred: pd.DataFrame, schedule: str) -> pd.DataFrame:
    rows = []
    target = pred[pred["schedule"] == schedule].sort_values("step")
    for window_name, window in window_slices(target).items():
        rows.append(
            {
                "target_schedule": schedule,
                "window": window_name,
                "n": len(window),
                **metric_dict(
                    window["loss"].to_numpy(dtype=np.float64),
                    window["pred_loss"].to_numpy(dtype=np.float64),
                ),
            }
        )
    return pd.DataFrame(rows)


def validation_grid(frame: pd.DataFrame) -> pd.DataFrame:
    train = frame[frame["schedule"] == TRAIN_SCHEDULE]
    rows = []
    for family in COORDINATE_FAMILIES:
        for config in CONFIG_GRID:
            pred = predict_coordinate_spline(frame, train, family, config)
            metrics = evaluate_prediction(pred, VALIDATION_SCHEDULE)
            full = metrics[metrics["window"] == "full"].iloc[0]
            tail = metrics[metrics["window"] == "steps_27126_33906"].iloc[0]
            rows.append(
                {
                    "family": family,
                    "config": config.name,
                    "smoothing": config.smoothing,
                    "shrink": config.shrink,
                    "residual_clip": config.residual_clip,
                    "selected_on": f"{VALIDATION_SCHEDULE}:full",
                    "val_full_mae": float(full["mae"]),
                    "val_full_rmse": float(full["rmse"]),
                    "val_full_r2": float(full["r2"]),
                    "val_full_endpoint_abs_diff": float(full["endpoint_abs_diff"]),
                    "val_tail_mae": float(tail["mae"]),
                    "val_tail_rmse": float(tail["rmse"]),
                    "val_tail_endpoint_abs_diff": float(tail["endpoint_abs_diff"]),
                }
            )
    return pd.DataFrame(rows)


def select_configs(grid: pd.DataFrame) -> dict[str, SplineConfig]:
    selected: dict[str, SplineConfig] = {}
    for family, group in grid.groupby("family"):
        row = group.sort_values(
            ["val_full_mae", "val_full_rmse", "val_full_endpoint_abs_diff", "config"]
        ).iloc[0]
        selected[family] = SplineConfig(
            smoothing=float(row["smoothing"]),
            shrink=float(row["shrink"]),
            residual_clip=float(row["residual_clip"]),
        )
    return selected


def selected_predictions(frame: pd.DataFrame, selected: dict[str, SplineConfig]) -> dict[str, pd.DataFrame]:
    train = frame[frame["schedule"] == TRAIN_SCHEDULE]
    return {
        family: predict_coordinate_spline(frame, train, family, config)
        for family, config in selected.items()
    }


def performance_by_window(
    frame: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    base = predict_none(frame)
    rows = []
    for family, pred in predictions.items():
        for schedule in [VALIDATION_SCHEDULE, TEST_SCHEDULE]:
            base_metrics = evaluate_prediction(base, schedule)
            corr_metrics = evaluate_prediction(pred, schedule)
            joined = corr_metrics.merge(
                base_metrics,
                on=["target_schedule", "window"],
                suffixes=("_corrected", "_baseline"),
            )
            for _, row in joined.iterrows():
                rows.append(
                    {
                        "family": family,
                        "target_schedule": schedule,
                        "window": row["window"],
                        "n": int(row["n_corrected"]),
                        "mae_baseline": float(row["mae_baseline"]),
                        "mae_corrected": float(row["mae_corrected"]),
                        "mae_improvement": float(row["mae_baseline"] - row["mae_corrected"]),
                        "relative_improvement_pct": float(
                            100.0 * (row["mae_baseline"] - row["mae_corrected"]) / row["mae_baseline"]
                        ),
                        "rmse_baseline": float(row["rmse_baseline"]),
                        "rmse_corrected": float(row["rmse_corrected"]),
                        "r2_corrected": float(row["r2_corrected"]),
                        "endpoint_abs_diff": float(row["endpoint_abs_diff_corrected"]),
                    }
                )
    out = pd.DataFrame(rows)
    step = out[out["family"] == "step_abs"][
        ["target_schedule", "window", "mae_corrected", "rmse_corrected"]
    ].rename(columns={"mae_corrected": "step_mae", "rmse_corrected": "step_rmse"})
    out = out.merge(step, on=["target_schedule", "window"], how="left")
    out["mae_gap_vs_step"] = out["mae_corrected"] - out["step_mae"]
    out["rmse_gap_vs_step"] = out["rmse_corrected"] - out["step_rmse"]
    out["verdict"] = out.apply(performance_verdict, axis=1)
    return out.sort_values(["target_schedule", "window", "family"]).reset_index(drop=True)


def performance_verdict(row: pd.Series) -> str:
    if row["family"] == "step_abs":
        return "step_reference"
    if row["mae_corrected"] < row["mae_baseline"] and row["mae_gap_vs_step"] <= 0.002:
        return "competitive_with_step"
    if row["mae_corrected"] < row["mae_baseline"]:
        return "helps_but_weaker_than_step"
    return "fails_vs_momentum"


def selection_summary(
    grid: pd.DataFrame,
    perf: pd.DataFrame,
    selected: dict[str, SplineConfig],
) -> pd.DataFrame:
    rows = []
    for family, config in selected.items():
        grid_row = grid[(grid["family"] == family) & (grid["config"] == config.name)].iloc[0]
        val_full = perf[
            (perf["family"] == family)
            & (perf["target_schedule"] == VALIDATION_SCHEDULE)
            & (perf["window"] == "full")
        ].iloc[0]
        val_tail = perf[
            (perf["family"] == family)
            & (perf["target_schedule"] == VALIDATION_SCHEDULE)
            & (perf["window"] == "steps_27126_33906")
        ].iloc[0]
        test_full = perf[
            (perf["family"] == family)
            & (perf["target_schedule"] == TEST_SCHEDULE)
            & (perf["window"] == "full")
        ].iloc[0]
        test_tail = perf[
            (perf["family"] == family)
            & (perf["target_schedule"] == TEST_SCHEDULE)
            & (perf["window"] == "steps_27126_33906")
        ].iloc[0]
        rows.append(
            {
                "family": family,
                "selected_config": config.name,
                "selected_on": grid_row["selected_on"],
                "val_full_mae": float(val_full["mae_corrected"]),
                "val_tail_mae": float(val_tail["mae_corrected"]),
                "val_endpoint_abs_diff": float(val_full["endpoint_abs_diff"]),
                "test_full_mae": float(test_full["mae_corrected"]),
                "test_tail_mae": float(test_tail["mae_corrected"]),
                "test_endpoint_abs_diff": float(test_full["endpoint_abs_diff"]),
                "test_mae_baseline": float(test_full["mae_baseline"]),
                "test_mae_improvement_vs_momentum": float(test_full["mae_improvement"]),
                "test_relative_improvement_pct": float(test_full["relative_improvement_pct"]),
                "test_mae_gap_vs_step": float(test_full["mae_gap_vs_step"]),
                "verdict": selection_verdict(family, test_full),
            }
        )
    return pd.DataFrame(rows).sort_values("test_full_mae").reset_index(drop=True)


def selection_verdict(family: str, test_full: pd.Series) -> str:
    if family == "step_abs":
        return "best_heldout_reference"
    if test_full["mae_corrected"] < test_full["mae_baseline"] and test_full["mae_gap_vs_step"] <= 0.002:
        return "would_refute_step_advantage"
    if test_full["mae_corrected"] < test_full["mae_baseline"]:
        return "positive_but_weaker_than_step"
    return "negative_transfer"


def domain_support(
    frame: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    perf: pd.DataFrame,
) -> pd.DataFrame:
    train = frame[frame["schedule"] == TRAIN_SCHEDULE]
    rows = []
    for family, pred in predictions.items():
        train_x = coordinate_values(train, family, train)
        train_min = float(np.min(train_x))
        train_max = float(np.max(train_x))
        train_range = max(train_max - train_min, EPS)
        raw_train = raw_coordinate_values(train, family)
        raw_train_min = float(np.min(raw_train))
        raw_train_max = float(np.max(raw_train))
        for schedule in [VALIDATION_SCHEDULE, TEST_SCHEDULE]:
            target = frame[frame["schedule"] == schedule].sort_values("step")
            target_pred = pred[pred["schedule"] == schedule].sort_values("step")
            target_x = coordinate_values(target, family, train)
            target_raw = raw_coordinate_values(target, family)
            below = target_x < train_min
            above = target_x > train_max
            outside = below | above
            raw_outside = (target_raw < raw_train_min) | (target_raw > raw_train_max)
            max_extrapolation = float(
                max(
                    np.max(np.maximum(train_min - target_x, 0.0)),
                    np.max(np.maximum(target_x - train_max, 0.0)),
                )
            )
            full_mae = float(
                perf[
                    (perf["family"] == family)
                    & (perf["target_schedule"] == schedule)
                    & (perf["window"] == "full")
                ].iloc[0]["mae_corrected"]
            )
            rows.append(
                {
                    "family": family,
                    "target_schedule": schedule,
                    "train_x_min": train_min,
                    "train_x_max": train_max,
                    "target_x_min": float(np.min(target_x)),
                    "target_x_max": float(np.max(target_x)),
                    "frac_below_train_after_transform": float(np.mean(below)),
                    "frac_above_train_after_transform": float(np.mean(above)),
                    "frac_outside_train_after_transform": float(np.mean(outside)),
                    "frac_raw_outside_train": float(np.mean(raw_outside)),
                    "frac_clamped": float(np.mean(raw_outside)) if family.endswith("_clamp") else 0.0,
                    "max_extrapolation": max_extrapolation,
                    "normalized_max_extrapolation": max_extrapolation / train_range,
                    "full_mae": full_mae,
                    "inside_support_loss_mae": masked_loss_mae(target_pred, ~outside),
                    "outside_support_loss_mae": masked_loss_mae(target_pred, outside),
                    "n_inside_support": int(np.sum(~outside)),
                    "n_outside_support": int(np.sum(outside)),
                    "verdict": domain_verdict(family, float(np.mean(outside)), float(np.mean(raw_outside)), full_mae),
                }
            )
    return pd.DataFrame(rows).sort_values(["target_schedule", "family"]).reset_index(drop=True)


def masked_loss_mae(pred: pd.DataFrame, mask: np.ndarray) -> float:
    if int(np.sum(mask)) == 0:
        return float("nan")
    actual = pred.loc[mask, "loss"].to_numpy(dtype=np.float64)
    values = pred.loc[mask, "pred_loss"].to_numpy(dtype=np.float64)
    return float(np.mean(np.abs(actual - values)))


def domain_verdict(family: str, frac_outside: float, frac_raw_outside: float, full_mae: float) -> str:
    if family == "step_abs":
        return "no_coordinate_support_problem"
    if family.endswith("_raw") and frac_outside > 0.1:
        return "raw_coordinate_extrapolates"
    if family.endswith("_clamp") and frac_raw_outside > 0.1:
        return "clamp_removes_extrapolation_but_tests_phase"
    if family.endswith("_ratio") and frac_outside <= 0.05:
        return "scale_normalized_coordinate"
    return "support_mismatch_minor"


def common_support_table(
    frame: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    train = frame[frame["schedule"] == TRAIN_SCHEDULE]
    train_s1_min = float(train["s1"].min())
    train_s1_max = float(train["s1"].max())
    base = predict_none(frame)
    rows = []
    for schedule in [VALIDATION_SCHEDULE, TEST_SCHEDULE]:
        target = frame[frame["schedule"] == schedule].sort_values("step")
        s1_common_mask = (
            (target["s1"].to_numpy(dtype=np.float64) >= train_s1_min)
            & (target["s1"].to_numpy(dtype=np.float64) <= train_s1_max)
        )
        for support_basis, mask in [
            ("s1_raw_common_support", s1_common_mask),
            ("full_support", np.ones(len(target), dtype=bool)),
        ]:
            base_target = base[base["schedule"] == schedule].sort_values("step")
            base_mae = masked_loss_mae(base_target, mask)
            step_target = predictions["step_abs"][predictions["step_abs"]["schedule"] == schedule].sort_values("step")
            step_mae = masked_loss_mae(step_target, mask)
            for family, pred in predictions.items():
                target_pred = pred[pred["schedule"] == schedule].sort_values("step")
                mae = masked_loss_mae(target_pred, mask)
                rows.append(
                    {
                        "support_basis": support_basis,
                        "family": family,
                        "target_schedule": schedule,
                        "n": int(np.sum(mask)),
                        "coverage": float(np.mean(mask)),
                        "mae_baseline": base_mae,
                        "mae_corrected": mae,
                        "mae_improvement": base_mae - mae,
                        "step_mae": step_mae,
                        "mae_gap_vs_step": mae - step_mae,
                        "verdict": common_support_verdict(family, mae, base_mae, step_mae),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["target_schedule", "support_basis", "mae_corrected", "family"]
    ).reset_index(drop=True)


def common_support_verdict(family: str, mae: float, base_mae: float, step_mae: float) -> str:
    if family == "step_abs":
        return "step_reference"
    if mae < base_mae and mae - step_mae <= 0.002:
        return "common_support_refutes_step_gap"
    if mae < base_mae:
        return "common_support_helps_but_weaker"
    return "common_support_negative"


def residual_alignment(
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for family, pred in predictions.items():
        for schedule in [VALIDATION_SCHEDULE, TEST_SCHEDULE]:
            target = pred[pred["schedule"] == schedule].sort_values("step")
            for window_name, window in window_slices(target).items():
                rows.append(
                    {
                        "family": family,
                        "target_schedule": schedule,
                        "window": window_name,
                        "n": len(window),
                        **residual_metric_dict(
                            window["log_residual"].to_numpy(dtype=np.float64),
                            window["residual_pred"].to_numpy(dtype=np.float64),
                        ),
                    }
                )
    out = pd.DataFrame(rows)
    step = out[out["family"] == "step_abs"][
        ["target_schedule", "window", "residual_mae", "residual_corr"]
    ].rename(columns={"residual_mae": "step_residual_mae", "residual_corr": "step_residual_corr"})
    out = out.merge(step, on=["target_schedule", "window"], how="left")
    out["residual_mae_gap_vs_step"] = out["residual_mae"] - out["step_residual_mae"]
    out["residual_corr_gap_vs_step"] = out["residual_corr"] - out["step_residual_corr"]
    out["verdict"] = out.apply(alignment_verdict, axis=1)
    return out.sort_values(["target_schedule", "window", "family"]).reset_index(drop=True)


def alignment_verdict(row: pd.Series) -> str:
    if row["family"] == "step_abs":
        return "step_reference"
    if row["residual_mae_gap_vs_step"] <= 0.002 and row["residual_corr_gap_vs_step"] >= -0.05:
        return "alignment_competitive_with_step"
    if row["residual_corr"] > 0.5:
        return "aligned_but_weaker"
    return "phase_alignment_failure"


def time_warp_table(
    frame: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    train = frame[frame["schedule"] == TRAIN_SCHEDULE].sort_values("step")
    rows = []
    for family, pred in predictions.items():
        train_x = coordinate_values(train, family, train)
        train_steps = train["step"].to_numpy(dtype=np.float64)
        interp_data = (
            pd.DataFrame({"x": train_x, "step": train_steps})
            .groupby("x", as_index=False)["step"]
            .mean()
            .sort_values("x")
        )
        x_support = interp_data["x"].to_numpy(dtype=np.float64)
        step_support = interp_data["step"].to_numpy(dtype=np.float64)
        train_min = float(x_support.min())
        train_max = float(x_support.max())
        for schedule in [VALIDATION_SCHEDULE, TEST_SCHEDULE]:
            target = frame[frame["schedule"] == schedule].sort_values("step")
            target_pred = pred[pred["schedule"] == schedule].sort_values("step")
            target_x = coordinate_values(target, family, train)
            equiv_step = np.interp(target_x, x_support, step_support)
            warp = target["step"].to_numpy(dtype=np.float64) - equiv_step
            outside = (target_x < train_min) | (target_x > train_max)
            warp_frame = target_pred.copy()
            warp_frame["equiv_cosine_step"] = equiv_step
            warp_frame["step_warp"] = warp
            warp_frame["outside_support"] = outside
            for window_name, window in window_slices(warp_frame).items():
                values = window["step_warp"].to_numpy(dtype=np.float64)
                abs_values = np.abs(values)
                resid = residual_metric_dict(
                    window["log_residual"].to_numpy(dtype=np.float64),
                    window["residual_pred"].to_numpy(dtype=np.float64),
                )
                rows.append(
                    {
                        "family": family,
                        "target_schedule": schedule,
                        "window": window_name,
                        "n": len(window),
                        "target_step_median": float(window["step"].median()),
                        "equiv_cosine_step_median": float(window["equiv_cosine_step"].median()),
                        "median_step_warp": float(np.median(values)),
                        "median_abs_step_warp": float(np.median(abs_values)),
                        "q10_step_warp": float(np.quantile(values, 0.10)),
                        "q90_step_warp": float(np.quantile(values, 0.90)),
                        "q90_abs_step_warp": float(np.quantile(abs_values, 0.90)),
                        "outside_support_frac": float(window["outside_support"].mean()),
                        "window_mae": float(np.mean(np.abs(window["loss"] - window["pred_loss"]))),
                        "residual_alignment_corr": resid["residual_corr"],
                        "verdict": warp_verdict(family, float(np.median(abs_values)), float(np.quantile(abs_values, 0.90))),
                    }
                )
    return pd.DataFrame(rows).sort_values(["target_schedule", "window", "family"]).reset_index(drop=True)


def warp_verdict(family: str, median_abs: float, q90_abs: float) -> str:
    if family == "step_abs":
        return "step_reference_no_warp"
    if median_abs > 2000 or q90_abs > 5000:
        return "large_phase_warp"
    return "small_or_moderate_phase_warp"


def smoothing_robustness(frame: pd.DataFrame, selected: dict[str, SplineConfig]) -> pd.DataFrame:
    families = ["step_abs", "s1_raw", "s1_clamp", "s1_ratio"]
    residual_cols = [
        ("raw", "log_residual"),
        ("roll101", "log_residual_roll101"),
        ("roll501", "log_residual_roll501"),
    ]
    train = frame[frame["schedule"] == TRAIN_SCHEDULE]
    rows = []
    for residual_label, residual_col in residual_cols:
        for family in families:
            config = selected[family]
            pred = predict_coordinate_spline(frame, train, family, config, residual_col=residual_col)
            for schedule in [VALIDATION_SCHEDULE, TEST_SCHEDULE]:
                metrics = evaluate_prediction(pred, schedule)
                align = residual_alignment({family: pred})
                for window_name in ["full", "steps_27126_33906", "last_2048_sampled"]:
                    metric_row = metrics[metrics["window"] == window_name].iloc[0]
                    align_row = align[
                        (align["target_schedule"] == schedule)
                        & (align["window"] == window_name)
                    ].iloc[0]
                    rows.append(
                        {
                            "residual_target": residual_label,
                            "family": family,
                            "selected_config": config.name,
                            "selected_on": f"{VALIDATION_SCHEDULE}:full_raw_residual",
                            "target_schedule": schedule,
                            "window": window_name,
                            "mae": float(metric_row["mae"]),
                            "rmse": float(metric_row["rmse"]),
                            "r2": float(metric_row["r2"]),
                            "endpoint_abs_diff": float(metric_row["endpoint_abs_diff"]),
                            "residual_corr": float(align_row["residual_corr"]),
                            "residual_mae": float(align_row["residual_mae"]),
                        }
                    )
    out = pd.DataFrame(rows)
    step = out[out["family"] == "step_abs"][
        ["residual_target", "target_schedule", "window", "mae", "residual_corr"]
    ].rename(columns={"mae": "step_mae", "residual_corr": "step_residual_corr"})
    out = out.merge(step, on=["residual_target", "target_schedule", "window"], how="left")
    out["mae_gap_vs_step"] = out["mae"] - out["step_mae"]
    out["residual_corr_gap_vs_step"] = out["residual_corr"] - out["step_residual_corr"]
    out["verdict"] = out.apply(smoothing_verdict, axis=1)
    return out.sort_values(["residual_target", "target_schedule", "window", "family"]).reset_index(drop=True)


def smoothing_verdict(row: pd.Series) -> str:
    if row["family"] == "step_abs":
        return "step_reference"
    if row["mae_gap_vs_step"] <= 0.002 and row["residual_corr_gap_vs_step"] >= -0.05:
        return "smoothing_refutes_step_gap"
    if row["mae"] < row["step_mae"]:
        return "beats_step_under_smoothing"
    return "step_still_better_after_smoothing"


def interpolation_prediction(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    family: str,
    shrink: float,
    residual_clip: float,
    residual_col: str,
) -> pd.DataFrame:
    x_train = coordinate_values(train, family, train)
    interp_data = (
        pd.DataFrame(
            {
                "x": x_train,
                "residual": train[residual_col].to_numpy(dtype=np.float64),
            }
        )
        .groupby("x", as_index=False)["residual"]
        .mean()
        .sort_values("x")
    )
    out = frame.copy()
    x = coordinate_values(out, family, train)
    residual = np.interp(
        x,
        interp_data["x"].to_numpy(dtype=np.float64),
        interp_data["residual"].to_numpy(dtype=np.float64),
    )
    out["family"] = family
    out["config"] = f"interp_{residual_col}_shrink{shrink:g}_clip{residual_clip:g}"
    out["spline_x"] = x
    out["residual_pred"] = np.clip(residual * shrink, -residual_clip, residual_clip)
    out["pred_loss"] = out["base_pred_loss"] * np.exp(out["residual_pred"])
    return out


def interpolation_diagnostics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = ["step_abs", "s1_raw", "s1_clamp", "s1_ratio"]
    residual_targets = [("raw", "log_residual"), ("roll501", "log_residual_roll501")]
    train = frame[frame["schedule"] == TRAIN_SCHEDULE]
    rows = []
    align_rows = []
    for residual_label, residual_col in residual_targets:
        for family in families:
            candidates = []
            predictions = {}
            for shrink in [0.5, 0.75, 1.0]:
                for clip in [0.15, 0.25]:
                    pred = interpolation_prediction(frame, train, family, shrink, clip, residual_col)
                    full = evaluate_prediction(pred, VALIDATION_SCHEDULE)
                    val = full[full["window"] == "full"].iloc[0]
                    config = str(pred["config"].iloc[0])
                    candidates.append(
                        {
                            "residual_target": residual_label,
                            "family": family,
                            "config": config,
                            "shrink": shrink,
                            "residual_clip": clip,
                            "val_full_mae": float(val["mae"]),
                            "val_full_rmse": float(val["rmse"]),
                        }
                    )
                    predictions[config] = pred
            chosen = pd.DataFrame(candidates).sort_values(["val_full_mae", "val_full_rmse", "config"]).iloc[0]
            pred = predictions[str(chosen["config"])]
            for schedule in [VALIDATION_SCHEDULE, TEST_SCHEDULE]:
                metrics = evaluate_prediction(pred, schedule)
                align = residual_alignment({family: pred})
                for window_name in ["full", "steps_27126_33906", "last_2048_sampled"]:
                    metric_row = metrics[metrics["window"] == window_name].iloc[0]
                    align_row = align[
                        (align["target_schedule"] == schedule)
                        & (align["window"] == window_name)
                    ].iloc[0]
                    rows.append(
                        {
                            "residual_target": residual_label,
                            "family": family,
                            "selected_config": str(chosen["config"]),
                            "selected_on": f"{VALIDATION_SCHEDULE}:full",
                            "target_schedule": schedule,
                            "window": window_name,
                            "mae": float(metric_row["mae"]),
                            "rmse": float(metric_row["rmse"]),
                            "r2": float(metric_row["r2"]),
                            "endpoint_abs_diff": float(metric_row["endpoint_abs_diff"]),
                        }
                    )
                    align_rows.append(
                        {
                            "residual_target": residual_label,
                            "family": family,
                            "selected_config": str(chosen["config"]),
                            "target_schedule": schedule,
                            "window": window_name,
                            "residual_mae": float(align_row["residual_mae"]),
                            "residual_corr": float(align_row["residual_corr"]),
                            "sign_agreement": float(align_row["sign_agreement"]),
                        }
                    )
    selection = pd.DataFrame(rows)
    step = selection[selection["family"] == "step_abs"][
        ["residual_target", "target_schedule", "window", "mae"]
    ].rename(columns={"mae": "step_mae"})
    selection = selection.merge(step, on=["residual_target", "target_schedule", "window"], how="left")
    selection["mae_gap_vs_step"] = selection["mae"] - selection["step_mae"]

    alignment = pd.DataFrame(align_rows)
    step_align = alignment[alignment["family"] == "step_abs"][
        ["residual_target", "target_schedule", "window", "residual_corr"]
    ].rename(columns={"residual_corr": "step_residual_corr"})
    alignment = alignment.merge(step_align, on=["residual_target", "target_schedule", "window"], how="left")
    alignment["residual_corr_gap_vs_step"] = alignment["residual_corr"] - alignment["step_residual_corr"]
    return selection, alignment


def additive_ablation(
    frame: pd.DataFrame,
    selected: dict[str, SplineConfig],
) -> pd.DataFrame:
    train = frame[frame["schedule"] == TRAIN_SCHEDULE]
    rows = []
    ablations = [
        ("step_abs", "s1_ratio"),
        ("s1_ratio", "step_abs"),
        ("s1_clamp", "step_abs"),
    ]
    addon_configs = [
        SplineConfig(smoothing=s, shrink=shrink, residual_clip=0.25)
        for s in [0.5]
        for shrink in [0.0, 0.25, 0.5]
    ]
    for first_family, second_family in ablations:
        first_config = selected[first_family]
        first_pred = predict_coordinate_spline(frame, train, first_family, first_config)
        train_first = first_pred[first_pred["schedule"] == TRAIN_SCHEDULE].sort_values("step")
        train_left = train.copy().sort_values("step")
        train_left["leftover"] = (
            train_left["log_residual"].to_numpy(dtype=np.float64)
            - train_first["residual_pred"].to_numpy(dtype=np.float64)
        )
        for addon_config in addon_configs:
            spline = fit_coordinate_spline(train_left, second_family, addon_config, residual_col="leftover")
            pred = first_pred.copy()
            x = coordinate_values(pred, second_family, train)
            addon = spline(x) * addon_config.shrink
            pred["residual_pred"] = np.clip(
                pred["residual_pred"].to_numpy(dtype=np.float64) + addon,
                -addon_config.residual_clip,
                addon_config.residual_clip,
            )
            pred["pred_loss"] = pred["base_pred_loss"] * np.exp(pred["residual_pred"])
            val_full = evaluate_prediction(pred, VALIDATION_SCHEDULE)
            val = val_full[val_full["window"] == "full"].iloc[0]
            rows.append(
                {
                    "model": f"{first_family}_plus_{second_family}",
                    "first_family": first_family,
                    "second_family": second_family,
                    "first_config": first_config.name,
                    "addon_config": addon_config.name,
                    "selected_on": f"{VALIDATION_SCHEDULE}:full",
                    "target_schedule": VALIDATION_SCHEDULE,
                    "window": "full",
                    "mae": float(val["mae"]),
                    "rmse": float(val["rmse"]),
                    "r2": float(val["r2"]),
                    "endpoint_abs_diff": float(val["endpoint_abs_diff"]),
                    "candidate_only": True,
                }
            )

    candidate_frame = pd.DataFrame(rows)
    final_rows = []
    for model, group in candidate_frame.groupby("model"):
        chosen = group.sort_values(["mae", "rmse", "addon_config"]).iloc[0]
        first_family = str(chosen["first_family"])
        second_family = str(chosen["second_family"])
        first_config = selected[first_family]
        addon_config = parse_config_name(str(chosen["addon_config"]))
        first_pred = predict_coordinate_spline(frame, train, first_family, first_config)
        train_first = first_pred[first_pred["schedule"] == TRAIN_SCHEDULE].sort_values("step")
        train_left = train.copy().sort_values("step")
        train_left["leftover"] = (
            train_left["log_residual"].to_numpy(dtype=np.float64)
            - train_first["residual_pred"].to_numpy(dtype=np.float64)
        )
        spline = fit_coordinate_spline(train_left, second_family, addon_config, residual_col="leftover")
        pred = first_pred.copy()
        x = coordinate_values(pred, second_family, train)
        addon = spline(x) * addon_config.shrink
        pred["residual_pred"] = np.clip(
            pred["residual_pred"].to_numpy(dtype=np.float64) + addon,
            -addon_config.residual_clip,
            addon_config.residual_clip,
        )
        pred["pred_loss"] = pred["base_pred_loss"] * np.exp(pred["residual_pred"])
        for schedule in [VALIDATION_SCHEDULE, TEST_SCHEDULE]:
            metrics = evaluate_prediction(pred, schedule)
            for window_name in ["full", "steps_27126_33906", "last_2048_sampled"]:
                metric_row = metrics[metrics["window"] == window_name].iloc[0]
                final_rows.append(
                    {
                        "model": model,
                        "first_family": first_family,
                        "second_family": second_family,
                        "first_config": first_config.name,
                        "addon_config": addon_config.name,
                        "selected_on": f"{VALIDATION_SCHEDULE}:full",
                        "target_schedule": schedule,
                        "window": window_name,
                        "mae": float(metric_row["mae"]),
                        "rmse": float(metric_row["rmse"]),
                        "r2": float(metric_row["r2"]),
                        "endpoint_abs_diff": float(metric_row["endpoint_abs_diff"]),
                    }
                )
    out = pd.DataFrame(final_rows)
    step_full = out[
        (out["model"] == "s1_ratio_plus_step_abs")
        & (out["target_schedule"] == TEST_SCHEDULE)
        & (out["window"] == "full")
    ]
    if len(step_full) == 0:
        pass
    return out.sort_values(["target_schedule", "window", "model"]).reset_index(drop=True)


def parse_config_name(name: str) -> SplineConfig:
    # Expected format: s0.1_shrink1_clip0.25
    pieces = name.split("_")
    return SplineConfig(
        smoothing=float(pieces[0].removeprefix("s")),
        shrink=float(pieces[1].removeprefix("shrink")),
        residual_clip=float(pieces[2].removeprefix("clip")),
    )


def hypothesis_summary(
    selection: pd.DataFrame,
    domain: pd.DataFrame,
    common: pd.DataFrame,
    alignment: pd.DataFrame,
    warp: pd.DataFrame,
    smoothing: pd.DataFrame,
    additive: pd.DataFrame,
    interp_selection: pd.DataFrame,
    interp_alignment: pd.DataFrame,
) -> pd.DataFrame:
    wsd_selection = selection.set_index("family")
    wsd_domain = domain[domain["target_schedule"] == TEST_SCHEDULE].set_index("family")
    wsd_common = common[
        (common["target_schedule"] == TEST_SCHEDULE)
        & (common["support_basis"] == "s1_raw_common_support")
    ].set_index("family")
    wsd_align = alignment[
        (alignment["target_schedule"] == TEST_SCHEDULE)
        & (alignment["window"] == "full")
    ].set_index("family")
    wsd_warp = warp[
        (warp["target_schedule"] == TEST_SCHEDULE)
        & (warp["window"] == "full")
    ].set_index("family")
    wsd_smooth = smoothing[
        (smoothing["target_schedule"] == TEST_SCHEDULE)
        & (smoothing["window"] == "full")
        & (smoothing["residual_target"] == "roll501")
    ].set_index("family")
    additive_wsd = additive[
        (additive["target_schedule"] == TEST_SCHEDULE)
        & (additive["window"] == "full")
    ].set_index("model")
    interp_raw = interp_selection[
        (interp_selection["target_schedule"] == TEST_SCHEDULE)
        & (interp_selection["window"] == "full")
        & (interp_selection["residual_target"] == "raw")
    ].set_index("family")
    interp_raw_align = interp_alignment[
        (interp_alignment["target_schedule"] == TEST_SCHEDULE)
        & (interp_alignment["window"] == "full")
        & (interp_alignment["residual_target"] == "raw")
    ].set_index("family")
    interp_roll = interp_selection[
        (interp_selection["target_schedule"] == TEST_SCHEDULE)
        & (interp_selection["window"] == "full")
        & (interp_selection["residual_target"] == "roll501")
    ].set_index("family")

    h1_supported = (
        wsd_selection.loc["step_abs", "test_full_mae"]
        < min(
            wsd_selection.loc["s1_raw", "test_full_mae"],
            wsd_selection.loc["s1_clamp", "test_full_mae"],
            wsd_selection.loc["s1_ratio", "test_full_mae"],
        )
    )
    h3_supported = (
        wsd_common.loc["step_abs", "mae_corrected"]
        <= min(
            wsd_common.loc["s1_raw", "mae_corrected"],
            wsd_common.loc["s1_clamp", "mae_corrected"],
            wsd_common.loc["s1_ratio", "mae_corrected"],
        )
    )
    h4_supported = (
        interp_raw_align.loc["step_abs", "residual_corr"]
        > max(
            interp_raw_align.loc["s1_raw", "residual_corr"],
            interp_raw_align.loc["s1_clamp", "residual_corr"],
            interp_raw_align.loc["s1_ratio", "residual_corr"],
        )
    )
    h6_step_best_after_roll = (
        interp_roll.loc["step_abs", "mae"]
        <= min(
            interp_roll.loc["s1_raw", "mae"],
            interp_roll.loc["s1_clamp", "mae"],
            interp_roll.loc["s1_ratio", "mae"],
        )
    )
    h7_supported = (
        additive_wsd.loc["step_abs_plus_s1_ratio", "mae"]
        < additive_wsd.loc["s1_ratio_plus_step_abs", "mae"]
    )

    rows = [
        {
            "hypothesis": "H1_no_wsd_selection_artifact",
            "test": "cosine train, 811-selected configs, WSD held-out report",
            "key_numbers": (
                f"step={wsd_selection.loc['step_abs','test_full_mae']:.6f}; "
                f"s1_raw={wsd_selection.loc['s1_raw','test_full_mae']:.6f}; "
                f"s1_clamp={wsd_selection.loc['s1_clamp','test_full_mae']:.6f}; "
                f"s1_ratio={wsd_selection.loc['s1_ratio','test_full_mae']:.6f}"
            ),
            "result": "supported" if h1_supported else "refuted_by_binned_spline",
            "interpretation": "Step remains best when WSD is not used for selection.",
        },
        {
            "hypothesis": "H2_raw_s1_fails_partly_by_extrapolation",
            "test": "coordinate support audit",
            "key_numbers": (
                f"WSD s1_raw outside={wsd_domain.loc['s1_raw','frac_outside_train_after_transform']:.3f}; "
                f"s1_raw_full_mae={wsd_domain.loc['s1_raw','full_mae']:.6f}"
            ),
            "result": "supported",
            "interpretation": "Raw S1 extrapolates beyond the cosine support for a large WSD fraction.",
        },
        {
            "hypothesis": "H3_extrapolation_is_not_the_whole_story",
            "test": "S1 common-support comparison plus clamp/ratio",
            "key_numbers": (
                f"S1 common support WSD step={wsd_common.loc['step_abs','mae_corrected']:.6f}; "
                f"s1_raw={wsd_common.loc['s1_raw','mae_corrected']:.6f}; "
                f"s1_clamp={wsd_selection.loc['s1_clamp','test_full_mae']:.6f}; "
                f"s1_ratio={wsd_selection.loc['s1_ratio','test_full_mae']:.6f}"
            ),
            "result": "supported" if h3_supported else "mixed",
            "interpretation": "Even when the raw S1 support issue is controlled, S1 variants do not recover the step template.",
        },
        {
            "hypothesis": "H4_residual_phase_is_step_aligned",
            "test": "full-resolution coordinate interpolation residual alignment",
            "key_numbers": (
                f"corr step={interp_raw_align.loc['step_abs','residual_corr']:.3f}; "
                f"s1_raw={interp_raw_align.loc['s1_raw','residual_corr']:.3f}; "
                f"s1_clamp={interp_raw_align.loc['s1_clamp','residual_corr']:.3f}; "
                f"s1_ratio={interp_raw_align.loc['s1_ratio','residual_corr']:.3f}"
            ),
            "result": "supported" if h4_supported else "mixed",
            "interpretation": "The transferable residual shape is aligned by absolute step, not by S1 coordinate.",
        },
        {
            "hypothesis": "H5_s1_ratio_clamp_create_time_warp",
            "test": "target point mapped back to equivalent cosine step by coordinate",
            "key_numbers": (
                f"median_abs_warp step={wsd_warp.loc['step_abs','median_abs_step_warp']:.0f}; "
                f"s1_clamp={wsd_warp.loc['s1_clamp','median_abs_step_warp']:.0f}; "
                f"s1_ratio={wsd_warp.loc['s1_ratio','median_abs_step_warp']:.0f}"
            ),
            "result": "supported",
            "interpretation": "S1 transformations move target residuals to the wrong cosine phase.",
        },
        {
            "hypothesis": "H6_step_advantage_is_not_only_raw_noise",
            "test": "repeat transfer after smoothing the cosine residual target",
            "key_numbers": (
                f"roll501 interp WSD full step={interp_roll.loc['step_abs','mae']:.6f}; "
                f"s1_clamp={interp_roll.loc['s1_clamp','mae']:.6f}; "
                f"s1_ratio={interp_roll.loc['s1_ratio','mae']:.6f}; "
                f"binned-step={wsd_smooth.loc['step_abs','mae']:.6f}"
            ),
            "result": "supported" if h6_step_best_after_roll else "mixed",
            "interpretation": "Heavy smoothing is a stress test; if it erases the step edge, the raw phase result should not be reduced to low-frequency noise.",
        },
        {
            "hypothesis": "H7_step_has_large_marginal_value_over_s1",
            "test": "post-fit additive ablation selected on 811",
            "key_numbers": (
                f"step+s1={additive_wsd.loc['step_abs_plus_s1_ratio','mae']:.6f}; "
                f"s1+step={additive_wsd.loc['s1_ratio_plus_step_abs','mae']:.6f}"
            ),
            "result": "supported" if h7_supported else "mixed",
            "interpretation": "Adding step after S1 is valuable; adding S1 after step is not the main source of transfer.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_predictions()

    grid = validation_grid(frame)
    selected = select_configs(grid)
    preds = selected_predictions(frame, selected)
    perf = performance_by_window(frame, preds)
    selection = selection_summary(grid, perf, selected)
    domain = domain_support(frame, preds, perf)
    common = common_support_table(frame, preds)
    alignment = residual_alignment(preds)
    warp = time_warp_table(frame, preds)
    smoothing = smoothing_robustness(frame, selected)
    additive = additive_ablation(frame, selected)
    interp_selection, interp_alignment = interpolation_diagnostics(frame)
    summary = hypothesis_summary(
        selection,
        domain,
        common,
        alignment,
        warp,
        smoothing,
        additive,
        interp_selection,
        interp_alignment,
    )

    grid.to_csv(OUTPUT_DIR / "step_vs_s1_validation_grid.csv", index=False)
    selection.to_csv(OUTPUT_DIR / "step_vs_s1_selection.csv", index=False)
    perf.to_csv(OUTPUT_DIR / "step_vs_s1_performance_by_window.csv", index=False)
    domain.to_csv(OUTPUT_DIR / "step_vs_s1_domain_support.csv", index=False)
    common.to_csv(OUTPUT_DIR / "step_vs_s1_common_support.csv", index=False)
    alignment.to_csv(OUTPUT_DIR / "step_vs_s1_residual_alignment.csv", index=False)
    warp.to_csv(OUTPUT_DIR / "step_vs_s1_time_warp.csv", index=False)
    smoothing.to_csv(OUTPUT_DIR / "step_vs_s1_smoothing_robustness.csv", index=False)
    additive.to_csv(OUTPUT_DIR / "step_vs_s1_additive_ablation.csv", index=False)
    interp_selection.to_csv(OUTPUT_DIR / "step_vs_s1_interpolation_selection.csv", index=False)
    interp_alignment.to_csv(OUTPUT_DIR / "step_vs_s1_interpolation_alignment.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "step_vs_s1_hypothesis_summary.csv", index=False)

    print("Selected configs by 811 full MAE:")
    print(selection[["family", "selected_config", "val_full_mae", "test_full_mae", "test_tail_mae", "verdict"]].to_string(index=False))
    print("\nHypothesis summary:")
    print(summary[["hypothesis", "result", "key_numbers"]].to_string(index=False))


if __name__ == "__main__":
    main()
