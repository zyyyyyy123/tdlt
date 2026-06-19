from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = PROJECT_DIR / "outputs" / ".matplotlib"
XDG_CACHE_DIR = PROJECT_DIR / "outputs" / ".cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))


OUTPUT_DIR = PROJECT_DIR / "outputs"
PROJECT_ROOT = PROJECT_DIR.parents[1]
PREDICTIONS_PATH = PROJECT_ROOT / "results" / "baselines" / "momentum_residual_mlp" / "predictions.csv"

TRAIN_SCHEDULE = "cosine"
VALIDATION_SCHEDULE = "811"
TEST_SCHEDULE = "wsd"
START_STEP = 1000
END_STEP = 33906
SAMPLE_INTERVAL = 2
EPS = 1e-12

NOISE_EPS_GRID = [1e-12, 1e-4, 1e-2, 1.0]
EWMA_HALF_LIFE_GRID = [64, 512, 4096]
EFFECTIVE_EPS_LR2_GRID = [0.0, 1e-12, 1e-10, 1e-8, 1e-6]
SOFTSIGN_EPS_LR_GRID = [0.0, 1e-5, 1e-4, 1e-3]


@dataclass(frozen=True)
class CoordinateSpec:
    name: str
    group: str
    formula: str
    parameters: str
    rationale: str


@dataclass(frozen=True)
class InterpConfig:
    residual_label: str
    residual_col: str
    shrink: float
    residual_clip: float

    @property
    def name(self) -> str:
        return (
            f"interp_{self.residual_label}"
            f"_shrink{self.shrink:g}_clip{self.residual_clip:g}"
        )


CONFIG_GRID = [
    InterpConfig(residual_label=label, residual_col=col, shrink=shrink, residual_clip=clip)
    for label, col in [("raw", "log_residual"), ("roll501", "log_residual_roll501")]
    for shrink in [0.5, 0.75, 1.0]
    for clip in [0.15, 0.25]
]

REPORT_WINDOWS = [
    "full",
    "steps_1000_10000",
    "steps_10000_20000",
    "steps_20000_27125",
    "steps_27126_33906",
    "steps_30000_33906",
    "last_2048_sampled",
    "last_512_sampled",
]


def float_token(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.0e}".replace("+", "")


def control_specs() -> list[CoordinateSpec]:
    return [
        CoordinateSpec(
            name="step_abs",
            group="control_step",
            formula="step",
            parameters="none",
            rationale="Absolute-step residual phase control from the prior task2 audits.",
        ),
        CoordinateSpec(
            name="s1_raw",
            group="control_s1",
            formula="s1",
            parameters="none",
            rationale="Raw cumulative learning-rate coordinate used as the intrinsic-time control.",
        ),
        CoordinateSpec(
            name="s1_ratio",
            group="control_s1",
            formula="s1 / max_schedule_s1",
            parameters="max computed after task2 sampling window",
            rationale="Scale-normalized S1 control for schedule-to-schedule transfer.",
        ),
    ]


def proxy_specs() -> list[CoordinateSpec]:
    specs = [
        CoordinateSpec(
            name="sqrt_cum_lr2",
            group="sujianlin_noise",
            formula="sqrt(sum_t lr_t^2)",
            parameters="full schedule cumulative sum before task2 sampling",
            rationale=(
                "Schedule-only diffusion/noise-scale proxy inspired by BatchSize/LR "
                "scaling arguments; it has no batch or gradient variance state."
            ),
        )
    ]

    for eps in NOISE_EPS_GRID:
        specs.append(
            CoordinateSpec(
                name=f"noise_ratio_eps{float_token(eps)}",
                group="sujianlin_noise",
                formula="sum_t lr_t^2 / (s1_t^2 + eps)",
                parameters=f"eps={eps:g}",
                rationale=(
                    "Schedule-only noise-ratio proxy: cumulative LR^2 divided by "
                    "squared cumulative LR, echoing noise-scale vs progress tradeoffs."
                ),
            )
        )

    for half_life in EWMA_HALF_LIFE_GRID:
        for eps_lr2 in EFFECTIVE_EPS_LR2_GRID:
            specs.append(
                CoordinateSpec(
                    name=f"effective_update_hl{half_life}_epslr2_{float_token(eps_lr2)}",
                    group="sujianlin_update_rms",
                    formula="sum_t lr_t / sqrt(EWMA(lr_t^2; half_life) + eps)",
                    parameters=f"half_life={half_life}; eps_lr2={eps_lr2:g}",
                    rationale=(
                        "Schedule-only update-RMS proxy inspired by Adam/RMSProp "
                        "normalization; LR RMS substitutes for the unavailable update RMS."
                    ),
                )
            )
        for eps_lr in SOFTSIGN_EPS_LR_GRID:
            specs.append(
                CoordinateSpec(
                    name=f"softsign_lr_time_hl{half_life}_epslr_{float_token(eps_lr)}",
                    group="sujianlin_softsign",
                    formula="sum_t lr_t / (eps + sqrt(EWMA(lr_t^2; half_life)))",
                    parameters=f"half_life={half_life}; eps_lr={eps_lr:g}",
                    rationale=(
                        "Schedule-only SoftSign/Adam-epsilon proxy; epsilon is outside "
                        "the RMS term to mimic damped adaptive-step normalization."
                    ),
                )
            )
    return specs


def ewma_lr2(lr: np.ndarray, half_life: int) -> np.ndarray:
    beta = 0.5 ** (1.0 / float(half_life))
    lr2 = lr * lr
    out = np.empty_like(lr2, dtype=np.float64)
    state = float(lr2[0])
    for idx, value in enumerate(lr2):
        if idx == 0:
            state = float(value)
        else:
            state = beta * state + (1.0 - beta) * float(value)
        out[idx] = state
    return out


def add_schedule_only_proxy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in frame.sort_values(["schedule", "step"]).groupby("schedule", sort=False):
        group = group.copy()
        lr = group["lr"].to_numpy(dtype=np.float64)
        s1 = group["s1"].to_numpy(dtype=np.float64)
        cum_lr2 = np.cumsum(lr * lr)
        group["cum_lr2"] = cum_lr2
        group["sqrt_cum_lr2"] = np.sqrt(cum_lr2)

        for eps in NOISE_EPS_GRID:
            group[f"noise_ratio_eps{float_token(eps)}"] = cum_lr2 / (s1 * s1 + eps)

        for half_life in EWMA_HALF_LIFE_GRID:
            rms = np.sqrt(ewma_lr2(lr, half_life))
            for eps_lr2 in EFFECTIVE_EPS_LR2_GRID:
                denom = np.sqrt(rms * rms + eps_lr2)
                group[f"effective_update_hl{half_life}_epslr2_{float_token(eps_lr2)}"] = np.cumsum(
                    lr / np.maximum(denom, EPS)
                )
            for eps_lr in SOFTSIGN_EPS_LR_GRID:
                denom = eps_lr + rms
                group[f"softsign_lr_time_hl{half_life}_epslr_{float_token(eps_lr)}"] = np.cumsum(
                    lr / np.maximum(denom, EPS)
                )

        parts.append(group)
    return pd.concat(parts, ignore_index=True).sort_values(["schedule", "step"]).reset_index(drop=True)


def load_predictions() -> tuple[pd.DataFrame, list[CoordinateSpec]]:
    frame = pd.read_csv(PREDICTIONS_PATH)
    frame = frame.rename(columns={"run": "schedule", "momentum_s2": "base_pred_loss"})
    required = {"schedule", "step", "loss", "lr", "s1", "s2", "base_pred_loss"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {PREDICTIONS_PATH}: {sorted(missing)}")

    frame = frame[["schedule", "step", "loss", "lr", "s1", "s2", "base_pred_loss"]].copy()
    frame["schedule"] = frame["schedule"].astype(str)
    frame = add_schedule_only_proxy_columns(frame)
    frame = frame[
        (frame["step"] >= START_STEP)
        & (frame["step"] <= END_STEP)
        & (((frame["step"] - START_STEP) % SAMPLE_INTERVAL) == 0)
    ].copy()
    frame = frame.sort_values(["schedule", "step"]).reset_index(drop=True)
    frame["step_abs"] = frame["step"].astype(np.float64)
    frame["s1_raw"] = frame["s1"].astype(np.float64)
    frame["s1_ratio"] = frame["s1"] / np.maximum(
        frame.groupby("schedule")["s1"].transform("max").to_numpy(dtype=np.float64),
        EPS,
    )
    frame["log_residual"] = (
        np.log(frame["loss"].to_numpy(dtype=np.float64))
        - np.log(np.maximum(frame["base_pred_loss"].to_numpy(dtype=np.float64), EPS))
    )
    frame["log_residual_roll501"] = (
        frame.groupby("schedule")["log_residual"]
        .transform(lambda values: values.rolling(501, center=True, min_periods=1).mean())
        .to_numpy(dtype=np.float64)
    )
    specs = control_specs() + proxy_specs()
    return frame, specs


def window_slices(group: pd.DataFrame) -> dict[str, pd.DataFrame]:
    group = group.sort_values("step")
    windows = {
        "full": group,
        "steps_1000_10000": group[(group["step"] >= 1000) & (group["step"] <= 10000)],
        "steps_10000_20000": group[(group["step"] >= 10000) & (group["step"] <= 20000)],
        "steps_20000_27125": group[(group["step"] >= 20000) & (group["step"] <= 27125)],
        "steps_27126_33906": group[(group["step"] >= 27126) & (group["step"] <= 33906)],
        "steps_30000_33906": group[(group["step"] >= 30000) & (group["step"] <= 33906)],
        "last_2048_sampled": group.tail(2048),
        "last_512_sampled": group.tail(512),
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
    true_std = float(np.std(true_residual))
    pred_std = float(np.std(pred_residual))
    if true_std <= EPS or pred_std <= EPS:
        corr = float("nan")
        spearman = float("nan")
    else:
        corr = float(np.corrcoef(true_residual, pred_residual)[0, 1])
        spearman = float(pd.Series(true_residual).corr(pd.Series(pred_residual), method="spearman"))
    return {
        "residual_mae": float(np.mean(np.abs(diff))),
        "residual_rmse": float(np.sqrt(np.mean(diff * diff))),
        "residual_corr": corr,
        "residual_spearman": spearman,
        "sign_agreement": float(np.mean(np.sign(true_residual) == np.sign(pred_residual))),
        "signed_bias": float(np.mean(pred_residual - true_residual)),
        "true_residual_std": true_std,
        "pred_residual_std": pred_std,
    }


def interpolation_table(train: pd.DataFrame, coordinate: str, residual_col: str) -> pd.DataFrame:
    fit_data = pd.DataFrame(
        {
            "x": train[coordinate].to_numpy(dtype=np.float64),
            "residual": train[residual_col].to_numpy(dtype=np.float64),
        }
    ).replace([np.inf, -np.inf], np.nan)
    fit_data = fit_data.dropna().groupby("x", as_index=False)["residual"].mean().sort_values("x")
    if len(fit_data) < 2:
        raise RuntimeError(f"Not enough unique coordinate values for {coordinate}")
    return fit_data


def predict_coordinate(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    spec: CoordinateSpec,
    config: InterpConfig,
) -> pd.DataFrame:
    fit_data = interpolation_table(train, spec.name, config.residual_col)
    x_support = fit_data["x"].to_numpy(dtype=np.float64)
    residual_support = fit_data["residual"].to_numpy(dtype=np.float64)
    out = frame.copy()
    x = out[spec.name].to_numpy(dtype=np.float64)
    residual = np.interp(x, x_support, residual_support) * config.shrink
    out["coordinate"] = spec.name
    out["coordinate_group"] = spec.group
    out["config"] = config.name
    out["coordinate_x"] = x
    out["residual_pred"] = np.clip(residual, -config.residual_clip, config.residual_clip)
    out["pred_loss"] = out["base_pred_loss"] * np.exp(out["residual_pred"])
    return out


def predict_none(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["coordinate"] = "momentum_baseline"
    out["coordinate_group"] = "baseline"
    out["config"] = "none"
    out["coordinate_x"] = np.nan
    out["residual_pred"] = 0.0
    out["pred_loss"] = out["base_pred_loss"]
    return out


def evaluate_prediction(pred: pd.DataFrame, target_schedule: str) -> pd.DataFrame:
    rows = []
    target = pred[pred["schedule"] == target_schedule].sort_values("step")
    for window_name, window in window_slices(target).items():
        rows.append(
            {
                "target_schedule": target_schedule,
                "window": window_name,
                "n": len(window),
                **metric_dict(
                    window["loss"].to_numpy(dtype=np.float64),
                    window["pred_loss"].to_numpy(dtype=np.float64),
                ),
            }
        )
    return pd.DataFrame(rows)


def validation_grid(frame: pd.DataFrame, specs: list[CoordinateSpec]) -> pd.DataFrame:
    train = frame[frame["schedule"] == TRAIN_SCHEDULE]
    rows = []
    for spec in specs:
        for config in CONFIG_GRID:
            pred = predict_coordinate(frame, train, spec, config)
            metrics = evaluate_prediction(pred, VALIDATION_SCHEDULE)
            full = metrics[metrics["window"] == "full"].iloc[0]
            tail = metrics[metrics["window"] == "steps_27126_33906"].iloc[0]
            rows.append(
                {
                    "coordinate": spec.name,
                    "coordinate_group": spec.group,
                    "formula": spec.formula,
                    "parameters": spec.parameters,
                    "config": config.name,
                    "residual_target": config.residual_label,
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
    return pd.DataFrame(rows).sort_values(["val_full_mae", "val_full_rmse", "coordinate", "config"]).reset_index(drop=True)


def select_configs(grid: pd.DataFrame) -> dict[str, InterpConfig]:
    selected: dict[str, InterpConfig] = {}
    for coordinate, group in grid.groupby("coordinate", sort=False):
        row = group.sort_values(
            ["val_full_mae", "val_full_rmse", "val_full_endpoint_abs_diff", "config"]
        ).iloc[0]
        selected[str(coordinate)] = InterpConfig(
            residual_label=str(row["residual_target"]),
            residual_col="log_residual" if row["residual_target"] == "raw" else "log_residual_roll501",
            shrink=float(row["shrink"]),
            residual_clip=float(row["residual_clip"]),
        )
    return selected


def selected_predictions(
    frame: pd.DataFrame,
    specs: list[CoordinateSpec],
    selected: dict[str, InterpConfig],
) -> dict[str, pd.DataFrame]:
    train = frame[frame["schedule"] == TRAIN_SCHEDULE]
    spec_map = {spec.name: spec for spec in specs}
    return {
        coordinate: predict_coordinate(frame, train, spec_map[coordinate], config)
        for coordinate, config in selected.items()
    }


def performance_by_window(predictions: dict[str, pd.DataFrame], frame: pd.DataFrame) -> pd.DataFrame:
    base = predict_none(frame)
    rows = []
    for coordinate, pred in predictions.items():
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
                        "coordinate": coordinate,
                        "coordinate_group": str(pred["coordinate_group"].iloc[0]),
                        "config": str(pred["config"].iloc[0]),
                        "target_schedule": schedule,
                        "window": row["window"],
                        "n": int(row["n_corrected"]),
                        "mae_baseline": float(row["mae_baseline"]),
                        "mae_corrected": float(row["mae_corrected"]),
                        "mae_improvement": float(row["mae_baseline"] - row["mae_corrected"]),
                        "relative_mae_improvement_pct": float(
                            100.0 * (row["mae_baseline"] - row["mae_corrected"]) / row["mae_baseline"]
                        ),
                        "rmse_baseline": float(row["rmse_baseline"]),
                        "rmse_corrected": float(row["rmse_corrected"]),
                        "rmse_improvement": float(row["rmse_baseline"] - row["rmse_corrected"]),
                        "r2_corrected": float(row["r2_corrected"]),
                        "signed_bias_corrected": float(row["signed_bias_corrected"]),
                        "endpoint_abs_diff": float(row["endpoint_abs_diff_corrected"]),
                    }
                )
    out = pd.DataFrame(rows)
    step = out[out["coordinate"] == "step_abs"][
        ["target_schedule", "window", "mae_corrected", "rmse_corrected"]
    ].rename(columns={"mae_corrected": "step_mae", "rmse_corrected": "step_rmse"})
    out = out.merge(step, on=["target_schedule", "window"], how="left")
    out["mae_gap_vs_step"] = out["mae_corrected"] - out["step_mae"]
    out["rmse_gap_vs_step"] = out["rmse_corrected"] - out["step_rmse"]
    out["verdict"] = out.apply(performance_verdict, axis=1)
    return out.sort_values(["target_schedule", "window", "mae_corrected", "coordinate"]).reset_index(drop=True)


def performance_verdict(row: pd.Series) -> str:
    if row["coordinate"] == "step_abs":
        return "step_reference"
    if row["mae_corrected"] < row["step_mae"]:
        return "beats_step_reference"
    if row["mae_corrected"] < row["mae_baseline"] and row["mae_gap_vs_step"] <= 0.002:
        return "competitive_with_step"
    if row["mae_corrected"] < row["mae_baseline"]:
        return "helps_but_weaker_than_step"
    return "fails_vs_momentum"


def selection_summary(
    grid: pd.DataFrame,
    perf: pd.DataFrame,
    specs: list[CoordinateSpec],
) -> pd.DataFrame:
    spec_map = {spec.name: spec for spec in specs}
    selected_rows = []
    selected_grid = (
        grid.sort_values(["val_full_mae", "val_full_rmse", "val_full_endpoint_abs_diff", "config"])
        .groupby("coordinate", as_index=False, sort=False)
        .head(1)
        .copy()
    )
    selected_grid["selected_rank_by_811"] = selected_grid["val_full_mae"].rank(method="first")
    for _, row in selected_grid.iterrows():
        coordinate = str(row["coordinate"])
        spec = spec_map[coordinate]
        val_full = perf[
            (perf["coordinate"] == coordinate)
            & (perf["target_schedule"] == VALIDATION_SCHEDULE)
            & (perf["window"] == "full")
        ].iloc[0]
        val_tail = perf[
            (perf["coordinate"] == coordinate)
            & (perf["target_schedule"] == VALIDATION_SCHEDULE)
            & (perf["window"] == "steps_27126_33906")
        ].iloc[0]
        test_full = perf[
            (perf["coordinate"] == coordinate)
            & (perf["target_schedule"] == TEST_SCHEDULE)
            & (perf["window"] == "full")
        ].iloc[0]
        test_tail = perf[
            (perf["coordinate"] == coordinate)
            & (perf["target_schedule"] == TEST_SCHEDULE)
            & (perf["window"] == "steps_27126_33906")
        ].iloc[0]
        test_last = perf[
            (perf["coordinate"] == coordinate)
            & (perf["target_schedule"] == TEST_SCHEDULE)
            & (perf["window"] == "last_2048_sampled")
        ].iloc[0]
        selected_rows.append(
            {
                "coordinate": coordinate,
                "coordinate_group": spec.group,
                "selected_rank_by_811": int(row["selected_rank_by_811"]),
                "selected_config": str(row["config"]),
                "selected_on": str(row["selected_on"]),
                "formula": spec.formula,
                "parameters": spec.parameters,
                "rationale": spec.rationale,
                "val_full_mae": float(val_full["mae_corrected"]),
                "val_tail_mae": float(val_tail["mae_corrected"]),
                "test_full_mae": float(test_full["mae_corrected"]),
                "test_tail_mae": float(test_tail["mae_corrected"]),
                "test_last_2048_mae": float(test_last["mae_corrected"]),
                "test_mae_baseline": float(test_full["mae_baseline"]),
                "test_mae_improvement_vs_momentum": float(test_full["mae_improvement"]),
                "test_relative_improvement_pct": float(test_full["relative_mae_improvement_pct"]),
                "test_step_mae": float(test_full["step_mae"]),
                "test_mae_gap_vs_step": float(test_full["mae_gap_vs_step"]),
                "test_endpoint_abs_diff": float(test_full["endpoint_abs_diff"]),
                "verdict": str(test_full["verdict"]),
            }
        )
    return pd.DataFrame(selected_rows).sort_values(["selected_rank_by_811", "test_full_mae"]).reset_index(drop=True)


def residual_alignment(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for coordinate, pred in predictions.items():
        for schedule in [VALIDATION_SCHEDULE, TEST_SCHEDULE]:
            target = pred[pred["schedule"] == schedule].sort_values("step")
            for window_name, window in window_slices(target).items():
                rows.append(
                    {
                        "coordinate": coordinate,
                        "coordinate_group": str(pred["coordinate_group"].iloc[0]),
                        "config": str(pred["config"].iloc[0]),
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
    step = out[out["coordinate"] == "step_abs"][
        ["target_schedule", "window", "residual_mae", "residual_corr", "sign_agreement"]
    ].rename(
        columns={
            "residual_mae": "step_residual_mae",
            "residual_corr": "step_residual_corr",
            "sign_agreement": "step_sign_agreement",
        }
    )
    out = out.merge(step, on=["target_schedule", "window"], how="left")
    out["residual_mae_gap_vs_step"] = out["residual_mae"] - out["step_residual_mae"]
    out["residual_corr_gap_vs_step"] = out["residual_corr"] - out["step_residual_corr"]
    out["sign_agreement_gap_vs_step"] = out["sign_agreement"] - out["step_sign_agreement"]
    return out.sort_values(["target_schedule", "window", "residual_mae", "coordinate"]).reset_index(drop=True)


def masked_loss_mae(pred: pd.DataFrame, mask: np.ndarray) -> float:
    if int(np.sum(mask)) == 0:
        return float("nan")
    actual = pred.loc[mask, "loss"].to_numpy(dtype=np.float64)
    values = pred.loc[mask, "pred_loss"].to_numpy(dtype=np.float64)
    return float(np.mean(np.abs(actual - values)))


def domain_support(
    frame: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    perf: pd.DataFrame,
) -> pd.DataFrame:
    train = frame[frame["schedule"] == TRAIN_SCHEDULE]
    rows = []
    for coordinate, pred in predictions.items():
        train_x = train[coordinate].to_numpy(dtype=np.float64)
        train_min = float(np.min(train_x))
        train_max = float(np.max(train_x))
        train_range = max(train_max - train_min, EPS)
        for schedule in [VALIDATION_SCHEDULE, TEST_SCHEDULE]:
            target = frame[frame["schedule"] == schedule].sort_values("step")
            target_pred = pred[pred["schedule"] == schedule].sort_values("step")
            target_x = target[coordinate].to_numpy(dtype=np.float64)
            below = target_x < train_min
            above = target_x > train_max
            outside = below | above
            max_extrapolation = float(
                max(
                    np.max(np.maximum(train_min - target_x, 0.0)),
                    np.max(np.maximum(target_x - train_max, 0.0)),
                )
            )
            full_mae = float(
                perf[
                    (perf["coordinate"] == coordinate)
                    & (perf["target_schedule"] == schedule)
                    & (perf["window"] == "full")
                ].iloc[0]["mae_corrected"]
            )
            rows.append(
                {
                    "coordinate": coordinate,
                    "coordinate_group": str(pred["coordinate_group"].iloc[0]),
                    "target_schedule": schedule,
                    "train_x_min": train_min,
                    "train_x_max": train_max,
                    "target_x_min": float(np.min(target_x)),
                    "target_x_max": float(np.max(target_x)),
                    "train_x_range": train_range,
                    "target_x_range": float(np.max(target_x) - np.min(target_x)),
                    "frac_below_train": float(np.mean(below)),
                    "frac_above_train": float(np.mean(above)),
                    "frac_outside_train": float(np.mean(outside)),
                    "max_extrapolation": max_extrapolation,
                    "normalized_max_extrapolation": max_extrapolation / train_range,
                    "full_mae": full_mae,
                    "inside_support_loss_mae": masked_loss_mae(target_pred, ~outside),
                    "outside_support_loss_mae": masked_loss_mae(target_pred, outside),
                    "n_inside_support": int(np.sum(~outside)),
                    "n_outside_support": int(np.sum(outside)),
                    "verdict": domain_verdict(coordinate, float(np.mean(outside)), full_mae),
                }
            )
    return pd.DataFrame(rows).sort_values(["target_schedule", "frac_outside_train", "full_mae", "coordinate"]).reset_index(drop=True)


def domain_verdict(coordinate: str, frac_outside: float, full_mae: float) -> str:
    if coordinate == "step_abs":
        return "no_coordinate_support_problem"
    if frac_outside > 0.25:
        return "large_support_mismatch"
    if frac_outside > 0.05:
        return "moderate_support_mismatch"
    if np.isfinite(full_mae):
        return "mostly_inside_cosine_support"
    return "not_evaluated"


def time_warp_summary(
    frame: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    train = frame[frame["schedule"] == TRAIN_SCHEDULE].sort_values("step")
    rows = []
    for coordinate, pred in predictions.items():
        interp_data = (
            pd.DataFrame(
                {
                    "x": train[coordinate].to_numpy(dtype=np.float64),
                    "step": train["step"].to_numpy(dtype=np.float64),
                }
            )
            .groupby("x", as_index=False)["step"]
            .mean()
            .sort_values("x")
        )
        x_support = interp_data["x"].to_numpy(dtype=np.float64)
        step_support = interp_data["step"].to_numpy(dtype=np.float64)
        train_min = float(np.min(x_support))
        train_max = float(np.max(x_support))
        for schedule in [VALIDATION_SCHEDULE, TEST_SCHEDULE]:
            target = frame[frame["schedule"] == schedule].sort_values("step")
            target_pred = pred[pred["schedule"] == schedule].sort_values("step").copy()
            target_x = target[coordinate].to_numpy(dtype=np.float64)
            equiv_step = np.interp(target_x, x_support, step_support)
            outside = (target_x < train_min) | (target_x > train_max)
            target_pred["equiv_cosine_step"] = equiv_step
            target_pred["step_warp"] = target["step"].to_numpy(dtype=np.float64) - equiv_step
            target_pred["outside_support"] = outside
            for window_name, window in window_slices(target_pred).items():
                warp = window["step_warp"].to_numpy(dtype=np.float64)
                abs_warp = np.abs(warp)
                resid = residual_metric_dict(
                    window["log_residual"].to_numpy(dtype=np.float64),
                    window["residual_pred"].to_numpy(dtype=np.float64),
                )
                rows.append(
                    {
                        "coordinate": coordinate,
                        "coordinate_group": str(pred["coordinate_group"].iloc[0]),
                        "target_schedule": schedule,
                        "window": window_name,
                        "n": len(window),
                        "target_step_median": float(window["step"].median()),
                        "equiv_cosine_step_median": float(window["equiv_cosine_step"].median()),
                        "median_step_warp": float(np.median(warp)),
                        "median_abs_step_warp": float(np.median(abs_warp)),
                        "q10_step_warp": float(np.quantile(warp, 0.10)),
                        "q90_step_warp": float(np.quantile(warp, 0.90)),
                        "q90_abs_step_warp": float(np.quantile(abs_warp, 0.90)),
                        "outside_support_frac": float(window["outside_support"].mean()),
                        "window_mae": float(np.mean(np.abs(window["loss"] - window["pred_loss"]))),
                        "residual_alignment_corr": resid["residual_corr"],
                        "sign_agreement": resid["sign_agreement"],
                    }
                )
    return pd.DataFrame(rows).sort_values(["target_schedule", "window", "median_abs_step_warp", "coordinate"]).reset_index(drop=True)


def coordinate_catalog(frame: pd.DataFrame, specs: list[CoordinateSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        for schedule, group in frame.groupby("schedule", sort=True):
            group = group.sort_values("step")
            x = group[spec.name].to_numpy(dtype=np.float64)
            step = group["step"].to_numpy(dtype=np.float64)
            diff = np.diff(x)
            nondecreasing = float(np.mean(diff >= -EPS)) if len(diff) else float("nan")
            nonincreasing = float(np.mean(diff <= EPS)) if len(diff) else float("nan")
            rows.append(
                {
                    "coordinate": spec.name,
                    "coordinate_group": spec.group,
                    "schedule": schedule,
                    "formula": spec.formula,
                    "parameters": spec.parameters,
                    "rationale": spec.rationale,
                    "x_min": float(np.min(x)),
                    "x_max": float(np.max(x)),
                    "x_range": float(np.max(x) - np.min(x)),
                    "pearson_corr_with_step": float(np.corrcoef(step, x)[0, 1]) if np.std(x) > EPS else float("nan"),
                    "spearman_corr_with_step": float(pd.Series(step).corr(pd.Series(x), method="spearman")),
                    "frac_nondecreasing_step_order": nondecreasing,
                    "frac_nonincreasing_step_order": nonincreasing,
                }
            )
    return pd.DataFrame(rows).sort_values(["coordinate_group", "coordinate", "schedule"]).reset_index(drop=True)


def hypothesis_summary(
    selection: pd.DataFrame,
    alignment: pd.DataFrame,
    domain: pd.DataFrame,
    warp: pd.DataFrame,
) -> pd.DataFrame:
    wsd_selection = selection.set_index("coordinate")
    wsd_align = alignment[
        (alignment["target_schedule"] == TEST_SCHEDULE)
        & (alignment["window"] == "full")
    ].set_index("coordinate")
    wsd_domain = domain[domain["target_schedule"] == TEST_SCHEDULE].set_index("coordinate")
    wsd_warp = warp[
        (warp["target_schedule"] == TEST_SCHEDULE)
        & (warp["window"] == "full")
    ].set_index("coordinate")

    proxy_selection = selection[selection["coordinate_group"].str.startswith("sujianlin")]
    noise_selection = selection[selection["coordinate_group"] == "sujianlin_noise"]
    update_selection = selection[
        selection["coordinate_group"].isin(["sujianlin_update_rms", "sujianlin_softsign"])
    ]

    best_proxy = proxy_selection.sort_values(["val_full_mae", "test_full_mae"]).iloc[0]
    best_noise = noise_selection.sort_values(["val_full_mae", "test_full_mae"]).iloc[0]
    best_update = update_selection.sort_values(["val_full_mae", "test_full_mae"]).iloc[0]
    step = wsd_selection.loc["step_abs"]
    s1_ratio = wsd_selection.loc["s1_ratio"]
    baseline_mae = float(step["test_mae_baseline"])

    best_proxy_name = str(best_proxy["coordinate"])
    best_noise_name = str(best_noise["coordinate"])
    best_update_name = str(best_update["coordinate"])

    proxy_result = "negative_result"
    if float(best_proxy["test_full_mae"]) < baseline_mae:
        proxy_result = "mixed"
    if float(best_proxy["test_full_mae"]) <= float(step["test_full_mae"]) + 0.002:
        proxy_result = "supported_but_step_like"

    noise_result = "negative_result"
    if float(best_noise["test_full_mae"]) < baseline_mae and float(wsd_align.loc[best_noise_name, "residual_corr"]) > 0.3:
        noise_result = "mixed"

    update_result = "negative_result"
    if float(best_update["test_full_mae"]) < baseline_mae:
        update_result = "mixed"
    if float(best_update["test_full_mae"]) <= float(step["test_full_mae"]) + 0.002:
        update_result = "supported_but_step_like"

    rows = [
        {
            "hypothesis": "H1_proxy_coordinates_can_transfer_residual_phase",
            "test": "cosine interpolation, config/coordinate ranked on 811 full MAE, WSD held out",
            "result": proxy_result,
            "key_numbers": (
                f"best_proxy={best_proxy_name}; "
                f"WSD_MAE={float(best_proxy['test_full_mae']):.6f}; "
                f"step={float(step['test_full_mae']):.6f}; "
                f"s1_ratio={float(s1_ratio['test_full_mae']):.6f}; "
                f"baseline={baseline_mae:.6f}"
            ),
            "interpretation": (
                "A schedule-only proxy is useful only to the extent that it transfers "
                "the cosine residual phase without using WSD for selection."
            ),
        },
        {
            "hypothesis": "H2_noise_scale_proxy_is_enough",
            "test": "sqrt(cum_lr2) and cum_lr2/(s1^2+eps) coordinates",
            "result": noise_result,
            "key_numbers": (
                f"best_noise={best_noise_name}; "
                f"WSD_MAE={float(best_noise['test_full_mae']):.6f}; "
                f"corr={float(wsd_align.loc[best_noise_name, 'residual_corr']):.3f}; "
                f"outside={float(wsd_domain.loc[best_noise_name, 'frac_outside_train']):.3f}"
            ),
            "interpretation": (
                "This checks the BatchSize/noise-scale-inspired coordinate alone; it "
                "does not estimate actual gradient noise or batch-size effects."
            ),
        },
        {
            "hypothesis": "H3_update_rms_or_softsign_time_is_better_than_s1",
            "test": "EWMA-LR2 effective update time and softsign epsilon variants",
            "result": update_result,
            "key_numbers": (
                f"best_update={best_update_name}; "
                f"WSD_MAE={float(best_update['test_full_mae']):.6f}; "
                f"corr={float(wsd_align.loc[best_update_name, 'residual_corr']):.3f}; "
                f"median_abs_warp={float(wsd_warp.loc[best_update_name, 'median_abs_step_warp']):.0f}"
            ),
            "interpretation": (
                "These are Adam/update-RMS-inspired schedule-only coordinates, with "
                "LR RMS standing in for unavailable gradient/update RMS state."
            ),
        },
        {
            "hypothesis": "H4_no_real_adam_or_gradient_state",
            "test": "scope check",
            "result": "caveat",
            "key_numbers": "all proxy inputs are step, lr, s1, and cumulative or EWMA transforms of lr",
            "interpretation": (
                "The experiment can support or reject these coordinates as black-box "
                "schedule proxies only; it cannot validate real Adam epsilon, update "
                "RMS, gradient noise, or BatchSize scaling mechanisms."
            ),
        },
    ]
    return pd.DataFrame(rows)


def write_outputs(
    grid: pd.DataFrame,
    selection: pd.DataFrame,
    perf: pd.DataFrame,
    alignment: pd.DataFrame,
    domain: pd.DataFrame,
    warp: pd.DataFrame,
    catalog: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_csv(OUTPUT_DIR / "sujianlin_noise_update_validation_grid.csv", index=False)
    selection.to_csv(OUTPUT_DIR / "sujianlin_noise_update_coordinate_selection.csv", index=False)
    perf[perf["target_schedule"] == TEST_SCHEDULE].to_csv(
        OUTPUT_DIR / "sujianlin_noise_update_wsd_window_metrics.csv",
        index=False,
    )
    alignment.to_csv(OUTPUT_DIR / "sujianlin_noise_update_residual_alignment.csv", index=False)
    domain.to_csv(OUTPUT_DIR / "sujianlin_noise_update_domain_support.csv", index=False)
    warp.to_csv(OUTPUT_DIR / "sujianlin_noise_update_warp_summary.csv", index=False)
    catalog.to_csv(OUTPUT_DIR / "sujianlin_noise_update_coordinate_catalog.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "sujianlin_noise_update_hypothesis_summary.csv", index=False)


def main() -> None:
    frame, specs = load_predictions()
    grid = validation_grid(frame, specs)
    selected = select_configs(grid)
    predictions = selected_predictions(frame, specs, selected)
    perf = performance_by_window(predictions, frame)
    selection = selection_summary(grid, perf, specs)
    alignment = residual_alignment(predictions)
    domain = domain_support(frame, predictions, perf)
    warp = time_warp_summary(frame, predictions)
    catalog = coordinate_catalog(frame, specs)
    summary = hypothesis_summary(selection, alignment, domain, warp)
    write_outputs(grid, selection, perf, alignment, domain, warp, catalog, summary)

    print("Top coordinates selected on 811 full MAE:")
    display = selection[
        [
            "selected_rank_by_811",
            "coordinate",
            "coordinate_group",
            "selected_config",
            "val_full_mae",
            "test_full_mae",
            "test_tail_mae",
            "test_mae_gap_vs_step",
            "verdict",
        ]
    ].head(12)
    print(display.to_string(index=False))

    print("\nWSD full residual alignment for top 811 coordinates:")
    top_names = display["coordinate"].tolist()
    align_report = alignment[
        (alignment["target_schedule"] == TEST_SCHEDULE)
        & (alignment["window"] == "full")
        & (alignment["coordinate"].isin(top_names))
    ][["coordinate", "residual_corr", "sign_agreement", "residual_mae", "residual_corr_gap_vs_step"]]
    print(align_report.sort_values("residual_mae").to_string(index=False))

    print("\nHypothesis summary:")
    print(summary[["hypothesis", "result", "key_numbers"]].to_string(index=False))


if __name__ == "__main__":
    main()
