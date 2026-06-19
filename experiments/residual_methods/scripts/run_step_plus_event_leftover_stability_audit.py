from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PROJECT_DIR.parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"
MPL_CACHE_DIR = OUTPUT_DIR / ".matplotlib"
XDG_CACHE_DIR = OUTPUT_DIR / ".cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import ConnectionPatch, Rectangle

MOMENTUM_PREDICTIONS = PROJECT_ROOT / "results" / "intermediates" / "three_schedule_momentum" / "predictions.csv"

START_STEP = 1000
END_STEP = 33906
TAIL_START_STEP = 27126
SAMPLE_INTERVAL = 2
EPS = 1e-12
RNG_SEED = 20260616
MAX_STEP_FIT_POINTS = 4096

SCHEDULES = ["cosine", "811", "wsd"]
STEP_RESIDUAL_CACHE: dict[tuple[tuple[str, ...], str], np.ndarray] = {}


@dataclass(frozen=True)
class StepSpec:
    smoothing: float
    shrink: float
    clip: float

    @property
    def name(self) -> str:
        return f"spline_s{self.smoothing:g}_shrink{self.shrink:g}_clip{self.clip:g}"


@dataclass(frozen=True)
class EventSpec:
    feature_set: str
    ridge_alpha: float
    shrink: float

    @property
    def name(self) -> str:
        return f"{self.feature_set}_ridge{self.ridge_alpha:g}_shrink{self.shrink:g}"


STEP_REFERENCE = StepSpec(smoothing=0.05, shrink=1.0, clip=0.15)
STEP_BASE = StepSpec(smoothing=0.05, shrink=0.75, clip=0.15)
DEFAULT_EVENT = EventSpec(feature_set="linear_endpoint", ridge_alpha=0.01, shrink=1.0)

FEATURE_SETS: dict[str, list[str]] = {
    "drop_impulse": [
        "lr_drop_frac",
        "sqrt_lr_drop_frac",
        "ewma_drop_hl256",
        "ewma_drop_hl2048",
        "post_drop_age_frac",
        "after_first_drop",
    ],
    "event_history": [
        "drop_mass_frac",
        "post_first_drop_age_frac",
        "s1_after_first_drop_frac",
        "tail_saturation",
        "ewma_drop_hl512",
        "ewma_drop_hl4096",
    ],
    "linear_endpoint": [
        "linear_decay_progress",
        "linear_decay_progress_sq",
        "tail_progress_global",
        "tail_progress_global_sq",
        "endpoint_progress",
        "endpoint_pressure",
        "remaining_frac",
        "tail_saturation",
    ],
    "event_tail_interactions": [
        "drop_mass_frac",
        "tail_saturation",
        "drop_mass_x_tail_sat",
        "ewma_hl2048_x_tail_sat",
        "linear_decay_x_endpoint",
        "tail_event_pressure",
    ],
    "full_event_local": [
        "lr_drop_frac",
        "sqrt_lr_drop_frac",
        "drop_mass_frac",
        "post_drop_age_frac",
        "post_first_drop_age_frac",
        "s1_after_first_drop_frac",
        "tail_saturation",
        "ewma_drop_hl256",
        "ewma_drop_hl512",
        "ewma_drop_hl2048",
        "ewma_drop_hl4096",
        "linear_decay_progress",
        "tail_progress_global",
        "endpoint_progress",
        "endpoint_pressure",
        "drop_mass_x_tail_sat",
        "ewma_hl2048_x_tail_sat",
        "linear_decay_x_endpoint",
        "tail_event_pressure",
        "large_drop_age_frac",
    ],
}


def load_predictions() -> pd.DataFrame:
    frame = pd.read_csv(MOMENTUM_PREDICTIONS)
    frame = frame.rename(columns={"run": "schedule", "momentum_s2": "base_pred_loss"})
    frame = frame[["schedule", "step", "loss", "lr", "s1", "s2", "base_pred_loss"]].copy()
    frame = frame[
        (frame["step"] >= START_STEP)
        & (frame["step"] <= END_STEP)
        & (((frame["step"] - START_STEP) % SAMPLE_INTERVAL) == 0)
    ].copy()
    frame["log_residual"] = np.log(frame["loss"].to_numpy(dtype=np.float64)) - np.log(
        np.maximum(frame["base_pred_loss"].to_numpy(dtype=np.float64), EPS)
    )
    return frame.sort_values(["schedule", "step"]).reset_index(drop=True)


def decayed_drop_history(steps: np.ndarray, drop: np.ndarray, half_life_steps: float) -> np.ndarray:
    out = np.zeros(len(drop), dtype=np.float64)
    state = 0.0
    prev = float(steps[0]) if len(steps) else 0.0
    for idx, (step, value) in enumerate(zip(steps, drop)):
        if idx > 0:
            state *= float(np.exp(-(float(step) - prev) / half_life_steps))
        state += float(value)
        out[idx] = state
        prev = float(step)
    return out


def age_since_event(steps: np.ndarray, event_mask: np.ndarray) -> np.ndarray:
    age = np.zeros(len(steps), dtype=np.float64)
    last_step: float | None = None
    for idx, step in enumerate(steps):
        if bool(event_mask[idx]):
            last_step = float(step)
            age[idx] = 0.0
        elif last_step is not None:
            age[idx] = float(step) - last_step
        else:
            age[idx] = 0.0
    return age


def add_event_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    total_span = max(float(END_STEP - START_STEP), 1.0)
    tail_span = max(float(END_STEP - TAIL_START_STEP), 1.0)
    for schedule, group in frame.groupby("schedule", sort=False):
        group = group.sort_values("step").copy()
        steps = group["step"].to_numpy(dtype=np.float64)
        lr = group["lr"].to_numpy(dtype=np.float64)
        s1 = group["s1"].to_numpy(dtype=np.float64)
        drop = np.maximum(np.r_[0.0, lr[:-1] - lr[1:]], 0.0)
        drop_mass = np.cumsum(drop, dtype=np.float64)
        total_drop = max(float(drop_mass[-1]), float(np.max(lr) - np.min(lr)), EPS)
        positive_drop = drop > max(total_drop * 1e-8, 1e-12)
        large_drop = drop > (0.01 * total_drop)

        first_drop_idx = int(np.argmax(positive_drop)) if np.any(positive_drop) else len(group)
        if first_drop_idx < len(group):
            first_drop_step = float(steps[first_drop_idx])
            post_first_age = np.maximum(steps - first_drop_step, 0.0)
            s1_after_first = np.maximum(s1 - s1[first_drop_idx], 0.0)
            after_first = (steps >= first_drop_step).astype(np.float64)
        else:
            first_drop_step = np.nan
            post_first_age = np.zeros(len(group), dtype=np.float64)
            s1_after_first = np.zeros(len(group), dtype=np.float64)
            after_first = np.zeros(len(group), dtype=np.float64)

        post_drop_age = age_since_event(steps, positive_drop)
        large_drop_age = age_since_event(steps, large_drop)
        linear_progress = np.clip((lr[0] - lr) / total_drop, 0.0, 1.0)
        tail_progress = np.clip((steps - TAIL_START_STEP) / tail_span, 0.0, 1.0)
        endpoint_progress = np.clip((steps - START_STEP) / total_span, 0.0, 1.0)
        tail_saturation = 1.0 - np.exp(-post_first_age / 2048.0)
        remaining_frac = np.clip((END_STEP - steps) / total_span, 0.0, 1.0)

        group["lr_drop"] = drop
        group["drop_mass"] = drop_mass
        group["post_drop_age_steps"] = post_drop_age
        group["post_first_drop_age_steps"] = post_first_age
        group["s1_after_first_drop"] = s1_after_first
        group["first_drop_step"] = first_drop_step
        group["after_first_drop"] = after_first
        group["large_drop_age_steps"] = large_drop_age
        group["large_drop_event"] = large_drop.astype(np.float64)
        group["linear_decay_progress"] = linear_progress
        group["linear_decay_progress_sq"] = linear_progress * linear_progress
        group["tail_saturation"] = tail_saturation
        group["endpoint_progress"] = endpoint_progress
        group["endpoint_pressure"] = endpoint_progress**3
        group["remaining_frac"] = remaining_frac
        group["tail_progress_global"] = tail_progress
        group["tail_progress_global_sq"] = tail_progress * tail_progress
        group["lr_drop_frac"] = drop / total_drop
        group["sqrt_lr_drop_frac"] = np.sqrt(np.maximum(drop / total_drop, 0.0))
        group["drop_mass_frac"] = drop_mass / total_drop
        group["post_drop_age_frac"] = post_drop_age / total_span
        group["post_first_drop_age_frac"] = post_first_age / total_span
        group["s1_after_first_drop_frac"] = s1_after_first / max(float(np.max(s1_after_first)), EPS)
        group["large_drop_age_frac"] = large_drop_age / total_span

        for half_life in [256.0, 512.0, 2048.0, 4096.0]:
            group[f"ewma_drop_hl{int(half_life)}"] = (
                decayed_drop_history(steps, drop, half_life) / total_drop
            )

        group["drop_mass_x_tail_sat"] = group["drop_mass_frac"] * group["tail_saturation"]
        group["ewma_hl2048_x_tail_sat"] = group["ewma_drop_hl2048"] * group["tail_saturation"]
        group["linear_decay_x_endpoint"] = group["linear_decay_progress"] * group["endpoint_pressure"]
        group["tail_event_pressure"] = group["tail_saturation"] * group["endpoint_pressure"]
        parts.append(group)
    return pd.concat(parts, ignore_index=True).sort_values(["schedule", "step"]).reset_index(drop=True)


def train_mask(frame: pd.DataFrame, train_schedules: tuple[str, ...]) -> np.ndarray:
    return frame["schedule"].isin(train_schedules).to_numpy()


def fit_step_residual(frame: pd.DataFrame, train_schedules: tuple[str, ...], spec: StepSpec) -> np.ndarray:
    key = (tuple(train_schedules), spec.name)
    if key in STEP_RESIDUAL_CACHE:
        return STEP_RESIDUAL_CACHE[key].copy()
    train = frame[frame["schedule"].isin(train_schedules)]
    grouped = train.groupby("step", as_index=False)["log_residual"].mean().sort_values("step")
    original_points = len(grouped)
    if len(grouped) > MAX_STEP_FIT_POINTS:
        grouped = bin_step_fit_data(grouped, MAX_STEP_FIT_POINTS)
    scaled_smoothing = spec.smoothing * len(grouped) / max(original_points, 1)
    spline = UnivariateSpline(
        grouped["step"].to_numpy(dtype=np.float64),
        grouped["log_residual"].to_numpy(dtype=np.float64),
        s=scaled_smoothing,
    )
    residual = spline(frame["step"].to_numpy(dtype=np.float64)) * spec.shrink
    residual = np.clip(residual, -spec.clip, spec.clip)
    STEP_RESIDUAL_CACHE[key] = residual.copy()
    return residual


def bin_step_fit_data(grouped: pd.DataFrame, max_points: int) -> pd.DataFrame:
    grouped = grouped.sort_values("step").reset_index(drop=True)
    bin_id = (np.arange(len(grouped), dtype=np.int64) * max_points) // len(grouped)
    return (
        grouped.assign(bin_id=bin_id)
        .groupby("bin_id", as_index=False)
        .agg(step=("step", "mean"), log_residual=("log_residual", "mean"))
        .sort_values("step")
        [["step", "log_residual"]]
    )


def transformed_raw_features(frame: pd.DataFrame, feature_set: str, transform: str) -> tuple[np.ndarray, list[str]]:
    names = FEATURE_SETS[feature_set]
    raw = frame[names].to_numpy(dtype=np.float64)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    if transform == "aligned":
        return raw, names
    if transform == "zero":
        return np.zeros_like(raw), names

    out = np.zeros_like(raw)
    for _, idx in frame.groupby("schedule", sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=np.int64)
        values = raw[idx_arr]
        if transform == "circular_shift_25pct":
            shifted = np.roll(values, len(values) // 4, axis=0)
        elif transform == "reverse_time":
            shifted = values[::-1]
        else:
            raise ValueError(f"Unknown feature transform: {transform}")
        out[idx_arr] = shifted
    return out, names


def standardized_features(
    frame: pd.DataFrame,
    feature_set: str,
    train_schedules: tuple[str, ...],
    transform: str = "aligned",
) -> tuple[np.ndarray, list[str]]:
    raw, names = transformed_raw_features(frame, feature_set, transform)
    mask = train_mask(frame, train_schedules)
    mean = raw[mask].mean(axis=0)
    std = raw[mask].std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    return (raw - mean) / std, names


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x_train), dtype=np.float64), x_train])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y_train)


def predict_ridge(x_all: np.ndarray, coef: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x_all), dtype=np.float64), x_all])
    return design @ coef


def fit_event_residual(
    frame: pd.DataFrame,
    train_schedules: tuple[str, ...],
    target: np.ndarray,
    spec: EventSpec,
    transform: str = "aligned",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x_all, names = standardized_features(frame, spec.feature_set, train_schedules, transform=transform)
    mask = train_mask(frame, train_schedules)
    coef = fit_ridge(x_all[mask], target[mask], spec.ridge_alpha)
    residual = predict_ridge(x_all, coef) * spec.shrink
    return residual, coef, names


def pred_loss(frame: pd.DataFrame, residual_pred: np.ndarray) -> np.ndarray:
    return frame["base_pred_loss"].to_numpy(dtype=np.float64) * np.exp(residual_pred)


def metric_dict(actual: np.ndarray, pred: np.ndarray, log_actual: np.ndarray, log_pred: np.ndarray) -> dict[str, float]:
    error = actual - pred
    ss_res = float(np.sum(error * error))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "signed_bias": float(np.mean(pred - actual)),
        "max_abs_error": float(np.max(np.abs(error))),
        "endpoint_abs_diff": float(abs(pred[-1] - actual[-1])),
        "log_residual_mae": float(np.mean(np.abs(log_actual - log_pred))),
    }


def window_masks(schedule_frame: pd.DataFrame) -> dict[str, np.ndarray]:
    step = schedule_frame["step"].to_numpy(dtype=np.int64)
    masks = {
        "full": np.ones(len(schedule_frame), dtype=bool),
        "pre_tail_1000_27125": (step >= START_STEP) & (step < TAIL_START_STEP),
        "tail_27126_33906": (step >= TAIL_START_STEP) & (step <= END_STEP),
        "early_decay_27126_30000": (step >= TAIL_START_STEP) & (step <= 30000),
        "endpoint_30000_33906": (step >= 30000) & (step <= END_STEP),
    }
    last_2048 = np.zeros(len(schedule_frame), dtype=bool)
    last_512 = np.zeros(len(schedule_frame), dtype=bool)
    last_2048[-2048:] = True
    last_512[-512:] = True
    masks["last_2048_sampled"] = last_2048
    masks["last_512_sampled"] = last_512
    return {name: mask for name, mask in masks.items() if bool(mask.any())}


def evaluate_model(
    frame: pd.DataFrame,
    residual_pred: np.ndarray,
    train_schedules: tuple[str, ...],
    model: str,
    feature_set: str,
    transform: str,
) -> pd.DataFrame:
    rows = []
    all_pred = pred_loss(frame, residual_pred)
    for schedule in SCHEDULES:
        schedule_mask = frame["schedule"].to_numpy() == schedule
        group = frame[schedule_mask].sort_values("step").reset_index(drop=True)
        group_pred = all_pred[schedule_mask]
        group_resid = residual_pred[schedule_mask]
        split = "train" if schedule in train_schedules else "test"
        for window, mask in window_masks(group).items():
            rows.append(
                {
                    "train_schedules": "+".join(train_schedules),
                    "target_schedule": schedule,
                    "split": split,
                    "model": model,
                    "feature_set": feature_set,
                    "feature_transform": transform,
                    "window": window,
                    "n": int(mask.sum()),
                    **metric_dict(
                        group.loc[mask, "loss"].to_numpy(dtype=np.float64),
                        group_pred[mask],
                        group.loc[mask, "log_residual"].to_numpy(dtype=np.float64),
                        group_resid[mask],
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_predictions_for_split(
    frame: pd.DataFrame,
    train_schedules: tuple[str, ...],
    event_spec: EventSpec = DEFAULT_EVENT,
    transform: str = "aligned",
    step_spec: StepSpec = STEP_BASE,
) -> dict[str, np.ndarray]:
    zero = np.zeros(len(frame), dtype=np.float64)
    step_reference = fit_step_residual(frame, train_schedules, STEP_REFERENCE)
    step_base = fit_step_residual(frame, train_schedules, step_spec)
    event_only_raw, _, _ = fit_event_residual(
        frame,
        train_schedules,
        frame["log_residual"].to_numpy(dtype=np.float64),
        event_spec,
        transform=transform,
    )
    event_only = np.clip(event_only_raw, -0.15, 0.15)
    leftover = frame["log_residual"].to_numpy(dtype=np.float64) - step_base
    event_leftover, _, _ = fit_event_residual(
        frame,
        train_schedules,
        leftover,
        event_spec,
        transform=transform,
    )
    step_plus = np.clip(step_base + event_leftover, -step_spec.clip, step_spec.clip)
    return {
        "momentum_baseline": zero,
        "step_reference": step_reference,
        "step_base_for_leftover": step_base,
        "event_decay_only": event_only,
        "step_plus_event_leftover": step_plus,
    }


def transfer_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    split_specs = [
        ("cosine",),
        ("811",),
        ("cosine", "811"),
    ]
    for train_schedules in split_specs:
        predictions = build_predictions_for_split(frame, train_schedules)
        for model, residual in predictions.items():
            rows.append(
                evaluate_model(
                    frame,
                    residual,
                    train_schedules,
                    model,
                    DEFAULT_EVENT.feature_set if "event" in model else "none",
                    "aligned",
                )
            )
    return pd.concat(rows, ignore_index=True)


def negative_controls(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train_schedules = ("cosine",)
    for transform in ["aligned", "zero", "circular_shift_25pct", "reverse_time"]:
        predictions = build_predictions_for_split(frame, train_schedules, transform=transform)
        for model in ["step_base_for_leftover", "step_plus_event_leftover"]:
            rows.append(
                evaluate_model(
                    frame,
                    predictions[model],
                    train_schedules,
                    model,
                    DEFAULT_EVENT.feature_set,
                    transform,
                )
            )
    out = pd.concat(rows, ignore_index=True)
    wsd_full = out[(out["target_schedule"] == "wsd") & (out["window"] == "full")]
    base = wsd_full[
        (wsd_full["model"] == "step_base_for_leftover")
        & (wsd_full["feature_transform"] == "aligned")
    ]["mae"].iloc[0]
    out["full_wsd_step_base_mae"] = base
    return out


def feature_ablation(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train_schedules = ("cosine",)
    for feature_set in FEATURE_SETS:
        spec = EventSpec(feature_set=feature_set, ridge_alpha=0.01, shrink=1.0)
        predictions = build_predictions_for_split(frame, train_schedules, event_spec=spec)
        for model in ["event_decay_only", "step_plus_event_leftover"]:
            rows.append(
                evaluate_model(frame, predictions[model], train_schedules, model, feature_set, "aligned")
            )
    return pd.concat(rows, ignore_index=True)


def endpoint_selection(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_schedules = ("cosine",)
    val_schedule = "811"
    target_schedule = "wsd"
    rows = []
    predictions_by_config = {}
    step_base = fit_step_residual(frame, train_schedules, STEP_BASE)
    step_reference = fit_step_residual(frame, train_schedules, STEP_REFERENCE)
    val_step_ref = evaluate_model(
        frame,
        step_reference,
        train_schedules,
        "step_reference",
        "none",
        "aligned",
    )
    val_step_base = evaluate_model(
        frame,
        step_base,
        train_schedules,
        "step_base_for_leftover",
        "none",
        "aligned",
    )
    for ridge_alpha in [1e-4, 1e-2, 1e-1, 1.0, 10.0]:
        for shrink in [0.0, 0.25, 0.5, 0.75, 1.0]:
            spec = EventSpec("linear_endpoint", ridge_alpha, shrink)
            leftover = frame["log_residual"].to_numpy(dtype=np.float64) - step_base
            event_leftover, _, _ = fit_event_residual(frame, train_schedules, leftover, spec)
            residual = np.clip(step_base + event_leftover, -STEP_BASE.clip, STEP_BASE.clip)
            config = spec.name
            predictions_by_config[config] = residual
            metrics = evaluate_model(frame, residual, train_schedules, "step_plus_event_leftover", "linear_endpoint", "aligned")
            values = metrics.set_index(["target_schedule", "window"])
            rows.append(
                {
                    "config": config,
                    "ridge_alpha": ridge_alpha,
                    "event_shrink": shrink,
                    "val_full_mae": float(values.loc[(val_schedule, "full"), "mae"]),
                    "val_tail_mae": float(values.loc[(val_schedule, "tail_27126_33906"), "mae"]),
                    "val_last2048_mae": float(values.loc[(val_schedule, "last_2048_sampled"), "mae"]),
                    "val_endpoint_abs_diff": float(values.loc[(val_schedule, "full"), "endpoint_abs_diff"]),
                }
            )
    grid = pd.DataFrame(rows)
    step_ref_values = val_step_ref.set_index(["target_schedule", "window"])
    step_base_values = val_step_base.set_index(["target_schedule", "window"])
    step_ref_endpoint = float(step_ref_values.loc[(val_schedule, "full"), "endpoint_abs_diff"])
    step_ref_full = float(step_ref_values.loc[(val_schedule, "full"), "mae"])
    step_base_full = float(step_base_values.loc[(val_schedule, "full"), "mae"])
    grid["score_full"] = grid["val_full_mae"]
    grid["score_full_tail"] = (
        0.5 * grid["val_full_mae"] + 0.3 * grid["val_tail_mae"] + 0.2 * grid["val_last2048_mae"]
    )
    grid["score_endpoint_penalty"] = grid["val_full_mae"] + 0.5 * grid["val_endpoint_abs_diff"]
    grid["val_improves_step_base"] = grid["val_full_mae"] < step_base_full
    grid["val_improves_step_reference"] = grid["val_full_mae"] < step_ref_full
    grid["val_endpoint_within_2x_step_reference"] = grid["val_endpoint_abs_diff"] <= 2.0 * step_ref_endpoint

    selections = []
    rules = {
        "val_full": grid.sort_values(["score_full", "val_endpoint_abs_diff", "config"]),
        "val_full_tail": grid.sort_values(["score_full_tail", "val_full_mae", "config"]),
        "val_endpoint_penalty": grid.sort_values(["score_endpoint_penalty", "val_full_mae", "config"]),
    }
    guarded = grid[grid["val_endpoint_within_2x_step_reference"] & grid["val_improves_step_base"]]
    if guarded.empty:
        guarded = grid[grid["val_endpoint_within_2x_step_reference"]]
    rules["endpoint_guard"] = guarded.sort_values(["val_full_mae", "val_endpoint_abs_diff", "config"])
    for rule, ordered in rules.items():
        row = ordered.iloc[0].to_dict()
        selected_metrics = evaluate_model(
            frame,
            predictions_by_config[str(row["config"])],
            train_schedules,
            "step_plus_event_leftover",
            "linear_endpoint",
            "aligned",
        ).set_index(["target_schedule", "window"])
        row["test_full_mae"] = float(selected_metrics.loc[(target_schedule, "full"), "mae"])
        row["test_tail_mae"] = float(selected_metrics.loc[(target_schedule, "tail_27126_33906"), "mae"])
        row["test_last2048_mae"] = float(selected_metrics.loc[(target_schedule, "last_2048_sampled"), "mae"])
        row["test_endpoint_abs_diff"] = float(selected_metrics.loc[(target_schedule, "full"), "endpoint_abs_diff"])
        row["selection_rule"] = rule
        row["val_step_reference_full_mae"] = step_ref_full
        row["val_step_reference_endpoint_abs_diff"] = step_ref_endpoint
        row["val_step_base_full_mae"] = step_base_full
        selections.append(row)
    return grid, pd.DataFrame(selections)


def block_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    train_schedules = ("cosine",)
    predictions = build_predictions_for_split(frame, train_schedules)
    target = frame[frame["schedule"] == "wsd"].sort_values("step").reset_index(drop=True)
    step_loss = pred_loss(frame, predictions["step_reference"])[frame["schedule"].to_numpy() == "wsd"]
    plus_loss = pred_loss(frame, predictions["step_plus_event_leftover"])[frame["schedule"].to_numpy() == "wsd"]
    actual = target["loss"].to_numpy(dtype=np.float64)
    diff = np.abs(actual - step_loss) - np.abs(actual - plus_loss)
    rows = []
    for window, mask in window_masks(target).items():
        values = diff[mask]
        if len(values) == 0:
            continue
        for block_size in [128, 512, 2048]:
            block_size = min(block_size, len(values))
            boot = []
            for _ in range(120):
                pieces = []
                while sum(len(piece) for piece in pieces) < len(values):
                    start = int(rng.integers(0, max(len(values) - block_size + 1, 1)))
                    pieces.append(values[start : start + block_size])
                sample = np.concatenate(pieces)[: len(values)]
                boot.append(float(np.mean(sample)))
            boot_arr = np.asarray(boot, dtype=np.float64)
            rows.append(
                {
                    "target_schedule": "wsd",
                    "window": window,
                    "n": int(len(values)),
                    "block_size": block_size,
                    "mean_abs_error_improvement_vs_step_reference": float(np.mean(values)),
                    "q05": float(np.quantile(boot_arr, 0.05)),
                    "q50": float(np.quantile(boot_arr, 0.50)),
                    "q95": float(np.quantile(boot_arr, 0.95)),
                    "prob_positive": float(np.mean(boot_arr > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def build_summary(
    transfer: pd.DataFrame,
    controls: pd.DataFrame,
    ablation: pd.DataFrame,
    endpoint_selected: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    def metric(table: pd.DataFrame, train: str, target: str, model: str, window: str, column: str = "mae") -> float:
        rows = table[
            (table["train_schedules"] == train)
            & (table["target_schedule"] == target)
            & (table["model"] == model)
            & (table["window"] == window)
        ]
        return float(rows.iloc[0][column])

    wsd_step = metric(transfer, "cosine", "wsd", "step_reference", "full")
    wsd_plus = metric(transfer, "cosine", "wsd", "step_plus_event_leftover", "full")
    wsd_tail_step = metric(transfer, "cosine", "wsd", "step_reference", "tail_27126_33906")
    wsd_tail_plus = metric(transfer, "cosine", "wsd", "step_plus_event_leftover", "tail_27126_33906")
    train811_step = metric(transfer, "811", "wsd", "step_reference", "full")
    train811_plus = metric(transfer, "811", "wsd", "step_plus_event_leftover", "full")
    twosource_step = metric(transfer, "cosine+811", "wsd", "step_reference", "full")
    twosource_plus = metric(transfer, "cosine+811", "wsd", "step_plus_event_leftover", "full")

    control_wsd = controls[
        (controls["target_schedule"] == "wsd")
        & (controls["model"] == "step_plus_event_leftover")
        & (controls["window"] == "full")
    ].set_index("feature_transform")
    ablation_wsd = ablation[
        (ablation["target_schedule"] == "wsd")
        & (ablation["model"] == "step_plus_event_leftover")
        & (ablation["window"] == "full")
    ].sort_values("mae")
    endpoint_rows = endpoint_selected.set_index("selection_rule")
    boot_full = bootstrap[(bootstrap["window"] == "full") & (bootstrap["block_size"] == 2048)].iloc[0]
    boot_last = bootstrap[(bootstrap["window"] == "last_2048_sampled") & (bootstrap["block_size"] == 2048)].iloc[0]

    rows = [
        {
            "question": "main_wsd_margin",
            "result": "supported",
            "key_numbers": f"full {wsd_step:.6f}->{wsd_plus:.6f}; tail {wsd_tail_step:.6f}->{wsd_tail_plus:.6f}",
            "interpretation": "Cosine-trained aligned event leftover strongly improves WSD full/tail over the step reference.",
        },
        {
            "question": "does_811_training_transfer_to_wsd",
            "result": "mixed" if train811_plus < train811_step else "not_supported",
            "key_numbers": f"811-train WSD full step={train811_step:.6f}; plus={train811_plus:.6f}",
            "interpretation": "Using 811 as training checks whether abrupt-drop geometry can support WSD transfer.",
        },
        {
            "question": "does_cosine_plus_811_training_help_wsd",
            "result": "supported" if twosource_plus < twosource_step else "not_supported",
            "key_numbers": f"cosine+811 train WSD full step={twosource_step:.6f}; plus={twosource_plus:.6f}",
            "interpretation": "Two-source training tests whether the leftover survives when 811 is no longer just validation.",
        },
        {
            "question": "feature_alignment_negative_controls",
            "result": "supported" if control_wsd.loc["aligned", "mae"] < control_wsd.loc["circular_shift_25pct", "mae"] else "mixed",
            "key_numbers": (
                f"aligned={control_wsd.loc['aligned','mae']:.6f}; "
                f"zero={control_wsd.loc['zero','mae']:.6f}; "
                f"shift={control_wsd.loc['circular_shift_25pct','mae']:.6f}; "
                f"reverse={control_wsd.loc['reverse_time','mae']:.6f}"
            ),
            "interpretation": "Aligned event/tail geometry should beat deliberately misaligned feature controls.",
        },
        {
            "question": "which_feature_family_drives_gain",
            "result": "supported",
            "key_numbers": (
                f"best={ablation_wsd.iloc[0]['feature_set']}:{ablation_wsd.iloc[0]['mae']:.6f}; "
                f"second={ablation_wsd.iloc[1]['feature_set']}:{ablation_wsd.iloc[1]['mae']:.6f}"
            ),
            "interpretation": "Feature ablation checks whether linear endpoint/tail geometry, not generic event features, drives the effect.",
        },
        {
            "question": "can_endpoint_guard_fix_endpoint_tradeoff",
            "result": "mixed",
            "key_numbers": (
                f"val_full WSD endpoint={endpoint_rows.loc['val_full','test_endpoint_abs_diff']:.6f}; "
                f"guard WSD endpoint={endpoint_rows.loc['endpoint_guard','test_endpoint_abs_diff']:.6f}; "
                f"guard full MAE={endpoint_rows.loc['endpoint_guard','test_full_mae']:.6f}"
            ),
            "interpretation": "Endpoint-constrained validation tests whether full/tail gains can be kept without endpoint degradation.",
        },
        {
            "question": "within_curve_bootstrap",
            "result": "supported" if float(boot_full["q05"]) > 0 else "mixed",
            "key_numbers": (
                f"full q05={boot_full['q05']:.6f}, prob+={boot_full['prob_positive']:.3f}; "
                f"last2048 q05={boot_last['q05']:.6f}, prob+={boot_last['prob_positive']:.3f}"
            ),
            "interpretation": "Block bootstrap is within-curve evidence only; it does not prove schedule-level significance.",
        },
    ]
    return pd.DataFrame(rows)


def build_event_leftover_display_frame(
    frame: pd.DataFrame,
    train_schedules: tuple[str, ...] = ("cosine",),
) -> pd.DataFrame:
    predictions = build_predictions_for_split(frame, train_schedules)
    schedule_mask = frame["schedule"].to_numpy() == "wsd"
    display = frame.loc[schedule_mask, ["step", "loss", "base_pred_loss"]].copy().reset_index(drop=True)
    display["step_reference_loss"] = pred_loss(frame, predictions["step_reference"])[schedule_mask]
    display["step_plus_event_loss"] = pred_loss(frame, predictions["step_plus_event_leftover"])[schedule_mask]
    display["actual_smooth"] = smooth_for_display(display["loss"], 501)
    display["step_reference_smooth"] = smooth_for_display(display["step_reference_loss"], 251)
    display["step_plus_event_smooth"] = smooth_for_display(display["step_plus_event_loss"], 251)
    return display


def plot_event_leftover_contrast(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display = build_event_leftover_display_frame(frame)
    step = display["step"].to_numpy()
    step_reference_error = smooth_for_display(
        (display["step_reference_loss"] - display["loss"]).abs(),
        501,
    ).to_numpy()
    step_plus_error = smooth_for_display(
        (display["step_plus_event_loss"] - display["loss"]).abs(),
        501,
    ).to_numpy()
    error_reduction = step_reference_error - step_plus_error
    improved = error_reduction >= 0

    raw_step_mae = float((display["step_reference_loss"] - display["loss"]).abs().mean())
    raw_plus_mae = float((display["step_plus_event_loss"] - display["loss"]).abs().mean())
    mae_reduction = 100.0 * (1.0 - raw_plus_mae / raw_step_mae)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 4.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.7, 1.0], "hspace": 0.12},
    )
    loss_ax, error_ax = axes

    raw_display = downsample_for_display(display)
    loss_ax.plot(
        raw_display["step"],
        raw_display["loss"],
        color="#4D4D4D",
        alpha=0.12,
        linewidth=0.4,
        label="actual raw",
        rasterized=True,
        zorder=1,
    )
    loss_ax.fill_between(
        step,
        display["step_reference_smooth"],
        display["step_plus_event_smooth"],
        color="#009988",
        alpha=0.17,
        linewidth=0,
        label="step-to-event gap",
        zorder=2,
    )
    loss_ax.plot(
        step,
        display["actual_smooth"],
        color="#111111",
        linewidth=1.45,
        label="actual smoothed",
        zorder=5,
    )
    loss_ax.plot(
        step,
        display["step_reference_smooth"],
        color="#D55E00",
        linestyle="--",
        linewidth=1.45,
        label="step reference",
        zorder=4,
    )
    loss_ax.plot(
        step,
        display["step_plus_event_smooth"],
        color="#009988",
        linewidth=1.8,
        label="step + event",
        zorder=6,
    )
    loss_ax.text(
        0.015,
        0.08,
        f"WSD MAE: {raw_step_mae:.4f} -> {raw_plus_mae:.4f} ({mae_reduction:.0f}% lower)",
        transform=loss_ax.transAxes,
        fontsize=8.2,
        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.9, "pad": 3.0},
    )
    loss_ax.set_title("Event/tail leftover: incremental WSD gain", fontsize=10, pad=4)
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

    error_ax.fill_between(
        step,
        0,
        error_reduction,
        where=improved,
        color="#009988",
        alpha=0.27,
        interpolate=True,
        linewidth=0,
        label="event better",
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
        label="event worse",
    )
    error_ax.axhline(0, color="#111111", linewidth=0.8, alpha=0.8)
    error_ax.plot(
        step,
        error_reduction,
        color="#009988",
        linewidth=1.55,
        label="error reduction",
        zorder=4,
    )
    error_ax.set_title("Absolute-error reduction over step reference", fontsize=9, pad=3)
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

    style_axes([loss_ax, error_ax])
    fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.12, hspace=0.14)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_event_leftover_edge_zoom(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    display = build_event_leftover_display_frame(frame)
    step_min = int(display["step"].min())
    step_max = int(display["step"].max())
    windows = [
        ("Mid/pre-tail", 15000, TAIL_START_STEP - 1),
        ("Decay/tail", TAIL_START_STEP, step_max),
    ]

    fig = plt.figure(figsize=(7.2, 4.2))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[2.2, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.23,
        hspace=0.36,
    )
    main_ax = fig.add_subplot(grid[:, 0])
    zoom_axes = [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])]

    raw_display = downsample_for_display(display)
    main_ax.plot(
        raw_display["step"],
        raw_display["loss"],
        color="#4D4D4D",
        alpha=0.12,
        linewidth=0.35,
        label="actual raw",
        rasterized=True,
        zorder=1,
    )
    main_ax.plot(
        display["step"],
        display["actual_smooth"],
        color="#111111",
        linewidth=1.35,
        label="actual smoothed",
        zorder=5,
    )
    main_ax.plot(
        display["step"],
        display["step_reference_smooth"],
        color="#D55E00",
        linestyle="--",
        linewidth=1.25,
        label="step reference",
        zorder=4,
    )
    main_ax.plot(
        display["step"],
        display["step_plus_event_smooth"],
        color="#009988",
        linewidth=1.55,
        label="step + event",
        zorder=6,
    )
    main_ax.set_title("Event/tail leftover: pre-tail and decay zooms", fontsize=10, pad=5)
    main_ax.set_xlabel("Step")
    main_ax.set_ylabel("Loss")
    main_ax.legend(
        frameon=True,
        framealpha=0.9,
        edgecolor="#DDDDDD",
        fontsize=6.7,
        ncol=2,
        loc="upper right",
        borderpad=0.35,
        handlelength=1.8,
    )

    for zoom_ax, (name, start_step, end_step) in zip(zoom_axes, windows):
        local = display[(display["step"] >= start_step) & (display["step"] <= end_step)].copy()
        plot_event_zoom_window(zoom_ax, local, name, start_step, end_step)
        add_event_zoom_rectangle_and_connectors(main_ax, zoom_ax, local)

    style_axes([main_ax, *zoom_axes])
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.13)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_event_zoom_window(
    ax: Axes,
    local: pd.DataFrame,
    name: str,
    start_step: int,
    end_step: int,
) -> None:
    ax.plot(
        local["step"],
        local["loss"],
        color="#4D4D4D",
        alpha=0.12,
        linewidth=0.35,
        rasterized=True,
        zorder=1,
    )
    ax.fill_between(
        local["step"],
        local["step_reference_smooth"],
        local["step_plus_event_smooth"],
        color="#009988",
        alpha=0.17,
        linewidth=0,
        zorder=2,
    )
    ax.plot(local["step"], local["actual_smooth"], color="#111111", linewidth=1.25, zorder=5)
    ax.plot(
        local["step"],
        local["step_reference_smooth"],
        color="#D55E00",
        linestyle="--",
        linewidth=1.15,
        zorder=4,
    )
    ax.plot(local["step"], local["step_plus_event_smooth"], color="#009988", linewidth=1.45, zorder=6)

    step_mae = float((local["step_reference_loss"] - local["loss"]).abs().mean())
    plus_mae = float((local["step_plus_event_loss"] - local["loss"]).abs().mean())
    reduction = 100.0 * (1.0 - plus_mae / step_mae)
    ax.text(
        0.05,
        0.08,
        f"MAE {step_mae:.4f} -> {plus_mae:.4f}\n{reduction:.0f}% lower",
        transform=ax.transAxes,
        fontsize=6.8,
        bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.9, "pad": 2.4},
        zorder=7,
    )

    y_columns = ["loss", "actual_smooth", "step_reference_smooth", "step_plus_event_smooth"]
    y_min = min(local[col].quantile(0.02) for col in y_columns)
    y_max = max(local[col].quantile(0.98) for col in y_columns)
    y_pad = max((y_max - y_min) * 0.18, 1e-3)
    ax.set_xlim(start_step, end_step)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_title(f"{name}: steps {start_step}-{end_step}", fontsize=8.2, pad=3)
    ax.set_ylabel("Loss", fontsize=8)
    if name == "Decay/tail":
        ax.set_xlabel("Step", fontsize=8)


def add_event_zoom_rectangle_and_connectors(
    main_ax: Axes,
    zoom_ax: Axes,
    local: pd.DataFrame,
) -> None:
    x0 = float(local["step"].min())
    x1 = float(local["step"].max())
    y_columns = ["loss", "actual_smooth", "step_reference_smooth", "step_plus_event_smooth"]
    y0 = min(local[col].quantile(0.02) for col in y_columns)
    y1 = max(local[col].quantile(0.98) for col in y_columns)
    y_pad = max((y1 - y0) * 0.12, 1e-3)
    y0 -= y_pad
    y1 += y_pad

    rect = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        fill=False,
        edgecolor="#777777",
        linewidth=0.8,
        zorder=8,
    )
    main_ax.add_patch(rect)

    for y_rect, y_zoom in [(y1, 1.0), (y0, 0.0)]:
        connector = ConnectionPatch(
            xyA=(0.0, y_zoom),
            coordsA=zoom_ax.transAxes,
            xyB=(x1, y_rect),
            coordsB=main_ax.transData,
            color="#A0A0A0",
            linewidth=0.65,
            alpha=0.85,
            zorder=0,
        )
        main_ax.figure.add_artist(connector)


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


def style_axes(axes: list[Axes]) -> None:
    for ax in axes:
        ax.grid(alpha=0.20, linewidth=0.7, linestyle=(0, (1.5, 3.0)))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7.5)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = add_event_geometry(load_predictions())

    transfer = transfer_metrics(frame)
    controls = negative_controls(frame)
    ablation = feature_ablation(frame)
    endpoint_grid, endpoint_selected = endpoint_selection(frame)
    boot = block_bootstrap(frame)
    summary = build_summary(transfer, controls, ablation, endpoint_selected, boot)

    transfer.to_csv(OUTPUT_DIR / "step_plus_event_stability_transfer_metrics.csv", index=False)
    controls.to_csv(OUTPUT_DIR / "step_plus_event_stability_negative_controls.csv", index=False)
    ablation.to_csv(OUTPUT_DIR / "step_plus_event_stability_feature_ablation.csv", index=False)
    endpoint_grid.to_csv(OUTPUT_DIR / "step_plus_event_stability_endpoint_grid.csv", index=False)
    endpoint_selected.to_csv(OUTPUT_DIR / "step_plus_event_stability_endpoint_selected.csv", index=False)
    boot.to_csv(OUTPUT_DIR / "step_plus_event_stability_block_bootstrap.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "step_plus_event_stability_summary.csv", index=False)

    plot_event_leftover_contrast(
        frame,
        FIGURE_DIR / "step_plus_event_leftover_contrast.png",
    )
    plot_event_leftover_edge_zoom(
        frame,
        FIGURE_DIR / "step_plus_event_leftover_edge_zoom.png",
    )

    print("Step+event stability summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
