from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


PROJECT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PROJECT_DIR.parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"
MOMENTUM_PREDICTIONS = PROJECT_ROOT / "results" / "intermediates" / "three_schedule_momentum" / "predictions.csv"

TRAIN_SCHEDULE = "cosine"
VALIDATION_SCHEDULE = "811"
TEST_SCHEDULE = "wsd"
SCHEDULES = [TRAIN_SCHEDULE, VALIDATION_SCHEDULE, TEST_SCHEDULE]
START_STEP = 1000
END_STEP = 33906
SAMPLE_INTERVAL = 2
TAIL_START_STEP = 27126


@dataclass(frozen=True)
class EventConfig:
    feature_set: str
    ridge_alpha: float
    shrink: float
    residual_clip: float

    @property
    def name(self) -> str:
        return (
            f"{self.feature_set}_ridge{self.ridge_alpha:g}"
            f"_shrink{self.shrink:g}_clip{self.residual_clip:g}"
        )


@dataclass(frozen=True)
class StepConfig:
    kind: str
    smoothing: float | None
    rolling_window: int | None
    shrink: float
    residual_clip: float

    @property
    def name(self) -> str:
        if self.kind == "spline":
            body = f"spline_s{self.smoothing:g}"
        elif self.kind == "interp_roll":
            body = f"interp_roll{self.rolling_window}"
        else:
            raise ValueError(f"Unknown step template kind: {self.kind}")
        return f"{body}_shrink{self.shrink:g}_clip{self.residual_clip:g}"


@dataclass(frozen=True)
class StepEventConfig:
    step: StepConfig
    event: EventConfig

    @property
    def name(self) -> str:
        return f"step[{self.step.name}]_event[{self.event.name}]"


SELECTED_EVENT_ONLY_BY_RULE = {
    "811_full": EventConfig("linear_endpoint", 10.0, 0.75, 0.15),
    "811_full_plus_tail": EventConfig("linear_endpoint", 0.01, 0.25, 0.15),
}
SELECTED_STEP_REFERENCE = StepConfig("spline", 0.05, None, 1.0, 0.15)
SELECTED_STEP_PLUS_EVENT_STEP = StepConfig("spline", 0.05, None, 0.75, 0.15)
SELECTED_STEP_PLUS_EVENT_EVENT = EventConfig("linear_endpoint", 0.01, 1.0, 0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate event/tail leftover residual models.")
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help=(
            "Regenerate only the selected event/tail metrics used by the report "
            "verification. Run without this flag to rebuild the full candidate grid."
        ),
    )
    return parser.parse_args()


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

TRIAL_EVAL_CACHE: dict[tuple[str, str], dict[str, np.ndarray]] | None = None


def load_predictions() -> pd.DataFrame:
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
    frame["log_residual"] = np.log(frame["loss"].to_numpy(dtype=np.float64)) - np.log(
        np.maximum(frame["base_pred_loss"].to_numpy(dtype=np.float64), 1e-12)
    )
    return frame.sort_values(["schedule", "step"]).reset_index(drop=True)


def decayed_drop_history(steps: np.ndarray, drop: np.ndarray, half_life_steps: float) -> np.ndarray:
    history = np.zeros(len(drop), dtype=np.float64)
    state = 0.0
    previous_step = float(steps[0]) if len(steps) else 0.0
    for idx, (step, value) in enumerate(zip(steps, drop)):
        if idx > 0:
            state *= float(np.exp(-(float(step) - previous_step) / half_life_steps))
        state += float(value)
        history[idx] = state
        previous_step = float(step)
    return history


def age_since_event(steps: np.ndarray, event_mask: np.ndarray) -> np.ndarray:
    age = np.zeros(len(steps), dtype=np.float64)
    last_event_step: float | None = None
    for idx, step in enumerate(steps):
        if bool(event_mask[idx]):
            last_event_step = float(step)
            age[idx] = 0.0
        elif last_event_step is not None:
            age[idx] = float(step) - last_event_step
        else:
            age[idx] = 0.0
    return age


def add_event_geometry(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    total_step_span = max(float(END_STEP - START_STEP), 1.0)
    tail_span = max(float(END_STEP - TAIL_START_STEP), 1.0)
    for schedule, group in frame.groupby("schedule", sort=False):
        group = group.sort_values("step").copy()
        steps = group["step"].to_numpy(dtype=np.float64)
        lr = group["lr"].to_numpy(dtype=np.float64)
        s1 = group["s1"].to_numpy(dtype=np.float64)
        drop = np.maximum(np.r_[0.0, lr[:-1] - lr[1:]], 0.0)
        drop_mass = np.cumsum(drop, dtype=np.float64)
        total_drop = max(float(drop_mass[-1]), float(np.max(lr) - np.min(lr)), 1e-12)
        positive_drop = drop > max(total_drop * 1e-8, 1e-12)
        large_drop = drop > (0.01 * total_drop)

        first_drop_idx = int(np.argmax(positive_drop)) if np.any(positive_drop) else len(group)
        first_drop_step = float(steps[first_drop_idx]) if first_drop_idx < len(group) else np.nan
        if first_drop_idx < len(group):
            post_first_age = np.maximum(steps - first_drop_step, 0.0)
            s1_after_first = np.maximum(s1 - s1[first_drop_idx], 0.0)
            after_first = (steps >= first_drop_step).astype(np.float64)
        else:
            post_first_age = np.zeros(len(group), dtype=np.float64)
            s1_after_first = np.zeros(len(group), dtype=np.float64)
            after_first = np.zeros(len(group), dtype=np.float64)

        post_drop_age = age_since_event(steps, positive_drop)
        large_drop_age = age_since_event(steps, large_drop)
        linear_progress = np.clip((lr[0] - lr) / total_drop, 0.0, 1.0)
        tail_progress = np.clip((steps - TAIL_START_STEP) / tail_span, 0.0, 1.0)
        endpoint_progress = np.clip((steps - START_STEP) / total_step_span, 0.0, 1.0)
        tail_saturation = 1.0 - np.exp(-post_first_age / 2048.0)
        remaining_frac = np.clip((END_STEP - steps) / total_step_span, 0.0, 1.0)

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
        group["endpoint_pressure"] = endpoint_progress * endpoint_progress * endpoint_progress
        group["remaining_frac"] = remaining_frac
        group["tail_progress_global"] = tail_progress
        group["tail_progress_global_sq"] = tail_progress * tail_progress
        group["lr_drop_frac"] = drop / total_drop
        group["sqrt_lr_drop_frac"] = np.sqrt(np.maximum(drop / total_drop, 0.0))
        group["drop_mass_frac"] = drop_mass / total_drop
        group["post_drop_age_frac"] = post_drop_age / total_step_span
        group["post_first_drop_age_frac"] = post_first_age / total_step_span
        group["s1_after_first_drop_frac"] = s1_after_first / max(float(np.max(s1_after_first)), 1e-12)
        group["large_drop_age_frac"] = large_drop_age / total_step_span

        for half_life in [256.0, 512.0, 2048.0, 4096.0]:
            name = f"ewma_drop_hl{int(half_life)}"
            group[name] = decayed_drop_history(steps, drop, half_life) / total_drop

        group["drop_mass_x_tail_sat"] = group["drop_mass_frac"] * group["tail_saturation"]
        group["ewma_hl2048_x_tail_sat"] = group["ewma_drop_hl2048"] * group["tail_saturation"]
        group["linear_decay_x_endpoint"] = group["linear_decay_progress"] * group["endpoint_pressure"]
        group["tail_event_pressure"] = group["tail_saturation"] * group["endpoint_pressure"]
        parts.append(group)
    return pd.concat(parts, ignore_index=True).sort_values(["schedule", "step"]).reset_index(drop=True)


def design_matrix(frame: pd.DataFrame, feature_set: str, train_mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    names = FEATURE_SETS[feature_set]
    raw = frame[names].to_numpy(dtype=np.float64)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    mean = raw[train_mask].mean(axis=0)
    std = raw[train_mask].std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (raw - mean) / std, names


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x_train), dtype=np.float64), x_train])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ y_train)


def predict_ridge(x_all: np.ndarray, coef: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x_all), dtype=np.float64), x_all])
    return design @ coef


def fit_step_residual(frame: pd.DataFrame, config: StepConfig) -> np.ndarray:
    train = frame[frame["schedule"] == TRAIN_SCHEDULE].sort_values("step")
    train_steps = train["step"].to_numpy(dtype=np.float64)
    train_residual = train["log_residual"].to_numpy(dtype=np.float64)
    all_steps = frame["step"].to_numpy(dtype=np.float64)

    if config.kind == "spline":
        spline = UnivariateSpline(train_steps, train_residual, s=float(config.smoothing))
        residual = spline(all_steps)
    elif config.kind == "interp_roll":
        smoothed = (
            pd.Series(train_residual)
            .rolling(int(config.rolling_window), center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=np.float64)
        )
        residual = np.interp(all_steps, train_steps, smoothed)
    else:
        raise ValueError(f"Unknown step template kind: {config.kind}")

    residual = residual * config.shrink
    return np.clip(residual, -config.residual_clip, config.residual_clip)


def apply_prediction(frame: pd.DataFrame, residual_pred: np.ndarray) -> np.ndarray:
    base = frame["base_pred_loss"].to_numpy(dtype=np.float64)
    return base * np.exp(residual_pred)


def metric_dict(actual: np.ndarray, pred: np.ndarray, log_actual: np.ndarray, log_pred: np.ndarray) -> dict[str, float]:
    error = actual - pred
    ss_res = float(np.sum(error * error))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mape": float(np.mean(np.abs(error / actual))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "signed_bias": float(np.mean(pred - actual)),
        "max_abs_error": float(np.max(np.abs(error))),
        "endpoint_abs_diff": float(abs(pred[-1] - actual[-1])),
        "log_residual_mae": float(np.mean(np.abs(log_actual - log_pred))),
    }


def window_masks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    step = frame["step"].to_numpy(dtype=np.int64)
    masks = {
        "full": np.ones(len(frame), dtype=bool),
        "steps_20000_30000": (step >= 20000) & (step <= 30000),
        "tail_27126_33906": (step >= TAIL_START_STEP) & (step <= END_STEP),
        "early_decay_27126_30000": (step >= TAIL_START_STEP) & (step <= 30000),
        "endpoint_30000_33906": (step >= 30000) & (step <= END_STEP),
    }
    tail_2048 = np.zeros(len(frame), dtype=bool)
    tail_512 = np.zeros(len(frame), dtype=bool)
    tail_2048[-2048:] = True
    tail_512[-512:] = True
    masks["last_2048_sampled"] = tail_2048
    masks["last_512_sampled"] = tail_512
    return {name: mask for name, mask in masks.items() if bool(mask.any())}


def build_trial_eval_cache(frame: pd.DataFrame) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    schedule_values = frame["schedule"].to_numpy()
    step_values = frame["step"].to_numpy(dtype=np.int64)
    loss_values = frame["loss"].to_numpy(dtype=np.float64)
    log_values = frame["log_residual"].to_numpy(dtype=np.float64)
    cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    split_by_schedule = {
        TRAIN_SCHEDULE: "train",
        VALIDATION_SCHEDULE: "validation",
        TEST_SCHEDULE: "test",
    }
    for schedule, split in split_by_schedule.items():
        schedule_indices = np.flatnonzero(schedule_values == schedule)
        schedule_steps = step_values[schedule_indices]
        masks = {
            "full": np.ones(len(schedule_indices), dtype=bool),
            "tail_27126_33906": (schedule_steps >= TAIL_START_STEP) & (schedule_steps <= END_STEP),
        }
        last_2048 = np.zeros(len(schedule_indices), dtype=bool)
        last_2048[-2048:] = True
        masks["last_2048_sampled"] = last_2048
        for window, mask in masks.items():
            indices = schedule_indices[mask]
            cache[(split, window)] = {
                "indices": indices,
                "loss": loss_values[indices],
                "log_residual": log_values[indices],
            }
    return cache


def evaluate_by_window(
    frame: pd.DataFrame,
    residual_pred: np.ndarray,
    model_class: str,
    config: str,
    selection_rule: str,
) -> pd.DataFrame:
    pred_loss = apply_prediction(frame, residual_pred)
    rows = []
    for schedule in SCHEDULES:
        schedule_mask = (frame["schedule"].to_numpy() == schedule)
        schedule_frame = frame[schedule_mask].sort_values("step").reset_index(drop=True)
        schedule_pred = pred_loss[schedule_mask]
        schedule_resid_pred = residual_pred[schedule_mask]
        split = {
            TRAIN_SCHEDULE: "train",
            VALIDATION_SCHEDULE: "validation",
            TEST_SCHEDULE: "test",
        }[schedule]
        for window_name, mask in window_masks(schedule_frame).items():
            if not bool(mask.any()):
                continue
            actual = schedule_frame.loc[mask, "loss"].to_numpy(dtype=np.float64)
            pred = schedule_pred[mask]
            log_actual = schedule_frame.loc[mask, "log_residual"].to_numpy(dtype=np.float64)
            log_pred = schedule_resid_pred[mask]
            rows.append(
                {
                    "model_class": model_class,
                    "config": config,
                    "selection_rule": selection_rule,
                    "split": split,
                    "schedule": schedule,
                    "window": window_name,
                    "n": int(mask.sum()),
                    **metric_dict(actual, pred, log_actual, log_pred),
                }
            )
    return pd.DataFrame(rows)


def compact_trial_metrics(frame: pd.DataFrame, residual_pred: np.ndarray) -> dict[str, float]:
    if TRIAL_EVAL_CACHE is None:
        raise RuntimeError("TRIAL_EVAL_CACHE was not initialized")
    pred_loss = apply_prediction(frame, residual_pred)

    def value(split: str, window: str, column: str) -> float:
        data = TRIAL_EVAL_CACHE[(split, window)]
        indices = data["indices"]
        actual = data["loss"]
        pred = pred_loss[indices]
        log_actual = data["log_residual"]
        log_pred = residual_pred[indices]
        return float(metric_dict(actual, pred, log_actual, log_pred)[column])

    val_full = value("validation", "full", "mae")
    val_tail = value("validation", "tail_27126_33906", "mae")
    val_last = value("validation", "last_2048_sampled", "mae")
    return {
        "train_full_mae": value("train", "full", "mae"),
        "validation_full_mae": val_full,
        "validation_tail_mae": val_tail,
        "validation_last2048_mae": val_last,
        "validation_full_tail_score": 0.5 * val_full + 0.3 * val_tail + 0.2 * val_last,
        "validation_endpoint_abs_diff": value("validation", "full", "endpoint_abs_diff"),
    }


def candidate_event_configs() -> list[EventConfig]:
    return [
        EventConfig(feature_set, alpha, shrink, clip)
        for feature_set in FEATURE_SETS
        for alpha in [1e-6, 1e-4, 1e-2, 1.0, 10.0]
        for shrink in [0.25, 0.5, 0.75, 1.0, 1.25]
        for clip in [0.15, 0.25]
    ]


def candidate_step_configs() -> list[StepConfig]:
    spline = [
        StepConfig("spline", smoothing, None, shrink, clip)
        for smoothing in [0.05, 0.1, 0.25, 0.5, 1.0]
        for shrink in [0.75, 1.0, 1.25]
        for clip in [0.15, 0.25]
    ]
    interp = [
        StepConfig("interp_roll", None, window, shrink, clip)
        for window in [101, 501, 1001, 2049]
        for shrink in [0.75, 1.0, 1.25]
        for clip in [0.15, 0.25]
    ]
    return spline + interp


def update_best(
    best: dict[tuple[str, str], dict[str, object]],
    model_class: str,
    row: dict[str, object],
    residual_pred: np.ndarray,
    coef_rows: list[dict[str, object]],
) -> None:
    rules = {
        "811_full": (
            float(row["validation_full_mae"]),
            float(row["validation_endpoint_abs_diff"]),
            str(row["config"]),
        ),
        "811_full_plus_tail": (
            float(row["validation_full_tail_score"]),
            float(row["validation_full_mae"]),
            float(row["validation_endpoint_abs_diff"]),
            str(row["config"]),
        ),
    }
    for rule, score in rules.items():
        key = (model_class, rule)
        if key not in best or score < best[key]["score"]:
            best[key] = {
                "score": score,
                "row": row.copy(),
                "residual_pred": residual_pred.copy(),
                "coef_rows": [item.copy() for item in coef_rows],
            }


def coef_table_rows(
    model_class: str,
    config: str,
    target: str,
    ridge_alpha: float,
    shrink: float,
    feature_names: list[str],
    coef: np.ndarray,
    extra: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    base = {
        "model_class": model_class,
        "config": config,
        "target": target,
        "ridge_alpha": ridge_alpha,
        "event_shrink": shrink,
        **(extra or {}),
    }
    rows = [{**base, "feature": "intercept", "coefficient": float(coef[0])}]
    rows.extend(
        {**base, "feature": feature, "coefficient": float(value)}
        for feature, value in zip(feature_names, coef[1:])
    )
    return rows


def run_event_only_candidates(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    feature_cache: dict[str, tuple[np.ndarray, list[str]]],
    best: dict[tuple[str, str], dict[str, object]],
) -> list[dict[str, object]]:
    y_train = frame.loc[train_mask, "log_residual"].to_numpy(dtype=np.float64)
    rows = []
    for config in candidate_event_configs():
        x_all, names = feature_cache[config.feature_set]
        coef = fit_ridge(x_all[train_mask], y_train, config.ridge_alpha)
        residual_pred = predict_ridge(x_all, coef) * config.shrink
        residual_pred = np.clip(residual_pred, -config.residual_clip, config.residual_clip)
        row = {
            "model_class": "event_decay_only",
            "config": config.name,
            "feature_set": config.feature_set,
            "step_template": "none",
            "ridge_alpha": config.ridge_alpha,
            "event_shrink": config.shrink,
            "step_shrink": np.nan,
            "residual_clip": config.residual_clip,
            **compact_trial_metrics(frame, residual_pred),
        }
        coef_rows = coef_table_rows(
            "event_decay_only",
            config.name,
            "log_residual",
            config.ridge_alpha,
            config.shrink,
            names,
            coef,
            {"step_template": "none"},
        )
        update_best(best, "event_decay_only", row, residual_pred, coef_rows)
        rows.append(row)
    return rows


def run_step_reference_candidates(
    frame: pd.DataFrame,
    best: dict[tuple[str, str], dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows = []
    residual_cache = {}
    for config in candidate_step_configs():
        residual_pred = fit_step_residual(frame, config)
        residual_cache[config.name] = residual_pred
        row = {
            "model_class": "step_reference",
            "config": config.name,
            "feature_set": "step_only",
            "step_template": config.kind,
            "ridge_alpha": np.nan,
            "event_shrink": np.nan,
            "step_shrink": config.shrink,
            "residual_clip": config.residual_clip,
            "step_smoothing": config.smoothing,
            "step_rolling_window": config.rolling_window,
            **compact_trial_metrics(frame, residual_pred),
        }
        step_rows = [
            {
                "model_class": "step_reference",
                "config": config.name,
                "target": "log_residual",
                "ridge_alpha": np.nan,
                "event_shrink": np.nan,
                "step_template": config.kind,
                "feature": "step_template",
                "coefficient": np.nan,
                "step_smoothing": config.smoothing,
                "step_rolling_window": config.rolling_window,
                "step_shrink": config.shrink,
                "residual_clip": config.residual_clip,
            }
        ]
        update_best(best, "step_reference", row, residual_pred, step_rows)
        rows.append(row)
    return rows, residual_cache


def run_step_plus_event_candidates(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    feature_cache: dict[str, tuple[np.ndarray, list[str]]],
    step_residual_cache: dict[str, np.ndarray],
    best: dict[tuple[str, str], dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    y = frame["log_residual"].to_numpy(dtype=np.float64)
    event_bases = [
        EventConfig(feature_set, alpha, shrink, 0.25)
        for feature_set in FEATURE_SETS
        for alpha in [1e-4, 1e-2, 1.0, 10.0]
        for shrink in [0.25, 0.5, 0.75, 1.0]
    ]
    step_configs = candidate_step_configs()
    step_config_by_name = {config.name: config for config in step_configs}

    for step_name, step_residual in step_residual_cache.items():
        step_config = step_config_by_name[step_name]
        leftover_train = y[train_mask] - step_residual[train_mask]
        for event_config in event_bases:
            x_all, names = feature_cache[event_config.feature_set]
            coef = fit_ridge(x_all[train_mask], leftover_train, event_config.ridge_alpha)
            event_leftover = predict_ridge(x_all, coef) * event_config.shrink
            event_leftover = np.clip(
                event_leftover,
                -event_config.residual_clip,
                event_config.residual_clip,
            )
            residual_pred = np.clip(
                step_residual + event_leftover,
                -step_config.residual_clip,
                step_config.residual_clip,
            )
            config = StepEventConfig(step_config, event_config)
            row = {
                "model_class": "step_plus_event_leftover",
                "config": config.name,
                "feature_set": event_config.feature_set,
                "step_template": step_config.kind,
                "ridge_alpha": event_config.ridge_alpha,
                "event_shrink": event_config.shrink,
                "step_shrink": step_config.shrink,
                "residual_clip": step_config.residual_clip,
                "step_smoothing": step_config.smoothing,
                "step_rolling_window": step_config.rolling_window,
                **compact_trial_metrics(frame, residual_pred),
            }
            coef_rows = coef_table_rows(
                "step_plus_event_leftover",
                config.name,
                "log_residual_minus_step_reference",
                event_config.ridge_alpha,
                event_config.shrink,
                names,
                coef,
                {
                    "step_template": step_config.kind,
                    "step_config": step_config.name,
                    "step_smoothing": step_config.smoothing,
                    "step_rolling_window": step_config.rolling_window,
                    "step_shrink": step_config.shrink,
                    "residual_clip": step_config.residual_clip,
                },
            )
            update_best(best, "step_plus_event_leftover", row, residual_pred, coef_rows)
            rows.append(row)
    return rows


def feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for schedule, group in frame.groupby("schedule", sort=False):
        group = group.sort_values("step")
        lr_drop = group["lr_drop"].to_numpy(dtype=np.float64)
        positive = lr_drop > max(float(group["drop_mass"].iloc[-1]) * 1e-8, 1e-12)
        large = group["large_drop_event"].to_numpy(dtype=np.float64) > 0
        rows.append(
            {
                "schedule": schedule,
                "n": len(group),
                "min_step": int(group["step"].min()),
                "max_step": int(group["step"].max()),
                "initial_lr": float(group["lr"].iloc[0]),
                "final_lr": float(group["lr"].iloc[-1]),
                "total_drop_mass": float(group["drop_mass"].iloc[-1]),
                "max_lr_drop": float(group["lr_drop"].max()),
                "positive_drop_samples": int(np.sum(positive)),
                "large_drop_events": int(np.sum(large)),
                "first_positive_drop_step": (
                    int(group.loc[positive, "step"].iloc[0]) if bool(np.any(positive)) else np.nan
                ),
                "first_large_drop_step": (
                    int(group.loc[large, "step"].iloc[0]) if bool(np.any(large)) else np.nan
                ),
                "final_linear_decay_progress": float(group["linear_decay_progress"].iloc[-1]),
                "final_post_first_drop_age_steps": float(group["post_first_drop_age_steps"].iloc[-1]),
                "final_s1_after_first_drop": float(group["s1_after_first_drop"].iloc[-1]),
                "final_tail_saturation": float(group["tail_saturation"].iloc[-1]),
                "peak_ewma_drop_hl512": float(group["ewma_drop_hl512"].max()),
                "peak_ewma_drop_hl2048": float(group["ewma_drop_hl2048"].max()),
                "mean_endpoint_pressure": float(group["endpoint_pressure"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_summary_verdict(
    selected_metrics: pd.DataFrame,
    trials: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    metric_key = selected_metrics.set_index(["selection_rule", "model_class", "split", "window"])
    for rule in ["811_full", "811_full_plus_tail"]:
        baseline_full = float(metric_key.loc[(rule, "momentum_baseline", "test", "full"), "mae"])
        baseline_tail = float(metric_key.loc[(rule, "momentum_baseline", "test", "tail_27126_33906"), "mae"])
        step_full = float(metric_key.loc[(rule, "step_reference", "test", "full"), "mae"])
        step_tail = float(metric_key.loc[(rule, "step_reference", "test", "tail_27126_33906"), "mae"])
        event_full = float(metric_key.loc[(rule, "event_decay_only", "test", "full"), "mae"])
        event_tail = float(metric_key.loc[(rule, "event_decay_only", "test", "tail_27126_33906"), "mae"])
        plus_full = float(metric_key.loc[(rule, "step_plus_event_leftover", "test", "full"), "mae"])
        plus_tail = float(metric_key.loc[(rule, "step_plus_event_leftover", "test", "tail_27126_33906"), "mae"])

        val_step = float(metric_key.loc[(rule, "step_reference", "validation", "full"), "mae"])
        val_plus = float(metric_key.loc[(rule, "step_plus_event_leftover", "validation", "full"), "mae"])
        margin_full = step_full - plus_full
        margin_tail = step_tail - plus_tail
        if margin_full > 1e-4 and val_step - val_plus > 1e-4:
            verdict = "step_plus_event adds positive validation-selected WSD margin over the step template"
        elif event_full < baseline_full:
            verdict = "event-local history helps over momentum but does not add reliable margin beyond the step template"
        else:
            verdict = "event-local history fails to beat the momentum baseline and does not add beyond the step template"

        rows.append(
            {
                "selection_rule": rule,
                "n_trials": int(len(trials)),
                "wsd_momentum_full_mae": baseline_full,
                "wsd_event_only_full_mae": event_full,
                "wsd_step_reference_full_mae": step_full,
                "wsd_step_plus_event_full_mae": plus_full,
                "wsd_event_only_vs_momentum_delta": baseline_full - event_full,
                "wsd_step_plus_vs_step_delta": margin_full,
                "wsd_momentum_tail_mae": baseline_tail,
                "wsd_event_only_tail_mae": event_tail,
                "wsd_step_reference_tail_mae": step_tail,
                "wsd_step_plus_event_tail_mae": plus_tail,
                "wsd_event_only_tail_vs_momentum_delta": baseline_tail - event_tail,
                "wsd_step_plus_tail_vs_step_delta": margin_tail,
                "validation_step_plus_vs_step_full_delta": val_step - val_plus,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def event_residual_and_coefficients(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    feature_cache: dict[str, tuple[np.ndarray, list[str]]],
    config: EventConfig,
    model_class: str,
    target: str,
    extra: dict[str, object] | None = None,
    target_values: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    x_all, names = feature_cache[config.feature_set]
    if target_values is None:
        y_train = frame.loc[train_mask, "log_residual"].to_numpy(dtype=np.float64)
    else:
        y_train = target_values
    coef = fit_ridge(x_all[train_mask], y_train, config.ridge_alpha)
    residual_pred = predict_ridge(x_all, coef) * config.shrink
    residual_pred = np.clip(residual_pred, -config.residual_clip, config.residual_clip)
    rows = coef_table_rows(
        model_class,
        config.name,
        target,
        config.ridge_alpha,
        config.shrink,
        names,
        coef,
        extra,
    )
    return residual_pred, rows


def run_selected_only(frame: pd.DataFrame) -> None:
    train_mask = frame["schedule"].to_numpy() == TRAIN_SCHEDULE
    feature_cache = {
        feature_set: design_matrix(frame, feature_set, train_mask)
        for feature_set in FEATURE_SETS
    }

    selected_parts = []
    coefficient_rows = []
    baseline_residual = np.zeros(len(frame), dtype=np.float64)

    for rule in ["811_full", "811_full_plus_tail"]:
        selected_parts.append(
            evaluate_by_window(frame, baseline_residual, "momentum_baseline", "none", rule)
        )

        event_config = SELECTED_EVENT_ONLY_BY_RULE[rule]
        event_residual, event_coef_rows = event_residual_and_coefficients(
            frame,
            train_mask,
            feature_cache,
            event_config,
            "event_decay_only",
            "log_residual",
            {"step_template": "none"},
        )
        selected_parts.append(
            evaluate_by_window(
                frame,
                event_residual,
                "event_decay_only",
                event_config.name,
                rule,
            )
        )
        coefficient_rows.extend({"selection_rule": rule, **row} for row in event_coef_rows)

        step_residual = fit_step_residual(frame, SELECTED_STEP_REFERENCE)
        selected_parts.append(
            evaluate_by_window(
                frame,
                step_residual,
                "step_reference",
                SELECTED_STEP_REFERENCE.name,
                rule,
            )
        )
        coefficient_rows.append(
            {
                "selection_rule": rule,
                "model_class": "step_reference",
                "config": SELECTED_STEP_REFERENCE.name,
                "target": "log_residual",
                "ridge_alpha": np.nan,
                "event_shrink": np.nan,
                "step_template": SELECTED_STEP_REFERENCE.kind,
                "feature": "step_template",
                "coefficient": np.nan,
                "step_smoothing": SELECTED_STEP_REFERENCE.smoothing,
                "step_rolling_window": SELECTED_STEP_REFERENCE.rolling_window,
                "step_shrink": SELECTED_STEP_REFERENCE.shrink,
                "residual_clip": SELECTED_STEP_REFERENCE.residual_clip,
            }
        )

        plus_step_residual = fit_step_residual(frame, SELECTED_STEP_PLUS_EVENT_STEP)
        leftover_train = (
            frame.loc[train_mask, "log_residual"].to_numpy(dtype=np.float64)
            - plus_step_residual[train_mask]
        )
        event_leftover, plus_coef_rows = event_residual_and_coefficients(
            frame,
            train_mask,
            feature_cache,
            SELECTED_STEP_PLUS_EVENT_EVENT,
            "step_plus_event_leftover",
            "log_residual_minus_step_reference",
            {
                "step_template": SELECTED_STEP_PLUS_EVENT_STEP.kind,
                "step_config": SELECTED_STEP_PLUS_EVENT_STEP.name,
                "step_smoothing": SELECTED_STEP_PLUS_EVENT_STEP.smoothing,
                "step_rolling_window": SELECTED_STEP_PLUS_EVENT_STEP.rolling_window,
                "step_shrink": SELECTED_STEP_PLUS_EVENT_STEP.shrink,
                "residual_clip": SELECTED_STEP_PLUS_EVENT_STEP.residual_clip,
            },
            target_values=leftover_train,
        )
        plus_residual = np.clip(
            plus_step_residual + event_leftover,
            -SELECTED_STEP_PLUS_EVENT_STEP.residual_clip,
            SELECTED_STEP_PLUS_EVENT_STEP.residual_clip,
        )
        plus_config = StepEventConfig(SELECTED_STEP_PLUS_EVENT_STEP, SELECTED_STEP_PLUS_EVENT_EVENT)
        selected_parts.append(
            evaluate_by_window(
                frame,
                plus_residual,
                "step_plus_event_leftover",
                plus_config.name,
                rule,
            )
        )
        coefficient_rows.extend({"selection_rule": rule, **row} for row in plus_coef_rows)

    selected_metrics = pd.concat(selected_parts, ignore_index=True)
    coefficients = pd.DataFrame(coefficient_rows)
    features = feature_summary(frame)

    selected_metrics = selected_metrics.sort_values(
        ["selection_rule", "split", "schedule", "window", "model_class"]
    ).reset_index(drop=True)
    coefficients = coefficients.sort_values(
        ["selection_rule", "model_class", "config", "feature"]
    ).reset_index(drop=True)

    selected_metrics.to_csv(
        OUTPUT_DIR / "sujianlin_event_decay_selected_metrics_by_window.csv",
        index=False,
    )
    coefficients.to_csv(OUTPUT_DIR / "sujianlin_event_decay_coefficients.csv", index=False)
    features.to_csv(OUTPUT_DIR / "sujianlin_event_decay_feature_summary.csv", index=False)

    print("Fixed selected configs:")
    for rule in ["811_full", "811_full_plus_tail"]:
        print(f"{rule} event_decay_only: {SELECTED_EVENT_ONLY_BY_RULE[rule].name}")
        print(f"{rule} step_reference: {SELECTED_STEP_REFERENCE.name}")
        plus_config = StepEventConfig(SELECTED_STEP_PLUS_EVENT_STEP, SELECTED_STEP_PLUS_EVENT_EVENT)
        print(f"{rule} step_plus_event_leftover: {plus_config.name}")

    print("\nWSD headline metrics:")
    report = selected_metrics[
        (selected_metrics["split"] == "test")
        & (selected_metrics["window"].isin(["full", "tail_27126_33906", "last_2048_sampled"]))
    ][
        [
            "selection_rule",
            "model_class",
            "window",
            "mae",
            "rmse",
            "r2",
            "endpoint_abs_diff",
            "log_residual_mae",
        ]
    ].sort_values(["selection_rule", "window", "model_class"])
    print(report.to_string(index=False))


def main() -> None:
    global TRIAL_EVAL_CACHE
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = add_event_geometry(load_predictions())
    if args.selected_only:
        run_selected_only(frame)
        return

    TRIAL_EVAL_CACHE = build_trial_eval_cache(frame)
    train_mask = (frame["schedule"].to_numpy() == TRAIN_SCHEDULE)
    feature_cache = {
        feature_set: design_matrix(frame, feature_set, train_mask)
        for feature_set in FEATURE_SETS
    }

    best: dict[tuple[str, str], dict[str, object]] = {}
    trial_rows: list[dict[str, object]] = []

    trial_rows.extend(run_event_only_candidates(frame, train_mask, feature_cache, best))
    step_rows, step_residual_cache = run_step_reference_candidates(frame, best)
    trial_rows.extend(step_rows)
    trial_rows.extend(run_step_plus_event_candidates(frame, train_mask, feature_cache, step_residual_cache, best))

    trials = pd.DataFrame(trial_rows)
    for rule in ["811_full", "811_full_plus_tail"]:
        trials[f"selected_by_{rule}"] = False
        for model_class in ["event_decay_only", "step_reference", "step_plus_event_leftover"]:
            selected_config = str(best[(model_class, rule)]["row"]["config"])
            mask = (trials["model_class"] == model_class) & (trials["config"] == selected_config)
            trials.loc[mask, f"selected_by_{rule}"] = True

    selected_parts = []
    coefficient_rows = []
    baseline_residual = np.zeros(len(frame), dtype=np.float64)
    for rule in ["811_full", "811_full_plus_tail"]:
        selected_parts.append(
            evaluate_by_window(frame, baseline_residual, "momentum_baseline", "none", rule)
        )
        for model_class in ["event_decay_only", "step_reference", "step_plus_event_leftover"]:
            selected = best[(model_class, rule)]
            selected_parts.append(
                evaluate_by_window(
                    frame,
                    selected["residual_pred"],
                    model_class,
                    str(selected["row"]["config"]),
                    rule,
                )
            )
            for coef_row in selected["coef_rows"]:
                coefficient_rows.append(
                    {
                        "selection_rule": rule,
                        **coef_row,
                    }
                )

    selected_metrics = pd.concat(selected_parts, ignore_index=True)
    coefficients = pd.DataFrame(coefficient_rows)
    features = feature_summary(frame)
    summary = build_summary_verdict(selected_metrics, trials)

    trials = trials.sort_values(
        ["model_class", "validation_full_mae", "validation_full_tail_score", "config"]
    ).reset_index(drop=True)
    selected_metrics = selected_metrics.sort_values(
        ["selection_rule", "split", "schedule", "window", "model_class"]
    ).reset_index(drop=True)
    coefficients = coefficients.sort_values(
        ["selection_rule", "model_class", "config", "feature"]
    ).reset_index(drop=True)

    trials.to_csv(OUTPUT_DIR / "sujianlin_event_decay_trials.csv", index=False)
    selected_metrics.to_csv(
        OUTPUT_DIR / "sujianlin_event_decay_selected_metrics_by_window.csv",
        index=False,
    )
    coefficients.to_csv(OUTPUT_DIR / "sujianlin_event_decay_coefficients.csv", index=False)
    features.to_csv(OUTPUT_DIR / "sujianlin_event_decay_feature_summary.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "sujianlin_event_decay_summary_verdict.csv", index=False)

    print("Selected configs:")
    for rule in ["811_full", "811_full_plus_tail"]:
        for model_class in ["event_decay_only", "step_reference", "step_plus_event_leftover"]:
            row = best[(model_class, rule)]["row"]
            print(f"{rule} {model_class}: {row['config']}")
    print("\nWSD headline metrics:")
    report = selected_metrics[
        (selected_metrics["split"] == "test")
        & (selected_metrics["window"].isin(["full", "tail_27126_33906", "last_2048_sampled"]))
    ][
        [
            "selection_rule",
            "model_class",
            "window",
            "mae",
            "rmse",
            "r2",
            "endpoint_abs_diff",
            "log_residual_mae",
        ]
    ].sort_values(["selection_rule", "window", "model_class"])
    print(report.to_string(index=False))
    print("\nSummary verdict:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))
    main()
