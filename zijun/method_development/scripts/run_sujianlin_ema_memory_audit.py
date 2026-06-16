from __future__ import annotations

import math
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
LOG_DIR = PROJECT_DIR / "experiment_logs"
PROJECT_ROOT = PROJECT_DIR.parents[1]
PREDICTIONS_CSV = PROJECT_ROOT / "results" / "momentum_residual_mlp_results" / "predictions.csv"
STEP_REFERENCE_METRICS_CSV = OUTPUT_DIR / "spline_stability_selected_metrics.csv"

TRAIN_SCHEDULE = "cosine"
VALIDATION_SCHEDULE = "811"
TEST_SCHEDULE = "wsd"
START_STEP = 1000
END_STEP = 33906
SAMPLE_INTERVAL = 2
EPS = 1e-12

HALF_LIFE_STEPS = [64, 256, 1024, 4096, 8192, 16384]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0]
SHRINKS = [0.5, 0.75, 1.0, 1.25]
RESIDUAL_CLIPS = [0.05, 0.10, 0.15, 0.25]

PRIMARY_WINDOWS = [
    "full",
    "steps_27126_33906",
    "steps_30000_33906",
    "last_2048_sampled",
]


EMA_FEATURE_TEMPLATES = {
    "ewma_lr": ["ewma_lr_h{h}"],
    "ewma_lr2": ["ewma_lr2_h{h}"],
    "ewma_positive_lr_drop": ["ewma_positive_lr_drop_h{h}"],
    "ewma_abs_lr_change": ["ewma_abs_lr_change_h{h}"],
    "ewma_lr_lr2": ["ewma_lr_h{h}", "ewma_lr2_h{h}"],
    "ewma_drop_abs_change": ["ewma_positive_lr_drop_h{h}", "ewma_abs_lr_change_h{h}"],
    "ewma_lr_drop": ["ewma_lr_h{h}", "ewma_positive_lr_drop_h{h}"],
    "current_lr_ewma_lr": ["lr_norm", "ewma_lr_h{h}"],
    "current_lr_ewma_lr2": ["lr_norm", "ewma_lr2_h{h}"],
    "current_lr_ewma_abs_change": ["lr_norm", "ewma_abs_lr_change_h{h}"],
    "lr_minus_ewma_lr": ["lr_minus_ewma_lr_h{h}"],
    "lr_memory_gap_pair": ["lr_norm", "lr_minus_ewma_lr_h{h}"],
    "ema_four": [
        "ewma_lr_h{h}",
        "ewma_lr2_h{h}",
        "ewma_positive_lr_drop_h{h}",
        "ewma_abs_lr_change_h{h}",
    ],
}

CONTROL_FEATURE_SETS = {
    "current_lr": ["lr_norm"],
    "current_lr_lr2": ["lr_norm", "lr2_norm"],
    "current_lr_change": ["lr_norm", "positive_lr_drop_norm", "abs_lr_change_norm"],
    "s1_s2_control": ["s1_norm", "s2_norm"],
    "current_lr_s1_s2": ["lr_norm", "s1_norm", "s2_norm"],
}


@dataclass(frozen=True)
class Candidate:
    feature_set: str
    feature_group: str
    feature_names: tuple[str, ...]
    half_life_steps: int | None
    alpha_per_sample: float | None
    ridge_alpha: float
    shrink: float
    residual_clip: float

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def candidate_id(self) -> str:
        h_part = f"h{self.half_life_steps}" if self.half_life_steps is not None else "static"
        return "_".join(
            [
                self.feature_set,
                h_part,
                f"ridge{format_float(self.ridge_alpha)}",
                f"shrink{format_float(self.shrink)}",
                f"clip{format_float(self.residual_clip)}",
            ]
        )


@dataclass(frozen=True)
class FitResult:
    candidate: Candidate
    coef: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    residual_unshrunk: np.ndarray
    residual_pred: np.ndarray
    pred_loss: np.ndarray


def format_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def alpha_from_half_life(half_life_steps: int) -> float:
    return float(1.0 - math.exp(-math.log(2.0) * SAMPLE_INTERVAL / half_life_steps))


def ewma(values: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    one_minus = 1.0 - alpha
    for idx in range(1, len(values)):
        out[idx] = alpha * values[idx] + one_minus * out[idx - 1]
    return out


def load_predictions() -> pd.DataFrame:
    frame = pd.read_csv(PREDICTIONS_CSV)
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
    frame["log_residual"] = (
        np.log(frame["loss"].to_numpy(dtype=np.float64))
        - np.log(np.maximum(frame["base_pred_loss"].to_numpy(dtype=np.float64), EPS))
    )
    return add_schedule_only_features(frame)


def add_schedule_only_features(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for _, group in frame.groupby("schedule", sort=False):
        group = group.sort_values("step").copy()
        lr = group["lr"].to_numpy(dtype=np.float64)
        prev_lr = np.r_[lr[0], lr[:-1]]
        positive_drop = np.maximum(prev_lr - lr, 0.0)
        abs_change = np.abs(lr - prev_lr)

        group["lr2"] = lr * lr
        group["positive_lr_drop"] = positive_drop
        group["abs_lr_change"] = abs_change

        for half_life in HALF_LIFE_STEPS:
            alpha = alpha_from_half_life(half_life)
            lr_ema = ewma(lr, alpha)
            group[f"ewma_lr_h{half_life}"] = lr_ema
            group[f"ewma_lr2_h{half_life}"] = ewma(lr * lr, alpha)
            group[f"ewma_positive_lr_drop_h{half_life}"] = ewma(positive_drop, alpha)
            group[f"ewma_abs_lr_change_h{half_life}"] = ewma(abs_change, alpha)
            group[f"lr_minus_ewma_lr_h{half_life}"] = lr - lr_ema
        pieces.append(group)

    out = pd.concat(pieces, ignore_index=True)
    train = out[out["schedule"] == TRAIN_SCHEDULE]
    lr_scale = max(float(train["lr"].max()), EPS)
    lr2_scale = max(float(train["lr2"].max()), EPS)
    s1_scale = max(float(train["s1"].max()), EPS)
    s2_scale = max(float(train["s2"].max()), EPS)
    positive_drop_scale = max(float(train["positive_lr_drop"].max()), EPS)
    abs_change_scale = max(float(train["abs_lr_change"].max()), EPS)

    out["lr_norm"] = out["lr"] / lr_scale
    out["lr2_norm"] = out["lr2"] / lr2_scale
    out["s1_norm"] = out["s1"] / s1_scale
    out["s2_norm"] = out["s2"] / s2_scale
    out["positive_lr_drop_norm"] = out["positive_lr_drop"] / positive_drop_scale
    out["abs_lr_change_norm"] = out["abs_lr_change"] / abs_change_scale
    return out.sort_values(["schedule", "step"]).reset_index(drop=True)


def build_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    for feature_set, templates in EMA_FEATURE_TEMPLATES.items():
        for half_life in HALF_LIFE_STEPS:
            alpha = alpha_from_half_life(half_life)
            feature_names = tuple(template.format(h=half_life) for template in templates)
            for ridge_alpha in RIDGE_ALPHAS:
                for shrink in SHRINKS:
                    for residual_clip in RESIDUAL_CLIPS:
                        candidates.append(
                            Candidate(
                                feature_set=feature_set,
                                feature_group="ema_memory",
                                feature_names=feature_names,
                                half_life_steps=half_life,
                                alpha_per_sample=alpha,
                                ridge_alpha=ridge_alpha,
                                shrink=shrink,
                                residual_clip=residual_clip,
                            )
                        )

    for feature_set, feature_names in CONTROL_FEATURE_SETS.items():
        for ridge_alpha in RIDGE_ALPHAS:
            for shrink in SHRINKS:
                for residual_clip in RESIDUAL_CLIPS:
                    candidates.append(
                        Candidate(
                            feature_set=feature_set,
                            feature_group="schedule_control",
                            feature_names=tuple(feature_names),
                            half_life_steps=None,
                            alpha_per_sample=None,
                            ridge_alpha=ridge_alpha,
                            shrink=shrink,
                            residual_clip=residual_clip,
                        )
                    )
    return candidates


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge_alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_alpha
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def fit_candidate(
    frame: pd.DataFrame,
    candidate: Candidate,
    x_raw: np.ndarray | None = None,
) -> FitResult:
    train_mask = frame["schedule"].to_numpy() == TRAIN_SCHEDULE
    if x_raw is None:
        x_raw = frame.loc[:, list(candidate.feature_names)].to_numpy(dtype=np.float64)
    x_train_raw = x_raw[train_mask]
    feature_mean = np.mean(x_train_raw, axis=0)
    feature_std = np.std(x_train_raw, axis=0)
    feature_std = np.where(feature_std < EPS, 1.0, feature_std)
    x_scaled = (x_raw - feature_mean) / feature_std
    coef = ridge_fit(
        x_scaled[train_mask],
        frame.loc[train_mask, "log_residual"].to_numpy(dtype=np.float64),
        candidate.ridge_alpha,
    )
    design = np.column_stack([np.ones(len(x_scaled), dtype=np.float64), x_scaled])
    residual_unshrunk = design @ coef
    residual_pred = np.clip(
        residual_unshrunk * candidate.shrink,
        -candidate.residual_clip,
        candidate.residual_clip,
    )
    pred_loss = frame["base_pred_loss"].to_numpy(dtype=np.float64) * np.exp(residual_pred)
    return FitResult(
        candidate=candidate,
        coef=coef,
        feature_mean=feature_mean,
        feature_std=feature_std,
        residual_unshrunk=residual_unshrunk,
        residual_pred=residual_pred,
        pred_loss=pred_loss,
    )


def window_indices(frame: pd.DataFrame, schedule: str, window: str) -> np.ndarray:
    schedule_idx = np.flatnonzero(frame["schedule"].to_numpy() == schedule)
    steps = frame["step"].to_numpy(dtype=np.int64)[schedule_idx]
    if window == "full":
        return schedule_idx
    if window == "steps_27126_33906":
        return schedule_idx[(steps >= 27126) & (steps <= 33906)]
    if window == "steps_30000_33906":
        return schedule_idx[(steps >= 30000) & (steps <= 33906)]
    if window == "last_2048_sampled":
        return schedule_idx[-2048:]
    raise ValueError(f"Unknown window: {window}")


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


def evaluate_loss(
    frame: pd.DataFrame,
    pred_loss: np.ndarray,
    schedule: str,
    window: str,
) -> dict[str, float]:
    idx = window_indices(frame, schedule, window)
    actual = frame["loss"].to_numpy(dtype=np.float64)[idx]
    pred = pred_loss[idx]
    return {"n": int(len(idx)), **metric_dict(actual, pred)}


def evaluate_candidate_grid(frame: pd.DataFrame, candidates: list[Candidate]) -> pd.DataFrame:
    rows = []
    feature_cache: dict[tuple[str, ...], np.ndarray] = {}
    grouped: dict[tuple[str, tuple[str, ...], float], list[Candidate]] = {}
    for candidate in candidates:
        key = (candidate.feature_set, candidate.feature_names, candidate.ridge_alpha)
        grouped.setdefault(key, []).append(candidate)

    base_pred_loss = frame["base_pred_loss"].to_numpy(dtype=np.float64)
    for _, group_candidates in grouped.items():
        fit_candidate_config = group_candidates[0]
        feature_names = fit_candidate_config.feature_names
        x_raw = feature_cache.get(feature_names)
        if x_raw is None:
            x_raw = frame.loc[:, list(feature_names)].to_numpy(dtype=np.float64)
            feature_cache[feature_names] = x_raw
        fit = fit_candidate(frame, fit_candidate_config, x_raw=x_raw)
        for candidate in group_candidates:
            residual_pred = np.clip(
                fit.residual_unshrunk * candidate.shrink,
                -candidate.residual_clip,
                candidate.residual_clip,
            )
            pred_loss = base_pred_loss * np.exp(residual_pred)
            train_full = evaluate_loss(frame, pred_loss, TRAIN_SCHEDULE, "full")
            val_full = evaluate_loss(frame, pred_loss, VALIDATION_SCHEDULE, "full")
            val_tail = evaluate_loss(frame, pred_loss, VALIDATION_SCHEDULE, "steps_27126_33906")
            val_last = evaluate_loss(frame, pred_loss, VALIDATION_SCHEDULE, "last_2048_sampled")
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "feature_group": candidate.feature_group,
                    "feature_set": candidate.feature_set,
                    "feature_names": "|".join(candidate.feature_names),
                    "feature_count": candidate.feature_count,
                    "half_life_steps": candidate.half_life_steps,
                    "alpha_per_sample": candidate.alpha_per_sample,
                    "ridge_alpha": candidate.ridge_alpha,
                    "shrink": candidate.shrink,
                    "residual_clip": candidate.residual_clip,
                    "selected_on": f"{VALIDATION_SCHEDULE}:full_mae",
                    "train_full_mae": train_full["mae"],
                    "train_full_rmse": train_full["rmse"],
                    "train_full_r2": train_full["r2"],
                    "validation_full_mae": val_full["mae"],
                    "validation_full_rmse": val_full["rmse"],
                    "validation_full_r2": val_full["r2"],
                    "validation_full_endpoint_abs_diff": val_full["endpoint_abs_diff"],
                    "validation_tail_27126_33906_mae": val_tail["mae"],
                    "validation_tail_27126_33906_rmse": val_tail["rmse"],
                    "validation_last_2048_mae": val_last["mae"],
                    "coef_intercept": float(fit.coef[0]),
                    "coef_l2": float(np.linalg.norm(fit.coef[1:])),
                }
            )
    grid = pd.DataFrame(rows).sort_values(
        [
            "validation_full_mae",
            "validation_full_rmse",
            "validation_full_endpoint_abs_diff",
            "feature_count",
            "candidate_id",
        ]
    )
    grid["validation_full_rank"] = np.arange(1, len(grid) + 1)
    return grid.reset_index(drop=True)


def baseline_predictions(frame: pd.DataFrame) -> np.ndarray:
    return frame["base_pred_loss"].to_numpy(dtype=np.float64)


def mean_shift_predictions(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_mask = frame["schedule"].to_numpy() == TRAIN_SCHEDULE
    residual_mean = float(frame.loc[train_mask, "log_residual"].mean())
    residual_pred = np.full(len(frame), residual_mean, dtype=np.float64)
    pred_loss = frame["base_pred_loss"].to_numpy(dtype=np.float64) * np.exp(residual_pred)
    return residual_pred, pred_loss


def load_step_reference_metrics() -> tuple[pd.DataFrame, dict[tuple[str, str], float], str]:
    if not STEP_REFERENCE_METRICS_CSV.exists():
        raise FileNotFoundError(
            f"Missing step reference metrics: {STEP_REFERENCE_METRICS_CSV}. "
            "Run scripts/run_spline_stability_audit.py first or remove the step-reference comparison."
        )
    reference = pd.read_csv(STEP_REFERENCE_METRICS_CSV)
    reference = reference[
        (reference["model"] == "step_spline_811_selected")
        & (reference["target_schedule"].isin([VALIDATION_SCHEDULE, TEST_SCHEDULE]))
        & (reference["window"].isin(PRIMARY_WINDOWS))
    ].copy()
    if reference.empty:
        raise RuntimeError(f"No selected step-spline rows found in {STEP_REFERENCE_METRICS_CSV}")
    reference["model"] = "step_spline_reference_811_selected"
    reference["feature_set"] = "absolute_step"
    reference["split"] = reference["target_schedule"].map(
        {VALIDATION_SCHEDULE: "validation", TEST_SCHEDULE: "test"}
    )
    reference_mae = {
        (str(row["target_schedule"]), str(row["window"])): float(row["mae"])
        for _, row in reference.iterrows()
    }
    config = str(reference["config"].iloc[0])
    return reference, reference_mae, config


def selected_metrics(
    frame: pd.DataFrame,
    selected_fits: list[tuple[str, str, str, np.ndarray]],
    step_reference_mae: dict[tuple[str, str], float],
) -> pd.DataFrame:
    base_pred = baseline_predictions(frame)
    base_metrics = {
        (schedule, window): evaluate_loss(frame, base_pred, schedule, window)
        for schedule in [TRAIN_SCHEDULE, VALIDATION_SCHEDULE, TEST_SCHEDULE]
        for window in PRIMARY_WINDOWS
    }
    rows = []
    for model, feature_set, config, pred_loss in selected_fits:
        for schedule, split in [
            (TRAIN_SCHEDULE, "train"),
            (VALIDATION_SCHEDULE, "validation"),
            (TEST_SCHEDULE, "test"),
        ]:
            for window in PRIMARY_WINDOWS:
                metrics = evaluate_loss(frame, pred_loss, schedule, window)
                base = base_metrics[(schedule, window)]
                reference_mae = step_reference_mae.get((schedule, window), float("nan"))
                rows.append(
                    {
                        "model": model,
                        "feature_set": feature_set,
                        "config": config,
                        "split": split,
                        "target_schedule": schedule,
                        "window": window,
                        **metrics,
                        "mae_baseline": base["mae"],
                        "mae_improvement_vs_momentum": base["mae"] - metrics["mae"],
                        "relative_mae_improvement_pct": 100.0 * (base["mae"] - metrics["mae"]) / base["mae"],
                        "step_reference_mae": reference_mae,
                        "mae_gap_vs_step_reference": metrics["mae"] - reference_mae,
                    }
                )
    return pd.DataFrame(rows)


def append_step_reference_metrics(
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    base_pred = baseline_predictions(frame)
    base_metrics = {
        (schedule, window): evaluate_loss(frame, base_pred, schedule, window)
        for schedule in [VALIDATION_SCHEDULE, TEST_SCHEDULE]
        for window in PRIMARY_WINDOWS
    }
    rows = []
    for _, ref in reference.iterrows():
        schedule = str(ref["target_schedule"])
        window = str(ref["window"])
        base = base_metrics[(schedule, window)]
        rows.append(
            {
                "model": str(ref["model"]),
                "feature_set": str(ref["feature_set"]),
                "config": str(ref["config"]),
                "split": str(ref["split"]),
                "target_schedule": schedule,
                "window": window,
                "n": int(ref["n"]),
                "mae": float(ref["mae"]),
                "rmse": float(ref["rmse"]),
                "mape": float(ref["mape"]),
                "r2": float(ref["r2"]),
                "signed_bias": float(ref["signed_bias"]),
                "max_abs_error": float(ref["max_abs_error"]),
                "endpoint_abs_diff": float(ref["endpoint_abs_diff"]),
                "mae_baseline": base["mae"],
                "mae_improvement_vs_momentum": base["mae"] - float(ref["mae"]),
                "relative_mae_improvement_pct": 100.0 * (base["mae"] - float(ref["mae"])) / base["mae"],
                "step_reference_mae": float(ref["mae"]),
                "mae_gap_vs_step_reference": 0.0,
            }
        )
    return pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) < EPS or float(np.std(b)) < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or float(np.std(a)) < EPS or float(np.std(b)) < EPS:
        return float("nan")
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def feature_summary(frame: pd.DataFrame, named_fits: list[tuple[str, FitResult]]) -> pd.DataFrame:
    rows = []
    schedules = {
        "train": TRAIN_SCHEDULE,
        "validation": VALIDATION_SCHEDULE,
    }
    residual = frame["log_residual"].to_numpy(dtype=np.float64)
    for model_name, fit in named_fits:
        candidate = fit.candidate
        rows.append(
            {
                "model": model_name,
                "candidate_id": candidate.candidate_id,
                "feature_group": candidate.feature_group,
                "feature_set": candidate.feature_set,
                "feature": "(intercept)",
                "coefficient_standardized": float(fit.coef[0]),
                "abs_coefficient_rank": 0,
                "train_pearson_corr": float("nan"),
                "train_spearman_corr": float("nan"),
                "validation_pearson_corr": float("nan"),
                "validation_spearman_corr": float("nan"),
                "train_feature_mean_raw": float("nan"),
                "train_feature_std_raw": float("nan"),
                "validation_feature_mean_z": float("nan"),
                "validation_feature_std_z": float("nan"),
            }
        )
        coefs = fit.coef[1:]
        rank_order = np.argsort(-np.abs(coefs))
        ranks = {int(idx): int(rank + 1) for rank, idx in enumerate(rank_order)}
        raw = frame.loc[:, list(candidate.feature_names)].to_numpy(dtype=np.float64)
        z = (raw - fit.feature_mean) / fit.feature_std
        for feature_idx, feature_name in enumerate(candidate.feature_names):
            corr_values: dict[str, float] = {}
            for split, schedule in schedules.items():
                mask = frame["schedule"].to_numpy() == schedule
                corr_values[f"{split}_pearson_corr"] = pearson_corr(z[mask, feature_idx], residual[mask])
                corr_values[f"{split}_spearman_corr"] = spearman_corr(z[mask, feature_idx], residual[mask])
            val_mask = frame["schedule"].to_numpy() == VALIDATION_SCHEDULE
            rows.append(
                {
                    "model": model_name,
                    "candidate_id": candidate.candidate_id,
                    "feature_group": candidate.feature_group,
                    "feature_set": candidate.feature_set,
                    "feature": feature_name,
                    "coefficient_standardized": float(coefs[feature_idx]),
                    "abs_coefficient_rank": ranks[feature_idx],
                    "train_pearson_corr": corr_values["train_pearson_corr"],
                    "train_spearman_corr": corr_values["train_spearman_corr"],
                    "validation_pearson_corr": corr_values["validation_pearson_corr"],
                    "validation_spearman_corr": corr_values["validation_spearman_corr"],
                    "train_feature_mean_raw": float(fit.feature_mean[feature_idx]),
                    "train_feature_std_raw": float(fit.feature_std[feature_idx]),
                    "validation_feature_mean_z": float(np.mean(z[val_mask, feature_idx])),
                    "validation_feature_std_z": float(np.std(z[val_mask, feature_idx])),
                }
            )
    return pd.DataFrame(rows)


def selected_row(metrics: pd.DataFrame, model: str, schedule: str, window: str) -> pd.Series:
    return metrics[
        (metrics["model"] == model)
        & (metrics["target_schedule"] == schedule)
        & (metrics["window"] == window)
    ].iloc[0]


def markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for header in headers:
            value = row[header]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_log(
    grid: pd.DataFrame,
    metrics: pd.DataFrame,
    global_fit: FitResult,
    ema_fit: FitResult,
    step_config: str,
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    base_wsd_full = selected_row(metrics, "momentum_baseline", TEST_SCHEDULE, "full")
    global_wsd_full = selected_row(metrics, "sujianlin_grid_811_selected", TEST_SCHEDULE, "full")
    ema_wsd_full = selected_row(metrics, "best_ema_memory_811_selected", TEST_SCHEDULE, "full")
    step_wsd_full = selected_row(metrics, "step_spline_reference_811_selected", TEST_SCHEDULE, "full")

    global_tail = selected_row(metrics, "sujianlin_grid_811_selected", TEST_SCHEDULE, "steps_27126_33906")
    ema_tail = selected_row(metrics, "best_ema_memory_811_selected", TEST_SCHEDULE, "steps_27126_33906")
    step_tail = selected_row(metrics, "step_spline_reference_811_selected", TEST_SCHEDULE, "steps_27126_33906")

    global_val = selected_row(metrics, "sujianlin_grid_811_selected", VALIDATION_SCHEDULE, "full")
    ema_val = selected_row(metrics, "best_ema_memory_811_selected", VALIDATION_SCHEDULE, "full")

    top_rows = grid.head(5)[
        [
            "validation_full_rank",
            "candidate_id",
            "feature_group",
            "feature_set",
            "validation_full_mae",
            "validation_tail_27126_33906_mae",
            "train_full_mae",
        ]
    ]
    top_table = markdown_table(top_rows)

    text = f"""# Attempt 009: Sujianlin EMA Memory Audit

- Idea: `reference/sujianlin.pdf` highlights the sliding-average view of
  optimizers: Adam/EMA style states make the effective update depend on
  historical learning-rate mass, not only the current learning rate. This audit
  tests whether that history-memory view explains the residual left after the
  momentum baseline.
- Protocol: read `results/momentum_residual_mlp_results/predictions.csv`; keep
  `1000 <= step <= 33906` with every-2-step sampling; define
  `residual = log(loss) - log(momentum_s2)`; train residual ridge models on
  `cosine`; choose candidates by `811` full-trajectory MAE; report WSD only
  after selection. Features are schedule-only: current LR controls, normalized
  `s1/s2`, and EMA/history features of LR, LR^2, positive LR drops, and absolute
  LR changes across half-lives `{HALF_LIFE_STEPS}`. No WSD loss history or
  scheduler label is used as a model input.
- Selection rule: global grid winner and best EMA-only winner are both selected
  by `811` full MAE. The step spline is included only as a reference, also
  selected on `811`, with config `{step_config}`.

## Selected Candidates

- Global grid winner: `{global_fit.candidate.candidate_id}`
  (`{global_fit.candidate.feature_group}`, features
  `{", ".join(global_fit.candidate.feature_names)}`), 811 full MAE
  `{global_val["mae"]:.6f}`.
- Best EMA-only winner: `{ema_fit.candidate.candidate_id}` (features
  `{", ".join(ema_fit.candidate.feature_names)}`), 811 full MAE
  `{ema_val["mae"]:.6f}`.

Top 5 validation candidates:

{top_table}

## Key WSD Numbers

| model | full MAE | tail `27126-33906` MAE | improvement vs momentum | gap vs step |
|---|---:|---:|---:|---:|
| momentum baseline | {base_wsd_full["mae"]:.6f} | {selected_row(metrics, "momentum_baseline", TEST_SCHEDULE, "steps_27126_33906")["mae"]:.6f} | 0.000000 | {base_wsd_full["mae_gap_vs_step_reference"]:.6f} |
| global grid winner | {global_wsd_full["mae"]:.6f} | {global_tail["mae"]:.6f} | {global_wsd_full["mae_improvement_vs_momentum"]:.6f} | {global_wsd_full["mae_gap_vs_step_reference"]:.6f} |
| best EMA-only | {ema_wsd_full["mae"]:.6f} | {ema_tail["mae"]:.6f} | {ema_wsd_full["mae_improvement_vs_momentum"]:.6f} | {ema_wsd_full["mae_gap_vs_step_reference"]:.6f} |
| step spline reference | {step_wsd_full["mae"]:.6f} | {step_tail["mae"]:.6f} | {step_wsd_full["mae_improvement_vs_momentum"]:.6f} | 0.000000 |

## Conclusion

- The EMA/history grid {'improves' if global_wsd_full["mae"] < base_wsd_full["mae"] else 'does not improve'} the WSD full-trajectory momentum baseline under the frozen 811 selection rule.
- It {'approaches' if global_wsd_full["mae_gap_vs_step_reference"] <= 0.002 else 'does not approach'} the step-spline reference: the WSD full MAE gap is `{global_wsd_full["mae_gap_vs_step_reference"]:.6f}` for the global winner and `{ema_wsd_full["mae_gap_vs_step_reference"]:.6f}` for the best EMA-only model.
- Failure mode: the selected low-dimensional schedule-memory features are too
  coarse to recover the step-aligned residual phase found in the spline audits.
  They can encode smoothed LR history and drop memory, but not the sharper
  absolute-step residual template that transfers from cosine to WSD in this
  three-schedule dataset.

Key output files:

```text
outputs/sujianlin_ema_memory_candidate_grid.csv
outputs/sujianlin_ema_memory_selected_metrics.csv
outputs/sujianlin_ema_memory_feature_summary.csv
```
"""
    (LOG_DIR / "attempt_009_sujianlin_ema_memory_audit.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_predictions()
    candidates = build_candidates()
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    grid = evaluate_candidate_grid(frame, candidates)

    global_candidate_id = str(grid.iloc[0]["candidate_id"])
    ema_candidate_id = str(
        grid[grid["feature_group"] == "ema_memory"].sort_values(
            [
                "validation_full_mae",
                "validation_full_rmse",
                "validation_full_endpoint_abs_diff",
                "feature_count",
                "candidate_id",
            ]
        ).iloc[0]["candidate_id"]
    )
    grid["is_global_811_selected"] = grid["candidate_id"] == global_candidate_id
    grid["is_best_ema_811_selected"] = grid["candidate_id"] == ema_candidate_id

    global_fit = fit_candidate(frame, candidate_by_id[global_candidate_id])
    ema_fit = fit_candidate(frame, candidate_by_id[ema_candidate_id])
    _, mean_pred = mean_shift_predictions(frame)
    step_reference, step_reference_mae, step_config = load_step_reference_metrics()
    base_pred = baseline_predictions(frame)

    selected = selected_metrics(
        frame,
        [
            ("momentum_baseline", "none", "none", base_pred),
            ("mean_shift", "constant_residual", "train_residual_mean", mean_pred),
            (
                "sujianlin_grid_811_selected",
                global_fit.candidate.feature_set,
                global_fit.candidate.candidate_id,
                global_fit.pred_loss,
            ),
            (
                "best_ema_memory_811_selected",
                ema_fit.candidate.feature_set,
                ema_fit.candidate.candidate_id,
                ema_fit.pred_loss,
            ),
        ],
        step_reference_mae,
    )
    selected = append_step_reference_metrics(frame, selected, step_reference)
    summary = feature_summary(
        frame,
        [
            ("sujianlin_grid_811_selected", global_fit),
            ("best_ema_memory_811_selected", ema_fit),
        ],
    )

    grid.to_csv(OUTPUT_DIR / "sujianlin_ema_memory_candidate_grid.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "sujianlin_ema_memory_selected_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "sujianlin_ema_memory_feature_summary.csv", index=False)
    write_log(grid, selected, global_fit, ema_fit, step_config)

    report = selected[
        (selected["target_schedule"] == TEST_SCHEDULE)
        & (selected["window"].isin(PRIMARY_WINDOWS))
        & (
            selected["model"].isin(
                [
                    "momentum_baseline",
                    "sujianlin_grid_811_selected",
                    "best_ema_memory_811_selected",
                    "step_spline_reference_811_selected",
                ]
            )
        )
    ].sort_values(["window", "model"])
    print(f"Global 811-selected candidate: {global_candidate_id}")
    print(f"Best EMA-only 811-selected candidate: {ema_candidate_id}")
    print(f"Step spline reference config: {step_config}")
    print("\nWSD selected metrics:")
    print(
        report[
            [
                "model",
                "feature_set",
                "window",
                "n",
                "mae",
                "rmse",
                "mae_improvement_vs_momentum",
                "mae_gap_vs_step_reference",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
