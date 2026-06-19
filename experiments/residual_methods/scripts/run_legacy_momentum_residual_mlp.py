"""Legacy momentum-law residual MLP diagnostic.

Default protocol:
  train: cosine
  validation: 811
  test: wsd

The analytic baseline is the momentum law
  L = L0 + A * S1^(-alpha) - C * S2

The residual model learns either loss - momentum_prediction or
log(loss) - log(momentum_prediction) with a small MLP. The default restores the
10-feature kernel-summary configuration that reaches WSD R2 above 0.94 in the
cross-schedule protocol. The final shrink factor is selected only on the 811
validation curve; WSD is held out until reporting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

RUN_ALIASES = {
    "811": "M:100M_gpt_D:20B_scheduler:811_rope",
    "wsd": "M:100M_gpt_D:20B_scheduler:wsd_rope",
    "cosine": "M:100M_gpt_D:20B_scheduler:cosine_rope",
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "gpt_loss+lrs.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "diagnostics" / "momentum_residual_mlp"


@dataclass
class Curve:
    alias: str
    run_name: str
    step: np.ndarray
    lr: np.ndarray
    loss: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    lr_drop: np.ndarray


@dataclass
class Standardizer:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean = np.asarray(np.nanmean(x, axis=0), dtype=np.float64)
        self.std = np.asarray(np.nanstd(x, axis=0), dtype=np.float64)
        self.std[self.std < 1e-12] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer has not been fitted.")
        out = (x - self.mean) / self.std
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class PhysicsScaler:
    s1_scale: float = 1.0
    diff_scale: float = 1.0

    def fit(self, x: np.ndarray) -> "PhysicsScaler":
        self.s1_scale = float(np.max(np.abs(x[:, 0]))) if x.shape[0] else 1.0
        if self.s1_scale < 1e-12:
            self.s1_scale = 1.0
        if x.shape[1] > 1:
            self.diff_scale = float(np.max(np.abs(x[:, 1:])))
            if self.diff_scale < 1e-12:
                self.diff_scale = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = x.astype(np.float64, copy=True)
        out[:, 0] = out[:, 0] / self.s1_scale
        if out.shape[1] > 1:
            out[:, 1:] = out[:, 1:] / self.diff_scale
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class TrainedResidual:
    model: MLPRegressor
    scaler: Standardizer | PhysicsScaler
    y_mean: float
    y_std: float
    raw_outputs: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a momentum-law residual MLP.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-run", default="cosine", choices=sorted(RUN_ALIASES))
    parser.add_argument("--validation-run", default="811", choices=sorted(RUN_ALIASES))
    parser.add_argument("--test-run", default="wsd", choices=sorted(RUN_ALIASES))
    parser.add_argument("--start-step", type=int, default=1000)
    parser.add_argument("--momentum-decay", type=float, default=0.999)
    parser.add_argument(
        "--feature-set",
        default="kernel_summary",
        choices=["base_lr", "momentum", "full_history_10", "kernel_summary"],
    )
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--target", default="log_residual", choices=["residual", "log_residual"])
    parser.add_argument("--seeds", default="3081")
    parser.add_argument("--hidden", default="64,32")
    parser.add_argument("--activation", default="relu", choices=["identity", "logistic", "tanh", "relu"])
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--scaler", default="physics", choices=["standard", "physics"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--shrink-grid", default="0,0.75,1.0,1.25")
    parser.add_argument("--huber-delta", type=float, default=1e-3)
    return parser.parse_args()


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_hidden(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("--hidden must contain at least one layer size")
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def compute_s1_s2(lr: np.ndarray, decay: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lr_drop = np.zeros_like(lr, dtype=np.float64)
    lr_drop[1:] = lr[:-1] - lr[1:]
    memory = np.zeros_like(lr, dtype=np.float64)
    for i in range(1, len(lr)):
        memory[i] = decay * memory[i - 1] + lr_drop[i]
    return np.cumsum(lr, dtype=np.float64), np.cumsum(memory, dtype=np.float64), lr_drop


def load_curves(data_path: Path, aliases: Iterable[str], momentum_decay: float) -> dict[str, Curve]:
    raw = pd.read_pickle(data_path)
    curves: dict[str, Curve] = {}
    for alias in aliases:
        run_name = RUN_ALIASES[alias]
        if run_name not in raw:
            raise KeyError(f"Run {run_name!r} not found in {data_path}")
        df = raw[run_name][["step", "lr", "Metrics/loss"]].copy()
        df = df.rename(columns={"Metrics/loss": "loss"})
        df = df.sort_values("step")
        df = df[np.isfinite(df["step"]) & np.isfinite(df["lr"]) & np.isfinite(df["loss"])]
        df["step"] = df["step"].astype(int)
        if df.empty:
            raise ValueError(f"Run {run_name!r} has no valid rows")

        full_step = np.arange(int(df["step"].min()), int(df["step"].max()) + 1, dtype=np.int64)
        filled = df.set_index("step").reindex(full_step)
        filled.index.name = "step"
        filled[["lr", "loss"]] = filled[["lr", "loss"]].interpolate(method="linear", limit_direction="both")
        filled = filled.reset_index()

        lr = filled["lr"].to_numpy(np.float64)
        s1, s2, lr_drop = compute_s1_s2(lr, momentum_decay)
        curves[alias] = Curve(
            alias=alias,
            run_name=run_name,
            step=filled["step"].to_numpy(np.int64),
            lr=lr,
            loss=filled["loss"].to_numpy(np.float64),
            s1=s1,
            s2=s2,
            lr_drop=lr_drop,
        )
    return curves


def huber(residual: np.ndarray, delta: float) -> np.ndarray:
    abs_residual = np.abs(residual)
    return np.where(abs_residual < delta, 0.5 * residual**2, delta * abs_residual - 0.5 * delta**2)


def momentum_predict(params: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    l0, a, c, alpha = params
    return l0 + a * np.power(np.maximum(s1, 1e-12), -alpha) - c * s2


def fit_momentum_law(
    curves: dict[str, Curve],
    train_indices: dict[str, np.ndarray],
    huber_delta: float,
) -> tuple[np.ndarray, float]:
    s1 = np.concatenate([curves[alias].s1[idx] for alias, idx in train_indices.items()])
    s2 = np.concatenate([curves[alias].s2[idx] for alias, idx in train_indices.items()])
    loss = np.concatenate([curves[alias].loss[idx] for alias, idx in train_indices.items()])

    def objective(params: np.ndarray) -> float:
        pred = momentum_predict(params, s1, s2)
        if np.any(~np.isfinite(pred)) or np.any(pred <= 0):
            return 1e8
        return float(np.sum(huber(np.log(loss) - np.log(pred), huber_delta)))

    starts = [
        (l0, a, c, alpha)
        for l0 in np.linspace(0.5, 2.8, 3)
        for a in np.linspace(0.5, 6.0, 3)
        for c in [0.01, 0.1, 0.5, 1.0]
        for alpha in [0.3, 0.6, 0.9, 1.2]
    ]
    bounds = [(0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)]
    best_params: np.ndarray | None = None
    best_value = math.inf
    for start in starts:
        result = minimize(
            objective,
            np.array(start, dtype=np.float64),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 30000, "ftol": 1e-9, "gtol": 1e-6, "eps": 1e-8},
        )
        if np.isfinite(result.fun) and float(result.fun) < best_value:
            best_value = float(result.fun)
            best_params = result.x.astype(np.float64)
    if best_params is None:
        raise RuntimeError("Momentum-law fit failed")
    return best_params, best_value


def ewm(values: np.ndarray, decay: float) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[0] = float(values[0])
    for i in range(1, len(values)):
        out[i] = decay * out[i - 1] + (1.0 - decay) * float(values[i])
    return out


def full_history_ewm(values: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if len(values) == 0:
        return out
    out[0] = float(values[0])
    for i in range(1, len(values)):
        out[i] = float(values[i]) + decay * out[i - 1]
    return out


def recent_diff_window(lr_drop: np.ndarray, m: int) -> np.ndarray:
    if m <= 0:
        raise ValueError("--m must be positive for kernel_summary features")
    n = len(lr_drop)
    indices = np.arange(n, dtype=np.int64)
    window = np.empty((n, m), dtype=np.float64)
    for lag in range(m):
        src = indices - lag
        valid = src >= 0
        values = np.zeros(n, dtype=np.float64)
        values[valid] = lr_drop[src[valid]]
        window[:, lag] = values
    return window


def feature_names(feature_set: str) -> list[str]:
    base_lr = [
        "step_norm",
        "log1p_step",
        "lr",
        "lr_next",
        "lr_delta_next",
        "lr_drop_positive",
        "S1",
        "log_S1",
        "S1_inv_sqrt",
        "S1_inv",
    ]
    if feature_set == "base_lr":
        return base_lr
    kernel_summary = [
        "S1",
        "kernel_ewm_diff_decay_0.9",
        "kernel_ewm_diff_decay_0.99",
        "kernel_ewm_diff_decay_0.995",
        "kernel_ewm_diff_decay_0.999",
        "kernel_diff_sum",
        "kernel_abs_diff_sum",
        "kernel_diff_max",
        "kernel_diff_min",
        "kernel_nonzero_diff_count",
    ]
    if feature_set == "kernel_summary":
        return kernel_summary
    full_history_10 = [
        "S1",
        "ewm_diff_decay_0.9_full",
        "ewm_diff_decay_0.99_full",
        "ewm_diff_decay_0.995_full",
        "ewm_diff_decay_0.999_full",
        "diff_sum_full",
        "abs_diff_sum_full",
        "diff_max_full",
        "diff_min_full",
        "nonzero_diff_count_full",
    ]
    if feature_set == "full_history_10":
        return full_history_10
    return base_lr + [
        "S2_momentum",
        "sum_lr_squared",
        "ewm_drop_0.9",
        "ewm_drop_0.99",
        "ewm_drop_0.999",
        "diff_sum_full",
        "abs_diff_sum_full",
        "momentum_pred",
        "momentum_log_pred",
        "momentum_delta_next",
    ]


def build_columns(curve: Curve, momentum_pred: np.ndarray, m: int) -> dict[str, np.ndarray]:
    n = len(curve.step)
    idx = np.arange(n, dtype=np.float64)
    denom = max(float(n - 1), 1.0)
    lr_next = np.concatenate([curve.lr[1:], curve.lr[-1:]])
    pred_next = np.concatenate([momentum_pred[1:], momentum_pred[-1:]])
    window = recent_diff_window(curve.lr_drop, m)
    lags = np.arange(m, dtype=np.float64)
    out = {
        "step_norm": idx / denom,
        "log1p_step": np.log1p(idx),
        "lr": curve.lr,
        "lr_next": lr_next,
        "lr_delta_next": lr_next - curve.lr,
        "lr_drop_positive": np.maximum(curve.lr - lr_next, 0.0),
        "S1": curve.s1,
        "log_S1": np.log(np.maximum(curve.s1, 1e-12)),
        "S1_inv_sqrt": np.power(np.maximum(curve.s1, 1e-12), -0.5),
        "S1_inv": np.power(np.maximum(curve.s1, 1e-12), -1.0),
        "S2_momentum": curve.s2,
        "sum_lr_squared": np.cumsum(curve.lr.astype(np.float64) ** 2, dtype=np.float64),
        "ewm_drop_0.9": ewm(curve.lr_drop, 0.9),
        "ewm_drop_0.99": ewm(curve.lr_drop, 0.99),
        "ewm_drop_0.999": ewm(curve.lr_drop, 0.999),
        "diff_sum_full": np.cumsum(curve.lr_drop, dtype=np.float64),
        "abs_diff_sum_full": np.cumsum(np.abs(curve.lr_drop), dtype=np.float64),
        "ewm_diff_decay_0.9_full": full_history_ewm(curve.lr_drop, 0.9),
        "ewm_diff_decay_0.99_full": full_history_ewm(curve.lr_drop, 0.99),
        "ewm_diff_decay_0.995_full": full_history_ewm(curve.lr_drop, 0.995),
        "ewm_diff_decay_0.999_full": full_history_ewm(curve.lr_drop, 0.999),
        "diff_max_full": np.maximum.accumulate(curve.lr_drop),
        "diff_min_full": np.minimum.accumulate(curve.lr_drop),
        "nonzero_diff_count_full": np.cumsum(curve.lr_drop != 0.0, dtype=np.float64),
        "momentum_pred": momentum_pred,
        "momentum_log_pred": np.log(np.maximum(momentum_pred, 1e-12)),
        "momentum_delta_next": pred_next - momentum_pred,
    }
    for decay in [0.9, 0.99, 0.995, 0.999]:
        out[f"kernel_ewm_diff_decay_{decay:g}"] = window @ np.power(decay, lags)
    out.update(
        {
            "kernel_diff_sum": window.sum(axis=1),
            "kernel_abs_diff_sum": np.abs(window).sum(axis=1),
            "kernel_diff_max": window.max(axis=1),
            "kernel_diff_min": window.min(axis=1),
            "kernel_nonzero_diff_count": (window != 0.0).sum(axis=1).astype(np.float64),
        }
    )
    return out


def build_feature_cache(
    curves: dict[str, Curve],
    momentum_predictions: dict[str, np.ndarray],
    feature_set: str,
    m: int,
) -> tuple[dict[str, np.ndarray], list[str]]:
    names = feature_names(feature_set)
    cache = {}
    for alias, curve in curves.items():
        cols = build_columns(curve, momentum_predictions[alias], m)
        cache[alias] = np.stack([cols[name] for name in names], axis=1)
        cache[alias] = np.nan_to_num(cache[alias], nan=0.0, posinf=0.0, neginf=0.0)
    return cache, names


def train_residual_mlp(
    curves: dict[str, Curve],
    feature_cache: dict[str, np.ndarray],
    train_indices: dict[str, np.ndarray],
    momentum_predictions: dict[str, np.ndarray],
    target: str,
    hidden: tuple[int, ...],
    activation: str,
    alpha: float,
    learning_rate: float,
    batch_size: int,
    max_iter: int,
    seed: int,
    scaler_name: str,
) -> TrainedResidual:
    x_train = np.vstack([feature_cache[alias][idx] for alias, idx in train_indices.items()])
    if scaler_name == "standard":
        scaler: Standardizer | PhysicsScaler = Standardizer().fit(x_train)
    elif scaler_name == "physics":
        scaler = PhysicsScaler().fit(x_train)
    else:
        raise ValueError(f"Unsupported scaler: {scaler_name}")

    y_parts = []
    for alias, idx in train_indices.items():
        loss = curves[alias].loss[idx]
        baseline = momentum_predictions[alias][idx]
        if target == "residual":
            y_parts.append(loss - baseline)
        elif target == "log_residual":
            y_parts.append(np.log(np.maximum(loss, 1e-12)) - np.log(np.maximum(baseline, 1e-12)))
        else:
            raise ValueError(f"Unsupported target: {target}")
    y_train = np.concatenate(y_parts).astype(np.float64)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    if y_std < 1e-12:
        y_std = 1.0
    y_scaled = (y_train - y_mean) / y_std

    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation=activation,
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=seed,
        shuffle=True,
        early_stopping=False,
        n_iter_no_change=30,
        tol=1e-7,
        verbose=False,
    )
    model.fit(scaler.transform(x_train), y_scaled)
    raw_outputs = {
        alias: model.predict(scaler.transform(features)) * y_std + y_mean
        for alias, features in feature_cache.items()
    }
    return TrainedResidual(model=model, scaler=scaler, y_mean=y_mean, y_std=y_std, raw_outputs=raw_outputs)


def residual_predictions(
    trained: TrainedResidual,
    momentum_predictions: dict[str, np.ndarray],
    shrink: float,
    target: str,
) -> dict[str, np.ndarray]:
    preds: dict[str, np.ndarray] = {}
    for alias, raw in trained.raw_outputs.items():
        baseline = momentum_predictions[alias]
        if target == "residual":
            pred = baseline + shrink * raw
        elif target == "log_residual":
            pred = baseline * np.exp(np.clip(shrink * raw, -50.0, 50.0))
        else:
            raise ValueError(f"Unsupported target: {target}")
        preds[alias] = np.maximum(np.asarray(pred, dtype=np.float64), 1e-8)
    return preds


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask].astype(np.float64)
    y_pred = y_pred[mask].astype(np.float64)
    residual = y_pred - y_true
    mse = float(np.mean(residual**2))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(mse)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "max_abs_error": float(np.max(np.abs(residual))),
        "endpoint_abs_diff": float(abs(y_pred[-1] - y_true[-1])),
    }


def evaluate_splits(
    curves: dict[str, Curve],
    predictions: dict[str, np.ndarray],
    splits: dict[str, dict[str, np.ndarray]],
    model_name: str,
) -> list[dict[str, object]]:
    rows = []
    for split_name, alias_indices in splits.items():
        all_true = []
        all_pred = []
        for alias, idx in alias_indices.items():
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "run": alias,
                    **metric_dict(curves[alias].loss[idx], predictions[alias][idx]),
                }
            )
            all_true.append(curves[alias].loss[idx])
            all_pred.append(predictions[alias][idx])
        if len(alias_indices) > 1:
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "run": "all",
                    **metric_dict(np.concatenate(all_true), np.concatenate(all_pred)),
                }
            )
    return rows


def plot_fit(
    curves: dict[str, Curve],
    predictions: dict[str, dict[str, np.ndarray]],
    splits: dict[str, dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    aliases = ["cosine", "811", "wsd"]
    labels = {alias: [] for alias in aliases}
    for split_name, alias_indices in splits.items():
        for alias, idx in alias_indices.items():
            labels[alias].append(f"{split_name}: {int(idx[0])}-{int(idx[-1])}")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex="row")
    for row, alias in enumerate(aliases):
        curve = curves[alias]
        idx = np.arange(len(curve.step), dtype=np.int64)
        stride = max(1, len(idx) // 2500)
        view = idx[::stride]
        ax_fit = axes[row, 0]
        ax_err = axes[row, 1]
        ax_fit.plot(curve.step[view], curve.loss[view], color="black", linewidth=1.1, label="true")
        for name, pred_by_alias in predictions.items():
            ax_fit.plot(curve.step[view], pred_by_alias[alias][view], linewidth=1.0, label=name)
            ax_err.plot(curve.step[view], pred_by_alias[alias][view] - curve.loss[view], linewidth=1.0, label=name)
        ax_err.axhline(0.0, color="black", linewidth=0.8)
        ax_fit.set_title(f"{alias} ({'; '.join(labels[alias])})")
        ax_err.set_title(f"{alias} error")
        ax_fit.set_ylabel("loss")
        ax_err.set_ylabel("prediction - true")
        ax_fit.grid(alpha=0.25)
        ax_err.grid(alpha=0.25)
        if row == 0:
            ax_fit.legend(fontsize=8)
            ax_err.legend(fontsize=8)
    axes[-1, 0].set_xlabel("step")
    axes[-1, 1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_int_list(args.seeds)
    hidden = parse_hidden(args.hidden)
    shrink_grid = parse_float_list(args.shrink_grid)
    set_seed(seeds[0])
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    aliases = sorted({args.train_run, args.validation_run, args.test_run})
    curves = load_curves(data_path, aliases, args.momentum_decay)
    end = min(len(curve.step) for curve in curves.values())
    idx = np.arange(args.start_step, end, dtype=np.int64)
    if len(idx) == 0:
        raise ValueError("No training/evaluation indices after --start-step")
    train_indices = {args.train_run: idx}
    splits = {
        "train": {args.train_run: idx},
        "validation": {args.validation_run: idx},
        "test": {args.test_run: idx},
    }

    momentum_params, momentum_objective = fit_momentum_law(curves, train_indices, args.huber_delta)
    momentum_predictions = {
        alias: np.maximum(momentum_predict(momentum_params, curve.s1, curve.s2), 1e-8)
        for alias, curve in curves.items()
    }
    feature_cache, features = build_feature_cache(curves, momentum_predictions, args.feature_set, args.m)

    metric_rows = evaluate_splits(curves, momentum_predictions, splits, "momentum_s2")
    trial_rows = []
    selected: dict[str, object] | None = None
    selected_predictions: dict[str, np.ndarray] | None = None

    for seed in seeds:
        trained = train_residual_mlp(
            curves=curves,
            feature_cache=feature_cache,
            train_indices=train_indices,
            momentum_predictions=momentum_predictions,
            target=args.target,
            hidden=hidden,
            activation=args.activation,
            alpha=args.alpha,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_iter=args.max_iter,
            seed=seed,
            scaler_name=args.scaler,
        )
        for shrink in shrink_grid:
            model_name = f"momentum_residual_mlp_seed{seed}_shrink{str(shrink).replace('.', 'p')}"
            preds = residual_predictions(trained, momentum_predictions, shrink, args.target)
            rows = evaluate_splits(curves, preds, splits, model_name)
            metric_rows.extend(rows)
            by_key = {(row["split"], row["run"]): row for row in rows}
            trial = {
                "model": model_name,
                "seed": seed,
                "shrink": shrink,
                "target": args.target,
                "feature_set": args.feature_set,
                "m": args.m if args.feature_set == "kernel_summary" else np.nan,
                "feature_count": len(features),
                "hidden": "-".join(str(v) for v in hidden),
                "activation": args.activation,
                "scaler": args.scaler,
                "alpha": args.alpha,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "max_iter": args.max_iter,
                "n_iter": int(trained.model.n_iter_),
                "loss_curve_final": float(trained.model.loss_),
                "validation_mae": by_key[("validation", args.validation_run)]["mae"],
                "test_mae": by_key[("test", args.test_run)]["mae"],
                "test_r2": by_key[("test", args.test_run)]["r2"],
                "test_endpoint_abs_diff": by_key[("test", args.test_run)]["endpoint_abs_diff"],
            }
            trial_rows.append(trial)
            if selected is None or (trial["validation_mae"], shrink, seed) < (
                selected["validation_mae"],
                selected["shrink"],
                selected["seed"],
            ):
                selected = trial
                selected_predictions = preds

    if selected is None or selected_predictions is None:
        raise RuntimeError("No residual MLP trial was trained")

    metrics = pd.DataFrame(metric_rows)
    trials = pd.DataFrame(trial_rows).sort_values(["validation_mae", "shrink", "seed"])
    baseline_test = metrics[
        (metrics["model"].eq("momentum_s2"))
        & (metrics["split"].eq("test"))
        & (metrics["run"].eq(args.test_run))
    ].iloc[0]
    selected_test = metrics[
        (metrics["model"].eq(selected["model"]))
        & (metrics["split"].eq("test"))
        & (metrics["run"].eq(args.test_run))
    ].iloc[0]
    improvement_pct = 100.0 * (float(baseline_test["mae"]) - float(selected_test["mae"])) / float(baseline_test["mae"])

    predictions_frame = []
    for alias, curve in curves.items():
        predictions_frame.append(
            pd.DataFrame(
                {
                    "run": alias,
                    "step": curve.step,
                    "loss": curve.loss,
                    "lr": curve.lr,
                    "s1": curve.s1,
                    "s2": curve.s2,
                    "momentum_s2": momentum_predictions[alias],
                    "momentum_residual_mlp": selected_predictions[alias],
                }
            )
        )
    pd.concat(predictions_frame, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    trials.to_csv(output_dir / "trials.csv", index=False)
    plot_fit(
        curves,
        {"momentum_s2": momentum_predictions, "momentum_residual_mlp": selected_predictions},
        splits,
        output_dir / "fit.png",
    )

    summary = {
        "protocol": {
            "train": args.train_run,
            "validation": args.validation_run,
            "test": args.test_run,
            "start_step": args.start_step,
            "selection_rule": "lowest validation MAE on validation run",
        },
        "momentum_law": {
            "formula": "L0 + A*S1^-alpha - C*S2",
            "params": {
                "L0": float(momentum_params[0]),
                "A": float(momentum_params[1]),
                "C": float(momentum_params[2]),
                "alpha": float(momentum_params[3]),
            },
            "objective": float(momentum_objective),
        },
        "residual_mlp": {
            "target": args.target,
            "target_definition": "loss - momentum_s2"
            if args.target == "residual"
            else "log(loss) - log(momentum_s2)",
            "feature_set": args.feature_set,
            "m": args.m if args.feature_set == "kernel_summary" else None,
            "features": features,
            "selected": selected,
        },
        "test_result": {
            "baseline_mae": float(baseline_test["mae"]),
            "residual_mlp_mae": float(selected_test["mae"]),
            "improvement_pct": float(improvement_pct),
            "baseline_r2": float(baseline_test["r2"]),
            "residual_mlp_r2": float(selected_test["r2"]),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = f"""# Momentum-Law Residual MLP

## Protocol

- Train run: `{args.train_run}`
- Validation run: `{args.validation_run}`
- Test run: `{args.test_run}`
- Start step: `{args.start_step}`
- m: `{args.m if args.feature_set == "kernel_summary" else "not used"}`
- Selection rule: lowest validation MAE. The test run is not used to select seed or shrink.
- Shrink grid: `{", ".join(str(v) for v in shrink_grid)}`. The shrink is selected only on the
  validation run; the test run is held out until final reporting.

## Selected Model

- Baseline: `momentum_s2`, formula `L0 + A*S1^-alpha - C*S2`
- Residual target: `{args.target}`
- Target definition: `{"loss - momentum_s2" if args.target == "residual" else "log(loss) - log(momentum_s2)"}`
- Feature set: `{args.feature_set}`
- Features: `{", ".join(features)}`
- Hidden layers: `{hidden}`
- Activation: `{args.activation}`
- Scaler: `{args.scaler}`
- Seed: `{selected["seed"]}`
- Shrink: `{selected["shrink"]}`

## Test Result

| model | test_run | MAE | R2 | endpoint_abs_diff |
|---|---|---:|---:|---:|
| momentum_s2 | {args.test_run} | {float(baseline_test["mae"]):.8f} | {float(baseline_test["r2"]):.8f} | {float(baseline_test["endpoint_abs_diff"]):.8f} |
| momentum_residual_mlp | {args.test_run} | {float(selected_test["mae"]):.8f} | {float(selected_test["r2"]):.8f} | {float(selected_test["endpoint_abs_diff"]):.8f} |

MAE improvement over momentum baseline: `{improvement_pct:.3f}%`.

## Files

- `summary.json`: protocol, selected model, and headline metrics.
- `metrics.csv`: train/validation/test metrics for baseline and all residual trials.
- `trials.csv`: residual MLP seed/shrink search table.
- `predictions.csv`: true loss, momentum prediction, and selected residual MLP prediction.
- `fit.png`: curve and residual visualization.
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")

    print(f"Selected residual MLP: seed={selected['seed']} shrink={selected['shrink']}")
    print(f"Momentum test MAE: {float(baseline_test['mae']):.8f}")
    print(f"Residual MLP test MAE: {float(selected_test['mae']):.8f}")
    print(f"Improvement: {improvement_pct:.3f}%")
    print(f"Outputs written to: {output_dir}")
    if improvement_pct <= 0:
        raise SystemExit("Residual MLP did not beat the momentum baseline; inspect trials.csv")


if __name__ == "__main__":
    main()
