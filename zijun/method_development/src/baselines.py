from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import huber


@dataclass
class BaselineFitConfig:
    train_schedule: str = "cosine"
    fit_stride: int = 50
    huber_delta: float = 1e-2
    maxiter: int = 200
    mpl_chunk_size: int = 256


def fit_and_predict_baselines(
    curves: pd.DataFrame,
    output_dir: Path,
    config: BaselineFitConfig | None = None,
) -> pd.DataFrame:
    cfg = config or BaselineFitConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    momentum_params = fit_momentum_baseline(curves, cfg)
    mpl_params = fit_mpl_baseline(curves, cfg)

    prediction_frames = []
    for method, params in [
        ("momentum", momentum_params),
        ("mpl", mpl_params),
    ]:
        for schedule, curve in curves.groupby("schedule", sort=False):
            curve = curve.sort_values("step").copy()
            if method == "momentum":
                pred = predict_momentum(curve, params)
            else:
                pred = predict_mpl(curve, params, chunk_size=cfg.mpl_chunk_size)

            frame = curve[["schedule", "step", "loss", "lr"]].copy()
            frame.insert(0, "method", method)
            frame["pred_loss"] = pred
            prediction_frames.append(frame)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "baseline_predictions.csv", index=False)

    params_payload = {
        "config": asdict(cfg),
        "momentum": dict(zip(["L0", "A", "C", "alpha"], momentum_params)),
        "mpl": dict(zip(["L0", "A", "alpha", "B", "C", "beta", "gamma"], mpl_params)),
    }
    with (output_dir / "baseline_params.json").open("w", encoding="utf-8") as f:
        json.dump(params_payload, f, indent=2)

    return predictions


def fit_momentum_baseline(curves: pd.DataFrame, config: BaselineFitConfig) -> np.ndarray:
    full_train_curve = _full_training_curve(curves, config)
    train_curve = _sample_training_curve(full_train_curve, config)
    steps = train_curve["step"].to_numpy(dtype=int)
    loss = train_curve["loss"].to_numpy(dtype=float)
    lr = _dense_lr(full_train_curve)
    S1 = np.cumsum(lr)[steps]
    S2 = _momentum_mass(lr)[steps]

    def objective(params: np.ndarray) -> float:
        L0, A, C, alpha = params
        pred = L0 + A * np.maximum(S1, 1e-12) ** (-alpha) - C * S2
        if np.any(~np.isfinite(pred)) or np.any(pred <= 0):
            return 1e9
        residual = np.log(loss) - np.log(pred)
        return float(huber(config.huber_delta, residual).sum())

    min_loss = float(loss.min())
    init_params = [
        [max(min_loss - 0.2, 0.1), 0.5, 10.0, 0.5],
        [max(min_loss - 0.1, 0.1), 1.0, 100.0, 0.4],
        [max(min_loss - 0.4, 0.1), 2.0, 1000.0, 0.3],
    ]
    bounds = [(0.01, 20.0), (1e-8, 1000.0), (0.0, 1e6), (1e-5, 5.0)]
    return _best_minimize(objective, init_params, bounds, config.maxiter)


def fit_mpl_baseline(curves: pd.DataFrame, config: BaselineFitConfig) -> np.ndarray:
    full_train_curve = _full_training_curve(curves, config)
    train_curve = _sample_training_curve(full_train_curve, config)
    steps = train_curve["step"].to_numpy(dtype=int)
    loss = train_curve["loss"].to_numpy(dtype=float)
    lr = _dense_lr(full_train_curve)
    lr_sum = np.cumsum(lr)
    S1 = lr_sum[steps]

    def objective(params: np.ndarray) -> float:
        L0, A, alpha, B, C, beta, gamma = params
        LD = _mpl_loss_drop(
            lr,
            steps,
            C,
            beta,
            gamma,
            chunk_size=config.mpl_chunk_size,
            source_stride=config.fit_stride,
        )
        pred = L0 + A * np.maximum(S1, 1e-12) ** (-alpha) + B * LD
        if np.any(~np.isfinite(pred)) or np.any(pred <= 0):
            return 1e9
        residual = np.log(loss) - np.log(pred)
        return float(huber(config.huber_delta, residual).sum())

    min_loss = float(loss.min())
    init_params = [
        [max(min_loss - 0.2, 0.1), 0.5, 0.5, 100.0, 1.0, 0.5, 0.5],
        [max(min_loss - 0.1, 0.1), 1.0, 0.4, 500.0, 2.0, 0.6, 0.6],
        [max(min_loss - 0.4, 0.1), 2.0, 0.3, 1000.0, 0.5, 0.4, 0.4],
    ]
    bounds = [
        (0.01, 20.0),
        (1e-8, 1000.0),
        (1e-5, 5.0),
        (0.0, 1e6),
        (1e-8, 1e4),
        (1e-5, 5.0),
        (1e-5, 5.0),
    ]
    return _best_minimize(objective, init_params, bounds, config.maxiter)


def predict_momentum(curve: pd.DataFrame, params: np.ndarray) -> np.ndarray:
    L0, A, C, alpha = params
    sorted_curve = curve.sort_values("step")
    steps = sorted_curve["step"].to_numpy(dtype=int)
    lr = _dense_lr(sorted_curve)
    S1 = np.cumsum(lr)[steps]
    S2 = _momentum_mass(lr)[steps]
    return L0 + A * np.maximum(S1, 1e-12) ** (-alpha) - C * S2


def predict_mpl(curve: pd.DataFrame, params: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    L0, A, alpha, B, C, beta, gamma = params
    sorted_curve = curve.sort_values("step")
    steps = sorted_curve["step"].to_numpy(dtype=int)
    lr = _dense_lr(sorted_curve)
    lr_sum = np.cumsum(lr)
    S1 = lr_sum[steps]
    LD = _mpl_loss_drop(lr, steps, C, beta, gamma, chunk_size=chunk_size)
    return L0 + A * np.maximum(S1, 1e-12) ** (-alpha) + B * LD


def _full_training_curve(curves: pd.DataFrame, config: BaselineFitConfig) -> pd.DataFrame:
    return curves[curves["schedule"] == config.train_schedule].sort_values("step").copy()


def _sample_training_curve(curve: pd.DataFrame, config: BaselineFitConfig) -> pd.DataFrame:
    sampled = curve.iloc[:: config.fit_stride].copy()
    if sampled["step"].iloc[-1] != curve["step"].iloc[-1]:
        sampled = pd.concat([sampled, curve.tail(1)], ignore_index=True)
    return sampled


def _dense_lr(curve: pd.DataFrame) -> np.ndarray:
    frame = curve.sort_values("step")
    max_step = int(frame["step"].max())
    dense = frame.set_index("step")["lr"].reindex(range(max_step + 1))
    dense = dense.interpolate(method="linear").ffill().bfill()
    return dense.to_numpy(dtype=float)


def _momentum_mass(lr: np.ndarray, decay: float = 0.999) -> np.ndarray:
    momentum = np.zeros_like(lr, dtype=float)
    for idx in range(1, len(lr)):
        momentum[idx] = decay * momentum[idx - 1] + (lr[idx - 1] - lr[idx])
    return np.cumsum(momentum)


def _mpl_loss_drop(
    lr: np.ndarray,
    steps: np.ndarray,
    C: float,
    beta: float,
    gamma: float,
    chunk_size: int,
    source_stride: int = 1,
) -> np.ndarray:
    lr_sum = np.cumsum(lr)
    lr_gap = np.zeros_like(lr, dtype=float)
    lr_gap[1:] = np.diff(lr)

    source = np.arange(1, len(lr), max(1, source_stride), dtype=int)
    if source_stride > 1:
        gap_source = np.arange(1, len(lr), dtype=int)
        gap_blocks = np.add.reduceat(lr_gap[1:], source - 1)
        source_gap = gap_blocks[: len(source)]
    else:
        gap_source = source
        source_gap = lr_gap[source]
    source_lr = np.maximum(lr[source], 1e-12)
    source_prev_sum = lr_sum[source - 1]

    result = np.zeros(len(steps), dtype=float)
    for start in range(0, len(steps), chunk_size):
        stop = min(start + chunk_size, len(steps))
        target_steps = steps[start:stop]
        elapsed_sum = lr_sum[target_steps, None] - source_prev_sum[None, :]
        elapsed_sum = np.maximum(elapsed_sum, 0.0)
        power_arg = 1.0 + C * source_lr[None, :] ** (-gamma) * elapsed_sum
        transform = 1.0 - np.maximum(power_arg, 1e-12) ** (-beta)
        valid = source[None, :] <= target_steps[:, None]
        result[start:stop] = np.sum(source_gap[None, :] * transform * valid, axis=1)
    return result


def _best_minimize(objective, init_params, bounds, maxiter: int) -> np.ndarray:
    best_result = None
    for init_param in init_params:
        result = minimize(
            objective,
            np.asarray(init_param, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7},
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result
    if best_result is None:
        raise RuntimeError("Baseline optimization did not run.")
    return best_result.x.astype(float)
