"""Reproduce the Multi-Power Law on the local GPT loss-curve data.

The paper models the post-warmup training loss for a learning-rate schedule
eta_0, ..., eta_t as

    L(t) = L0 + A * S1(t)^(-alpha) - LD(t)
    S1(t) = sum_{tau=0}^t eta_tau

where LR decay contributes an additional loss reduction

    LD(t) = B * sum_{k=1}^t (eta_{k-1} - eta_k)
                  * G(eta_k^(-gamma) * S_k(t))
    S_k(t) = sum_{tau=k}^t eta_tau
    G(x) = 1 - (1 + C * x)^(-beta)

The fitted parameters are L0, A, B, C, alpha, beta, and gamma. Following
Appendix D.1 of Luo et al., the objective is the summed Huber loss on log
predictions and log ground-truth losses, optimized with Adam. By default this
script fits on the cosine schedule and validates on the WSD schedule, matching
the final-project requirement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / "results" / ".cache"
(CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


RUN_ALIASES = {
    "811": "M:100M_gpt_D:20B_scheduler:811_rope",
    "wsd": "M:100M_gpt_D:20B_scheduler:wsd_rope",
    "cosine": "M:100M_gpt_D:20B_scheduler:cosine_rope",
}

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "gpt_loss+lrs.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "baselines" / "multi_power_law"
DEFAULT_BASELINE_SUMMARY_PATH = PROJECT_ROOT / "results" / "baselines" / "momentum" / "summary.json"


@dataclass
class Curve:
    alias: str
    run_name: str
    observed_step: np.ndarray
    observed_loss: np.ndarray
    observed_lr: np.ndarray
    full_step: np.ndarray
    full_lr: np.ndarray
    s1: np.ndarray
    missing_steps: list[int]


@dataclass
class FeatureBatch:
    alias: str
    steps: np.ndarray
    s1: torch.Tensor
    loss: torch.Tensor
    tail_lr_sum: torch.Tensor
    lr: torch.Tensor
    reductions: torch.Tensor


@dataclass
class FitResult:
    params: dict[str, float]
    objective_value: float
    history: list[dict[str, float]]
    optimizer_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the Multi-Power Law on cosine GPT loss data and validate on WSD."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=DEFAULT_BASELINE_SUMMARY_PATH,
        help="Momentum baseline summary used for direct metric comparison.",
    )
    parser.add_argument("--fit-runs", default="cosine")
    parser.add_argument("--eval-runs", default="cosine,wsd")
    parser.add_argument("--sample-start", type=int, default=1000)
    parser.add_argument("--sample-end", type=int, default=None)
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=128,
        help="Use every n-th observed step for fitting and validation metrics.",
    )
    parser.add_argument(
        "--plot-stride",
        type=int,
        default=32,
        help="Use every n-th observed step for plotting predictions.",
    )
    parser.add_argument("--huber-delta", type=float, default=1e-3)
    parser.add_argument(
        "--adam-steps",
        type=int,
        default=1500,
        help="First Adam stage. Paper setting is 50000.",
    )
    parser.add_argument(
        "--finetune-steps",
        type=int,
        default=500,
        help="Second low-learning-rate Adam stage. Paper setting is 50000.",
    )
    parser.add_argument("--coeff-lr", type=float, default=5e-2)
    parser.add_argument("--exp-lr", type=float, default=5e-3)
    parser.add_argument("--finetune-coeff-lr", type=float, default=1e-5)
    parser.add_argument("--finetune-exp-lr", type=float, default=1e-6)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Rows per block when computing MPL predictions.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--list-runs", action="store_true")
    return parser.parse_args()


def starts_with_explicit_relative(path: Path) -> bool:
    return bool(path.parts) and path.parts[0] in {".", ".."}


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()

    candidates: list[Path]
    if starts_with_explicit_relative(path):
        candidates = [(Path.cwd() / path).resolve(), (PROJECT_ROOT / path).resolve()]
    else:
        candidates = [(PROJECT_ROOT / path).resolve(), (Path.cwd() / path).resolve()]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0] == "..":
        return (Path(__file__).resolve().parent / path).resolve()
    if path.parts and path.parts[0] == ".":
        return (PROJECT_ROOT / Path(*path.parts[1:])).resolve()
    return (PROJECT_ROOT / path).resolve()


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def split_aliases(value: str) -> list[str]:
    aliases = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(aliases) - set(RUN_ALIASES))
    if unknown:
        raise ValueError(f"Unknown run aliases: {unknown}. Available: {sorted(RUN_ALIASES)}")
    return aliases


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return torch.device(value)


def load_curves(data_path: Path, aliases: Iterable[str]) -> dict[str, Curve]:
    raw = pd.read_pickle(data_path)
    curves: dict[str, Curve] = {}

    for alias in aliases:
        run_name = RUN_ALIASES[alias]
        if run_name not in raw:
            raise KeyError(f"Run {run_name!r} not found in {data_path}")

        df = raw[run_name].copy()
        required = {"step", "lr", "Metrics/loss"}
        missing_columns = required - set(df.columns)
        if missing_columns:
            raise ValueError(f"Run {run_name!r} misses columns: {sorted(missing_columns)}")

        df = df.sort_values("step")
        df = df[np.isfinite(df["step"]) & np.isfinite(df["lr"]) & np.isfinite(df["Metrics/loss"])]
        df["step"] = df["step"].astype(int)
        if df["step"].duplicated().any():
            duplicated = df.loc[df["step"].duplicated(), "step"].head().tolist()
            raise ValueError(f"Run {run_name!r} contains duplicated steps, e.g. {duplicated}")
        if int(df["step"].min()) != 0:
            raise ValueError(f"Run {run_name!r} must start at step 0 for this reproduction script")

        max_step = int(df["step"].max())
        full_step = np.arange(max_step + 1, dtype=np.int64)
        indexed = df.set_index("step")
        lr_series = indexed["lr"].reindex(full_step)
        missing_steps = [int(step) for step in full_step[lr_series.isna().to_numpy()]]
        full_lr = (
            lr_series.interpolate(method="linear", limit_direction="both")
            .ffill()
            .bfill()
            .to_numpy(dtype=np.float64)
        )

        curves[alias] = Curve(
            alias=alias,
            run_name=run_name,
            observed_step=df["step"].to_numpy(dtype=np.int64),
            observed_loss=df["Metrics/loss"].to_numpy(dtype=np.float64),
            observed_lr=df["lr"].to_numpy(dtype=np.float64),
            full_step=full_step,
            full_lr=full_lr.copy(),
            s1=np.cumsum(full_lr).astype(np.float64),
            missing_steps=missing_steps,
        )

    return curves


def observed_step_set(curve: Curve) -> set[int]:
    return set(int(step) for step in curve.observed_step)


def infer_sample_end(curves: dict[str, Curve], aliases: list[str], start: int, interval: int) -> int:
    max_common = min(int(curves[alias].observed_step.max()) for alias in aliases)
    if max_common < start:
        raise ValueError(f"No sampled steps available: max common step {max_common} < start {start}")
    return start + ((max_common - start) // interval) * interval


def make_sample_steps(curve: Curve, start: int, end: int, interval: int) -> np.ndarray:
    if interval <= 0:
        raise ValueError("sample_interval must be positive")
    if end < start:
        raise ValueError("sample_end must be greater than or equal to sample_start")

    wanted = np.arange(start, end + 1, interval, dtype=np.int64)
    available = observed_step_set(curve)
    return np.array([int(step) for step in wanted if int(step) in available], dtype=np.int64)


def steps_to_loss(curve: Curve, steps: np.ndarray) -> np.ndarray:
    loss_by_step = dict(zip(curve.observed_step.tolist(), curve.observed_loss.tolist(), strict=True))
    missing = [int(step) for step in steps if int(step) not in loss_by_step]
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"Run {curve.alias!r} has no observed loss for steps {preview}"
            + (" ..." if len(missing) > len(preview) else "")
        )
    return np.array([loss_by_step[int(step)] for step in steps], dtype=np.float64)


def build_tail_lr_sum(curve: Curve, steps: np.ndarray) -> np.ndarray:
    indices = steps.astype(np.int64)
    n = len(curve.full_lr)
    previous_s1 = np.concatenate(([0.0], curve.s1[:-1]))
    tail = curve.s1[indices, None] - previous_s1[None, :]
    mask = np.arange(n, dtype=np.int64)[None, :] <= indices[:, None]
    tail = np.where(mask, tail, 0.0)
    return np.maximum(tail, 0.0).astype(np.float64, copy=False)


def build_feature_batch(curve: Curve, steps: np.ndarray, device: torch.device) -> FeatureBatch:
    reductions = np.zeros_like(curve.full_lr, dtype=np.float64)
    reductions[1:] = np.maximum(curve.full_lr[:-1] - curve.full_lr[1:], 0.0)

    return FeatureBatch(
        alias=curve.alias,
        steps=steps,
        s1=torch.as_tensor(curve.s1[steps], dtype=torch.float64, device=device),
        loss=torch.as_tensor(steps_to_loss(curve, steps), dtype=torch.float64, device=device),
        tail_lr_sum=torch.as_tensor(build_tail_lr_sum(curve, steps), dtype=torch.float64, device=device),
        lr=torch.as_tensor(curve.full_lr, dtype=torch.float64, device=device),
        reductions=torch.as_tensor(reductions, dtype=torch.float64, device=device),
    )


def huber_loss(residual: torch.Tensor, delta: float) -> torch.Tensor:
    abs_residual = torch.abs(residual)
    return torch.where(
        abs_residual < delta,
        0.5 * residual**2,
        delta * abs_residual - 0.5 * delta**2,
    )


def inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-12)
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


def inverse_scaled_sigmoid(value: float, upper: float) -> float:
    ratio = min(max(float(value) / upper, 1e-6), 1.0 - 1e-6)
    return math.log(ratio / (1.0 - ratio))


class MultiPowerLawModel(torch.nn.Module):
    def __init__(self, init_params: dict[str, float]):
        super().__init__()
        self.raw_L0 = torch.nn.Parameter(torch.tensor(inverse_softplus(init_params["L0"]), dtype=torch.float64))
        self.raw_A = torch.nn.Parameter(torch.tensor(inverse_softplus(init_params["A"]), dtype=torch.float64))
        self.raw_B = torch.nn.Parameter(torch.tensor(inverse_softplus(init_params["B"]), dtype=torch.float64))
        self.raw_C = torch.nn.Parameter(torch.tensor(inverse_softplus(init_params["C"]), dtype=torch.float64))
        self.raw_alpha = torch.nn.Parameter(
            torch.tensor(inverse_scaled_sigmoid(init_params["alpha"], 2.0), dtype=torch.float64)
        )
        self.raw_beta = torch.nn.Parameter(
            torch.tensor(inverse_scaled_sigmoid(init_params["beta"], 1.0), dtype=torch.float64)
        )
        self.raw_gamma = torch.nn.Parameter(
            torch.tensor(inverse_scaled_sigmoid(init_params["gamma"], 1.0), dtype=torch.float64)
        )

    def positive_params(self) -> dict[str, torch.Tensor]:
        eps = torch.tensor(1e-12, dtype=torch.float64, device=self.raw_L0.device)
        return {
            "L0": torch.nn.functional.softplus(self.raw_L0) + eps,
            "A": torch.nn.functional.softplus(self.raw_A) + eps,
            "B": torch.nn.functional.softplus(self.raw_B) + eps,
            "C": torch.nn.functional.softplus(self.raw_C) + eps,
            "alpha": 2.0 * torch.sigmoid(self.raw_alpha) + eps,
            "beta": torch.sigmoid(self.raw_beta) + eps,
            "gamma": torch.sigmoid(self.raw_gamma) + eps,
        }

    def parameter_groups(self) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        coeff_params = [self.raw_L0, self.raw_A, self.raw_B, self.raw_C]
        exp_params = [self.raw_alpha, self.raw_beta, self.raw_gamma]
        return coeff_params, exp_params

    def predict_batch(self, batch: FeatureBatch) -> torch.Tensor:
        params = self.positive_params()
        lr_power = torch.pow(torch.clamp(batch.lr, min=1e-12), -params["gamma"])
        x = batch.tail_lr_sum * lr_power
        g = 1.0 - torch.pow(1.0 + params["C"] * x, -params["beta"])
        ld = torch.sum(g * batch.reductions, dim=1)
        return params["L0"] + params["A"] * torch.pow(batch.s1, -params["alpha"]) - params["B"] * ld

    def as_float_dict(self) -> dict[str, float]:
        return {name: float(value.detach().cpu()) for name, value in self.positive_params().items()}


def initial_params_from_curve(curve: Curve, sample_steps: np.ndarray) -> dict[str, float]:
    s1 = curve.s1[sample_steps]
    loss = steps_to_loss(curve, sample_steps)
    min_loss = float(np.min(loss))
    l0 = max(1e-4, min_loss - 0.2)
    adjusted_loss = np.maximum(loss - l0, 1e-6)
    slope, intercept = np.polyfit(np.log(s1), np.log(adjusted_loss), deg=1)
    alpha = float(np.clip(-slope, 0.05, 1.5))
    a = float(max(np.exp(intercept), 1e-6))
    return {
        "L0": l0,
        "A": a,
        "B": 500.0,
        "C": 1.0,
        "alpha": alpha,
        "beta": 0.5,
        "gamma": 0.5,
    }


def objective(model: MultiPowerLawModel, batches: list[FeatureBatch], huber_delta: float) -> torch.Tensor:
    total = torch.tensor(0.0, dtype=torch.float64, device=next(model.parameters()).device)
    for batch in batches:
        pred = model.predict_batch(batch)
        invalid_penalty = torch.relu(1e-10 - pred).pow(2).sum() * 1e8
        residual = torch.log(torch.clamp(pred, min=1e-10)) - torch.log(batch.loss)
        total = total + huber_loss(residual, huber_delta).sum() + invalid_penalty
    return total


class SimpleAdam:
    """Small Adam fallback for environments where importing torch.optim fails."""

    def __init__(
        self,
        parameter_groups: list[dict[str, object]],
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.parameter_groups = parameter_groups
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.state: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def zero_grad(self) -> None:
        for group in self.parameter_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad = None

    def step(self) -> None:
        self.step_count += 1
        for group in self.parameter_groups:
            lr = float(group["lr"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state_key = id(param)
                if state_key not in self.state:
                    self.state[state_key] = (torch.zeros_like(param), torch.zeros_like(param))
                first_moment, second_moment = self.state[state_key]
                first_moment.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
                second_moment.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
                first_unbiased = first_moment / (1.0 - self.beta1**self.step_count)
                second_unbiased = second_moment / (1.0 - self.beta2**self.step_count)
                param.data.addcdiv_(first_unbiased, second_unbiased.sqrt().add_(self.eps), value=-lr)


def make_adam_optimizer(
    coeff_params: list[torch.nn.Parameter],
    exp_params: list[torch.nn.Parameter],
    lr_coeff: float,
    lr_exp: float,
):
    parameter_groups = [
        {"params": coeff_params, "lr": lr_coeff},
        {"params": exp_params, "lr": lr_exp},
    ]
    try:
        optimizer = torch.optim.Adam(parameter_groups)
        return optimizer, "torch.optim.Adam"
    except Exception as exc:
        print(f"Falling back to SimpleAdam because torch.optim.Adam failed: {exc}")
        return SimpleAdam(parameter_groups), "SimpleAdam"


def fit_mpl(
    batches: list[FeatureBatch],
    init_params: dict[str, float],
    huber_delta: float,
    adam_steps: int,
    finetune_steps: int,
    coeff_lr: float,
    exp_lr: float,
    finetune_coeff_lr: float,
    finetune_exp_lr: float,
) -> FitResult:
    model = MultiPowerLawModel(init_params)
    device = batches[0].s1.device
    model.to(device)
    history: list[dict[str, float]] = []
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_loss = float("inf")
    optimizer_names: list[str] = []

    phases = [
        ("adam", adam_steps, coeff_lr, exp_lr),
        ("finetune", finetune_steps, finetune_coeff_lr, finetune_exp_lr),
    ]

    global_step = 0
    for phase_name, steps, lr_coeff, lr_exp in phases:
        if steps <= 0:
            continue
        coeff_params, exp_params = model.parameter_groups()
        optimizer, optimizer_name = make_adam_optimizer(
            coeff_params,
            exp_params,
            lr_coeff,
            lr_exp,
        )
        optimizer_names.append(optimizer_name)

        iterator = tqdm(range(steps), desc=f"MPL {phase_name}", ascii=True)
        for _ in iterator:
            if optimizer_name == "torch.optim.Adam":
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad()
            loss = objective(model, batches, huber_delta)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            if np.isfinite(loss_value) and loss_value < best_loss:
                best_loss = loss_value
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

            if global_step % 25 == 0 or global_step == adam_steps + finetune_steps - 1:
                row = {"step": float(global_step), "objective": loss_value, "best_objective": best_loss}
                row.update(model.as_float_dict())
                history.append(row)
                iterator.set_postfix(loss=f"{loss_value:.4g}", best=f"{best_loss:.4g}")
            global_step += 1

    model.load_state_dict(best_state)
    return FitResult(
        params=model.as_float_dict(),
        objective_value=best_loss,
        history=history,
        optimizer_name=",".join(dict.fromkeys(optimizer_names)),
    )


def predict_steps(
    params: dict[str, float],
    curve: Curve,
    steps: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    reductions = np.zeros_like(curve.full_lr, dtype=np.float64)
    reductions[1:] = np.maximum(curve.full_lr[:-1] - curve.full_lr[1:], 0.0)
    lr_power = np.power(np.clip(curve.full_lr, 1e-12, None), -params["gamma"])
    preds = []

    chunk_size = max(1, chunk_size)
    for start in range(0, len(steps), chunk_size):
        chunk_steps = steps[start : start + chunk_size].astype(np.int64)
        tail = build_tail_lr_sum(curve, chunk_steps)
        x = tail * lr_power[None, :]
        g = 1.0 - np.power(1.0 + params["C"] * x, -params["beta"])
        ld = np.sum(g * reductions[None, :], axis=1)
        pred = params["L0"] + params["A"] * np.power(curve.s1[chunk_steps], -params["alpha"]) - params["B"] * ld
        preds.append(pred.astype(np.float64))

    return np.concatenate(preds)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        raise ValueError("No valid points available for metric computation.")
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    mse = float(np.mean(residual**2))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "mae": mae,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mean_relative_error": float(np.mean(np.abs(residual) / y_true)),
        "worst_relative_error": float(np.max(np.abs(residual) / y_true)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "endpoint_true": float(y_true[-1]),
        "endpoint_pred": float(y_pred[-1]),
        "endpoint_diff": float(y_pred[-1] - y_true[-1]),
        "endpoint_abs_diff": float(abs(y_pred[-1] - y_true[-1])),
    }


def load_baseline_metrics(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        summary = json.load(f)

    rows = summary.get("metrics")
    if rows is None and summary.get("fits"):
        rows = summary["fits"][-1].get("metrics", [])
    if not isinstance(rows, list):
        return {}

    baseline: dict[str, dict[str, float]] = {}
    for row in rows:
        if not isinstance(row, dict) or "run" not in row:
            continue
        baseline[str(row["run"])] = row
    return baseline


def compare_with_baseline(
    metrics: pd.DataFrame,
    baseline_metrics: dict[str, dict[str, float]],
) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    lower_is_better = {"mae", "mse", "rmse", "endpoint_abs_diff"}
    higher_is_better = {"r2"}

    for row in metrics.to_dict(orient="records"):
        run = str(row["run"])
        baseline = baseline_metrics.get(run)
        if baseline is None:
            continue

        comparison: dict[str, object] = {"run": run}
        for key in ["mae", "mse", "rmse", "r2", "endpoint_abs_diff"]:
            if key not in row or key not in baseline:
                continue
            value = row[key]
            baseline_value = baseline[key]
            if value is None or baseline_value is None:
                continue
            if not np.isfinite(float(value)) or not np.isfinite(float(baseline_value)):
                continue

            delta = float(value) - float(baseline_value)
            comparison[f"mpl_{key}"] = float(value)
            comparison[f"momentum_{key}"] = float(baseline_value)
            comparison[f"{key}_delta_vs_momentum"] = delta
            if key in lower_is_better:
                comparison[f"beats_momentum_{key}"] = delta < 0
                if float(baseline_value) != 0.0:
                    comparison[f"{key}_relative_improvement"] = (
                        (float(baseline_value) - float(value)) / abs(float(baseline_value))
                    )
            elif key in higher_is_better:
                comparison[f"beats_momentum_{key}"] = delta > 0
        comparisons.append(comparison)

    return comparisons


def evaluate_and_collect_predictions(
    curves: dict[str, Curve],
    aliases: list[str],
    fit_aliases: list[str],
    params: dict[str, float],
    sample_start: int,
    sample_end: int,
    sample_interval: int,
    plot_stride: int,
    chunk_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_frames = []
    plot_stride = max(1, plot_stride)

    for alias in aliases:
        curve = curves[alias]
        metric_steps = make_sample_steps(curve, sample_start, sample_end, sample_interval)
        metric_loss = steps_to_loss(curve, metric_steps)
        metric_pred = predict_steps(params, curve, metric_steps, chunk_size)
        metrics = regression_metrics(metric_loss, metric_pred)
        metric_rows.append(
            {
                "run": alias,
                "full_run_name": curve.run_name,
                "split": "fit" if alias in fit_aliases else "validation",
                "n_metric_points": len(metric_steps),
                "metric_step_start": int(metric_steps[0]),
                "metric_step_end": int(metric_steps[-1]),
                **params,
                **metrics,
            }
        )

        plot_steps = curve.observed_step[::plot_stride]
        if plot_steps[-1] != curve.observed_step[-1]:
            plot_steps = np.concatenate([plot_steps, curve.observed_step[-1:]])
        plot_pred = predict_steps(params, curve, plot_steps, chunk_size)
        plot_loss = steps_to_loss(curve, plot_steps)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "run": alias,
                    "step": plot_steps,
                    "lr": curve.full_lr[plot_steps],
                    "loss": plot_loss,
                    "prediction": plot_pred,
                    "is_fit_run": alias in fit_aliases,
                    "is_metric_step": np.isin(plot_steps, metric_steps),
                }
            )
        )

    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def save_figures(
    curves: dict[str, Curve],
    predictions: pd.DataFrame,
    eval_aliases: list[str],
    fit_aliases: list[str],
    output_dir: Path,
    history: list[dict[str, float]],
    params: dict[str, float],
    sample_start: int,
) -> None:
    colors = {"cosine": "C0", "wsd": "C1", "811": "C2"}

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for alias in eval_aliases:
        curve = curves[alias]
        group = predictions[predictions["run"] == alias]
        color = colors.get(alias)
        split = "fit" if alias in fit_aliases else "validation"
        raw_idx = curve.observed_step >= sample_start
        pred_group = group[group["step"] >= sample_start]
        axes[0].plot(
            curve.observed_step[raw_idx],
            curve.observed_loss[raw_idx],
            linewidth=0.8,
            alpha=0.2,
            color=color,
            label=f"{alias} raw loss ({split})",
        )
        axes[0].plot(
            pred_group["step"],
            pred_group["prediction"],
            linewidth=1.9,
            linestyle="--",
            color=color,
            label=f"{alias} MPL prediction",
        )
        axes[1].plot(group["step"], group["lr"], linewidth=1.5, color=color, label=alias)

    param_text = "\n".join(
        [
            "Fitted MPL parameters",
            f"L0={params['L0']:.6g}, A={params['A']:.6g}, B={params['B']:.6g}",
            f"C={params['C']:.6g}, alpha={params['alpha']:.6g}",
            f"beta={params['beta']:.6g}, gamma={params['gamma']:.6g}",
        ]
    )
    axes[0].text(
        0.99,
        0.97,
        param_text,
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.88},
    )
    axes[0].set_title("Multi-Power Law fitted on cosine and validated on WSD")
    axes[0].set_ylabel("Loss")
    axes[0].set_xlim(0, None)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_ylabel("Learning rate")
    axes[1].set_xlabel("Step")
    axes[1].set_xlim(0, None)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "mpl_fit_prediction.png", dpi=200)
    plt.close(fig)

    if history:
        history_df = pd.DataFrame(history)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history_df["step"], history_df["objective"], label="objective")
        ax.plot(history_df["step"], history_df["best_objective"], label="best objective")
        ax.set_xlabel("Adam step")
        ax.set_ylabel("Summed Huber loss")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "loss_monitor.png", dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = resolve_input_path(args.data_path)
    output_dir = resolve_output_path(args.output_dir)
    baseline_summary_path = resolve_input_path(args.baseline_summary)
    device = choose_device(args.device)

    raw = pd.read_pickle(data_path)
    if args.list_runs:
        print("Available runs:")
        for run_name, frame in raw.items():
            print(f"- {run_name}: shape={frame.shape}, columns={list(frame.columns)}")
        return

    fit_aliases = split_aliases(args.fit_runs)
    eval_aliases = split_aliases(args.eval_runs)
    all_aliases = sorted(set(fit_aliases + eval_aliases), key=(fit_aliases + eval_aliases).index)

    output_dir.mkdir(parents=True, exist_ok=True)
    curves = load_curves(data_path, all_aliases)
    sample_end = args.sample_end
    if sample_end is None:
        sample_end = infer_sample_end(curves, all_aliases, args.sample_start, args.sample_interval)

    train_batches: list[FeatureBatch] = []
    fit_sample_steps: dict[str, list[int]] = {}
    for alias in fit_aliases:
        steps = make_sample_steps(curves[alias], args.sample_start, sample_end, args.sample_interval)
        if len(steps) == 0:
            raise ValueError(f"No sampled fit steps for run {alias!r}")
        fit_sample_steps[alias] = steps.tolist()
        train_batches.append(build_feature_batch(curves[alias], steps, device))

    init_params = initial_params_from_curve(curves[fit_aliases[0]], np.array(fit_sample_steps[fit_aliases[0]]))
    fit_result = fit_mpl(
        batches=train_batches,
        init_params=init_params,
        huber_delta=args.huber_delta,
        adam_steps=args.adam_steps,
        finetune_steps=args.finetune_steps,
        coeff_lr=args.coeff_lr,
        exp_lr=args.exp_lr,
        finetune_coeff_lr=args.finetune_coeff_lr,
        finetune_exp_lr=args.finetune_exp_lr,
    )

    metrics, predictions = evaluate_and_collect_predictions(
        curves=curves,
        aliases=eval_aliases,
        fit_aliases=fit_aliases,
        params=fit_result.params,
        sample_start=args.sample_start,
        sample_end=sample_end,
        sample_interval=args.sample_interval,
        plot_stride=args.plot_stride,
        chunk_size=args.chunk_size,
    )

    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    if fit_result.history:
        pd.DataFrame(fit_result.history).to_csv(output_dir / "training_history.csv", index=False)
    save_figures(
        curves,
        predictions,
        eval_aliases,
        fit_aliases,
        output_dir,
        fit_result.history,
        fit_result.params,
        args.sample_start,
    )
    baseline_metrics = load_baseline_metrics(baseline_summary_path)
    baseline_comparison = compare_with_baseline(metrics, baseline_metrics)

    summary = {
        "paper": "A multi-power law for loss curve prediction across learning rate schedules",
        "model": (
            "L(t)=L0 + A*S1(t)^(-alpha) - B*sum_k (eta_{k-1}-eta_k)"
            "*(1-(1+C*eta_k^(-gamma)*S_k(t))^(-beta))"
        ),
        "training_method": {
            "objective": "sum_t Huber_delta(log L_pred(t) - log L_true(t))",
            "optimizer": "Adam",
            "implementation": fit_result.optimizer_name,
            "paper_learning_rates": {
                "coefficient_or_constant_params": 5e-2,
                "exponent_params": 5e-3,
                "second_stage_coefficient_or_constant_params": 1e-5,
                "second_stage_exponent_params": 1e-6,
                "paper_steps_per_stage": 50000,
            },
            "actual_steps": {
                "adam_steps": args.adam_steps,
                "finetune_steps": args.finetune_steps,
            },
            "huber_delta": args.huber_delta,
        },
        "fit_runs": fit_aliases,
        "eval_runs": eval_aliases,
        "data_path": display_path(data_path),
        "output_dir": display_path(output_dir),
        "baseline_summary": display_path(baseline_summary_path) if baseline_summary_path.exists() else None,
        "sampling": {
            "sample_start": args.sample_start,
            "sample_end": sample_end,
            "sample_interval": args.sample_interval,
            "fit_sample_steps": fit_sample_steps,
        },
        "filled_lr_only_for_missing_steps": {
            alias: curves[alias].missing_steps for alias in all_aliases if curves[alias].missing_steps
        },
        "initial_params": init_params,
        "fitted_params": fit_result.params,
        "objective_value": fit_result.objective_value,
        "metrics": metrics.to_dict(orient="records"),
        "baseline_comparison": baseline_comparison,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Fitted Multi-Power Law parameters:")
    for name in ["L0", "A", "B", "C", "alpha", "beta", "gamma"]:
        print(f"  {name}={fit_result.params[name]:.10g}")
    print(f"Objective: {fit_result.objective_value:.10g}")
    print("\nEvaluation metrics:")
    display_cols = [
        "run",
        "split",
        "n_metric_points",
        "mae",
        "rmse",
        "mean_relative_error",
        "worst_relative_error",
        "r2",
        "endpoint_true",
        "endpoint_pred",
        "endpoint_abs_diff",
    ]
    print(metrics[display_cols].to_string(index=False))
    if baseline_comparison:
        print("\nComparison with momentum baseline:")
        comparison_df = pd.DataFrame(baseline_comparison)
        comparison_cols = [
            "run",
            "mpl_mse",
            "momentum_mse",
            "mse_delta_vs_momentum",
            "mse_relative_improvement",
            "beats_momentum_mse",
            "mpl_r2",
            "momentum_r2",
            "r2_delta_vs_momentum",
            "beats_momentum_r2",
            "mpl_endpoint_abs_diff",
            "momentum_endpoint_abs_diff",
            "endpoint_abs_diff_delta_vs_momentum",
            "beats_momentum_endpoint_abs_diff",
        ]
        comparison_cols = [col for col in comparison_cols if col in comparison_df.columns]
        print(comparison_df[comparison_cols].to_string(index=False))
    print(f"\nSaved outputs to: {display_path(output_dir)}")


if __name__ == "__main__":
    main()
