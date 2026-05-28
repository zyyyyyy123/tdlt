"""Reproduce the momentum-law baseline from Tissue et al. on local loss curves.

Default experiment:
  fit on cosine LRS, evaluate on WSD LRS.

The implemented model is
    L(s) = L0 + A * S1(s)^(-alpha) - C * S2(s),
where S1 is cumulative learning rate and S2 is the cumulative momentum of
learning-rate decreases.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm


RUN_ALIASES = {
    "811": "M:100M_gpt_D:20B_scheduler:811_rope",
    "wsd": "M:100M_gpt_D:20B_scheduler:wsd_rope",
    "cosine": "M:100M_gpt_D:20B_scheduler:cosine_rope",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "loss curves" / "gpt_loss+lrs.pkl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "reproduction" / "momentum"


@dataclass
class Curve:
    alias: str
    run_name: str
    step: np.ndarray
    lr: np.ndarray
    loss: np.ndarray
    s1: np.ndarray
    s2: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate the LR-annealing momentum scaling law."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the local pickle file containing loss curves.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics, predictions, and figures.",
    )
    parser.add_argument(
        "--fit-runs",
        default="cosine",
        help="Comma-separated aliases used for fitting. Available: cosine, wsd, 811.",
    )
    parser.add_argument(
        "--eval-runs",
        default="cosine,wsd",
        help="Comma-separated aliases used for evaluation.",
    )
    parser.add_argument(
        "--sample-start",
        type=int,
        default=1000,
        help="First sampled step.",
    )
    parser.add_argument(
        "--sample-end",
        type=int,
        default=None,
        help="Last sampled step, inclusive. Default: maximum common step across fit/eval curves.",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=2,
        help="Step interval for sampled fitting/evaluation points.",
    )
    parser.add_argument(
        "--plot-stride",
        type=int,
        default=1,
        help="Use every n-th full-resolution point when drawing prediction curves.",
    )
    parser.add_argument(
        "--ema-span",
        type=int,
        default=99,
        help="EMA span, in raw points, used to smooth loss curves in the final plot.",
    )
    parser.add_argument(
        "--decay-factor",
        type=float,
        default=0.999,
        help="Momentum decay lambda used in S2.",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=1e-3,
        help="Huber delta used on log residuals, matching the reference code.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=100000,
        help="Maximum L-BFGS-B iterations for each initialization.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="Only list available runs in the local pickle file.",
    )
    return parser.parse_args()


def starts_with_explicit_relative(path: Path) -> bool:
    return bool(path.parts) and path.parts[0] in {".", ".."}


def resolve_input_path(path: Path) -> Path:
    """Resolve input paths from either cwd or the project root."""
    if path.is_absolute():
        return path.resolve()

    candidates = []
    if starts_with_explicit_relative(path):
        candidates.extend([(Path.cwd() / path).resolve(), (PROJECT_ROOT / path).resolve()])
    else:
        candidates.extend([(PROJECT_ROOT / path).resolve(), (Path.cwd() / path).resolve()])

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_output_path(path: Path) -> Path:
    """Resolve output paths without relying on cwd.

    `conda run` can expose mojibake cwd strings on Windows when the project path
    contains Chinese characters, so project-local relative paths are resolved
    from this script's location instead of from the process cwd.
    """
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


def huber_loss(residual: np.ndarray, delta: float) -> np.ndarray:
    abs_residual = np.abs(residual)
    return np.where(
        abs_residual < delta,
        0.5 * residual**2,
        delta * abs_residual - 0.5 * delta**2,
    )


def compute_s1_s2(lr: np.ndarray, decay_factor: float) -> tuple[np.ndarray, np.ndarray]:
    s1 = np.cumsum(lr).astype(np.float64)
    momentum = np.zeros_like(lr, dtype=np.float64)
    for i in range(1, len(lr)):
        momentum[i] = decay_factor * momentum[i - 1] + (lr[i - 1] - lr[i])
    s2 = np.cumsum(momentum).astype(np.float64)
    return s1, s2


def load_curves(
    data_path: Path,
    aliases: Iterable[str],
    decay_factor: float,
) -> dict[str, Curve]:
    raw = pd.read_pickle(data_path)
    curves: dict[str, Curve] = {}

    for alias in aliases:
        run_name = RUN_ALIASES[alias]
        if run_name not in raw:
            raise KeyError(f"Run {run_name!r} not found in {data_path}")

        df = raw[run_name].copy()
        required = {"step", "lr", "Metrics/loss"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Run {run_name!r} misses columns: {sorted(missing)}")

        df = df.sort_values("step")
        df = df[np.isfinite(df["step"]) & np.isfinite(df["lr"]) & np.isfinite(df["Metrics/loss"])]
        df = df.reset_index(drop=True)
        if df.empty:
            raise ValueError(f"Run {run_name!r} has no valid data")

        lr = df["lr"].to_numpy(dtype=np.float64)
        s1, s2 = compute_s1_s2(lr, decay_factor)
        curves[alias] = Curve(
            alias=alias,
            run_name=run_name,
            step=df["step"].to_numpy(dtype=np.int64),
            lr=lr,
            loss=df["Metrics/loss"].to_numpy(dtype=np.float64),
            s1=s1,
            s2=s2,
        )

    return curves


def predict_loss(params: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    l0, a, c, alpha = params
    return l0 + a * np.power(s1, -alpha) - c * s2


def exponential_moving_average(values: np.ndarray, span: int) -> np.ndarray:
    if span <= 1:
        return values.astype(np.float64, copy=True)

    alpha = 2.0 / (span + 1.0)
    ema = np.empty_like(values, dtype=np.float64)
    ema[0] = float(values[0])
    for i in range(1, len(values)):
        ema[i] = alpha * float(values[i]) + (1.0 - alpha) * ema[i - 1]
    return ema


def sampled_indices(curve: Curve, sample_steps: np.ndarray) -> np.ndarray:
    step_to_index = {int(step): i for i, step in enumerate(curve.step)}
    missing = [int(step) for step in sample_steps if int(step) not in step_to_index]
    if missing:
        preview = missing[:5]
        raise ValueError(
            f"Run {curve.alias!r} is missing sampled steps {preview}"
            + (" ..." if len(missing) > len(preview) else "")
        )
    return np.array([step_to_index[int(step)] for step in sample_steps], dtype=np.int64)


def infer_sample_end(curves: dict[str, Curve], aliases: list[str], start: int, interval: int) -> int:
    max_common_step = min(int(curves[alias].step.max()) for alias in aliases)
    if max_common_step < start:
        raise ValueError(f"No sampled steps available: max common step {max_common_step} < start {start}")
    return start + ((max_common_step - start) // interval) * interval


def make_sample_steps(start: int, end: int, interval: int) -> np.ndarray:
    if interval <= 0:
        raise ValueError("sample_interval must be positive")
    if end < start:
        raise ValueError("sample_end must be greater than or equal to sample_start")
    return np.arange(start, end + 1, interval, dtype=np.int64)


def build_fit_arrays(curves: dict[str, Curve], aliases: list[str], sample_steps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s1_parts = []
    s2_parts = []
    loss_parts = []

    for alias in aliases:
        curve = curves[alias]
        idx = sampled_indices(curve, sample_steps)
        s1_parts.append(curve.s1[idx])
        s2_parts.append(curve.s2[idx])
        loss_parts.append(curve.loss[idx])

    return np.concatenate(s1_parts), np.concatenate(s2_parts), np.concatenate(loss_parts)


def initial_parameter_grid() -> list[tuple[float, float, float, float]]:
    """Initialization grid used by the original reference implementation."""
    l0_values = np.linspace(0.1, 2.1, 2)
    a_values = np.linspace(1, 22, 3)
    c_values = np.linspace(1, 22, 3)
    alpha_values = np.linspace(0, 0.8, 3)
    return list(product(l0_values, a_values, c_values, alpha_values))


def fit_momentum_law(
    s1: np.ndarray,
    s2: np.ndarray,
    loss: np.ndarray,
    huber_delta: float,
    maxiter: int,
) -> tuple[np.ndarray, float]:
    def objective(params: np.ndarray) -> float:
        pred = predict_loss(params, s1, s2)
        if np.any(~np.isfinite(pred)) or np.any(pred <= 0):
            bad = pred[~np.isfinite(pred) | (pred <= 0)]
            penalty = 1e6 if bad.size == 0 else 1e6 + float(np.sum(np.square(np.minimum(bad, 0.0))))
            return penalty
        residual = np.log(loss) - np.log(pred)
        return float(np.sum(huber_loss(residual, huber_delta)))

    best_params: np.ndarray | None = None
    best_loss = np.inf
    starts = initial_parameter_grid()

    for start in tqdm(starts, desc="Fitting momentum law", ascii=True):
        result = minimize(
            objective,
            np.array(start, dtype=np.float64),
            method="L-BFGS-B",
            bounds=[(0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)],
            options={
                "maxiter": maxiter,
                "ftol": 1e-9,
                "gtol": 1e-6,
                "eps": 1e-8,
            },
        )
        if result.fun < best_loss:
            best_loss = float(result.fun)
            best_params = result.x.astype(np.float64)

    if best_params is None:
        raise RuntimeError("Optimization failed for all initializations.")
    return best_params, best_loss


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    mse = float(np.mean(residual**2))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    endpoint_true = float(y_true[-1])
    endpoint_pred = float(y_pred[-1])
    endpoint_diff = float(endpoint_pred - endpoint_true)
    endpoint_abs_diff = float(abs(endpoint_diff))
    return {
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "endpoint_true": endpoint_true,
        "endpoint_pred": endpoint_pred,
        "endpoint_diff": endpoint_diff,
        "endpoint_abs_diff": endpoint_abs_diff,
    }


def evaluate(
    curves: dict[str, Curve],
    aliases: list[str],
    params: np.ndarray,
    sample_steps: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    prediction_frames = []

    for alias in aliases:
        curve = curves[alias]
        pred_full = predict_loss(params, curve.s1, curve.s2)
        idx = sampled_indices(curve, sample_steps)
        pred_sample = pred_full[idx]

        metrics = regression_metrics(curve.loss[idx], pred_sample)
        metric_rows.append(
            {
                "run": alias,
                "full_run_name": curve.run_name,
                "n_points": len(idx),
                **metrics,
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "run": alias,
                    "step": curve.step,
                    "lr": curve.lr,
                    "loss": curve.loss,
                    "prediction": pred_full,
                    "is_sampled": np.isin(curve.step, sample_steps),
                    "s1": curve.s1,
                    "s2": curve.s2,
                }
            )
        )

    return pd.DataFrame(metric_rows), pd.concat(prediction_frames, ignore_index=True)


def save_figures(
    curves: dict[str, Curve],
    eval_aliases: list[str],
    fit_aliases: list[str],
    params: np.ndarray,
    output_dir: Path,
    sample_steps: np.ndarray,
    plot_stride: int,
    ema_span: int,
) -> None:
    plot_stride = max(1, plot_stride)
    ema_span = max(1, ema_span)
    colors = {"cosine": "C0", "wsd": "C1", "811": "C2"}

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for alias in eval_aliases:
        curve = curves[alias]
        full_plot_idx = np.arange(0, len(curve.step), plot_stride)
        loss_plot_idx = full_plot_idx[curve.step[full_plot_idx] >= sample_steps[0]]
        sample_idx = sampled_indices(curve, sample_steps)
        pred = predict_loss(params, curve.s1, curve.s2)
        ema_loss = exponential_moving_average(curve.loss, ema_span)
        label_suffix = "fit" if alias in fit_aliases else "eval"
        color = colors.get(alias, None)

        axes[0].plot(
            curve.step[loss_plot_idx],
            curve.loss[loss_plot_idx],
            linestyle="-",
            linewidth=0.8,
            alpha=0.2,
            color=color,
            label=f"{alias} raw loss",
        )
        # axes[0].plot(`
        #     curve.step[loss_plot_idx],
        #     ema_loss[loss_plot_idx],
        #     linestyle="-",
        #     linewidth=1.8,
        #     alpha=1.0,
        #     color=color,
        #     label=f"{alias} EMA loss (span={ema_span})",
        # )`
        # axes[0].scatter(
        #     curve.step[sample_idx],
        #     curve.loss[sample_idx],
        #     s=12,
        #     alpha=0.8,
        #     color=color,
        #     label=f"{alias} sampled loss ({label_suffix})",
        # )
        axes[0].plot(
            curve.step[loss_plot_idx],
            pred[loss_plot_idx],
            linestyle="--",
            linewidth=2.0,
            color=color,
            label=f"{alias} prediction",
        )
        axes[1].plot(curve.step[full_plot_idx], curve.lr[full_plot_idx], linewidth=1.5, color=color, label=alias)

    l0, a, c, alpha = params
    axes[0].set_title(
        f"Momentum law: L = {l0:.4g} + {a:.4g} * S1^(-{alpha:.4g}) - {c:.4g} * S2"
    )
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
    fig.savefig(output_dir / "momentum_fit_prediction.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for alias in eval_aliases:
        curve = curves[alias]
        plot_idx = np.arange(0, len(curve.step), plot_stride)
        color = colors.get(alias, None)
        axes[0].plot(curve.step[plot_idx], curve.s1[plot_idx], linewidth=1.5, color=color, label=alias)
        axes[1].plot(curve.step[plot_idx], curve.s2[plot_idx], linewidth=1.5, color=color, label=alias)
    axes[0].set_ylabel("S1")
    axes[0].set_xlim(0, None)
    axes[1].set_ylabel("S2")
    axes[1].set_xlabel("Step")
    axes[1].set_xlim(0, None)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(output_dir / "momentum_s1_s2.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = resolve_input_path(args.data_path)
    output_dir = resolve_output_path(args.output_dir)

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

    curves = load_curves(
        data_path=data_path,
        aliases=all_aliases,
        decay_factor=args.decay_factor,
    )
    sample_end = (
        infer_sample_end(curves, all_aliases, args.sample_start, args.sample_interval)
        if args.sample_end is None
        else args.sample_end
    )
    sample_steps = make_sample_steps(args.sample_start, sample_end, args.sample_interval)
    s1_fit, s2_fit, loss_fit = build_fit_arrays(curves, fit_aliases, sample_steps)
    params, objective_value = fit_momentum_law(
        s1=s1_fit,
        s2=s2_fit,
        loss=loss_fit,
        huber_delta=args.huber_delta,
        maxiter=args.maxiter,
    )

    metrics_df, predictions_df = evaluate(curves, eval_aliases, params, sample_steps)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    save_figures(
        curves=curves,
        eval_aliases=all_aliases,
        fit_aliases=fit_aliases,
        params=params,
        output_dir=output_dir,
        sample_steps=sample_steps,
        plot_stride=args.plot_stride,
        ema_span=args.ema_span,
    )

    summary = {
        "model": "L(s) = L0 + A * S1(s)^(-alpha) - C * S2(s)",
        "params": {
            "L0": float(params[0]),
            "A": float(params[1]),
            "C": float(params[2]),
            "alpha": float(params[3]),
        },
        "objective_value": objective_value,
        "fit_runs": fit_aliases,
        "eval_runs": eval_aliases,
        "metrics": metrics_df.to_dict(orient="records"),
        "data_path": display_path(data_path),
        "output_dir": display_path(output_dir),
        "sampling": {
            "sample_start": args.sample_start,
            "sample_end": sample_end,
            "sample_interval": args.sample_interval,
            "sampled_steps": [int(step) for step in sample_steps],
            "matches_reference_code": (
                args.sample_start == 1000
                and sample_end == 20000
                and args.sample_interval == 1000
            ),
            "auto_sample_end": args.sample_end is None,
        },
        "decay_factor": args.decay_factor,
        "huber_delta": args.huber_delta,
        "initialization": {
            "L0": "np.linspace(0.1, 2.1, 2)",
            "A": "np.linspace(1, 22, 3)",
            "C": "np.linspace(1, 22, 3)",
            "alpha": "np.linspace(0, 0.8, 3)",
        },
        "optimizer": {
            "method": "L-BFGS-B",
            "bounds": "(0, inf) for L0, A, C, alpha",
            "maxiter": args.maxiter,
            "ftol": 1e-9,
            "gtol": 1e-6,
            "eps": 1e-8,
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Fitted parameters:")
    print(
        f"  L0={params[0]:.8g}, A={params[1]:.8g}, "
        f"C={params[2]:.8g}, alpha={params[3]:.8g}"
    )
    print(f"Objective value: {objective_value:.8g}")
    print("\nEvaluation metrics:")
    display_cols = [
        "run",
        "mae",
        "mse",
        "r2",
        "endpoint_true",
        "endpoint_pred",
        "endpoint_diff",
        "endpoint_abs_diff",
    ]
    print(metrics_df[display_cols].to_string(index=False))
    print(f"\nSaved outputs to: {display_path(output_dir)}")


if __name__ == "__main__":
    main()
