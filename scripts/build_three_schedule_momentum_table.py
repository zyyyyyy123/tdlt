"""Build the three-schedule momentum baseline table used by residual methods.

This is a lightweight intermediate generator for the residual spline and
event/tail experiments. It does not train an MLP; it only applies the fitted
momentum-law baseline to cosine, 8-1-1, and WSD on the full step grid.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from reproduce_momentum import (
    DEFAULT_DATA_PATH,
    display_path,
    load_curves,
    resolve_input_path,
    resolve_output_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "intermediates" / "three_schedule_momentum"
SCHEDULES = ("cosine", "811", "wsd")
REPORT_MOMENTUM_PARAMS = np.array(
    [2.775526277107112, 1.1022055209072599, 0.18095388831549908, 1.007070154150098],
    dtype=np.float64,
)
REPORT_MOMENTUM_OBJECTIVE = 0.36022910023818566


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate full-grid momentum predictions for cosine, 8-1-1, and WSD."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-run", default="cosine", choices=SCHEDULES)
    parser.add_argument("--start-step", type=int, default=1000)
    parser.add_argument("--decay-factor", type=float, default=0.999)
    parser.add_argument("--huber-delta", type=float, default=1e-3)
    parser.add_argument("--maxiter", type=int, default=30000)
    parser.add_argument(
        "--refit",
        action="store_true",
        help=(
            "Refit the cosine momentum calibration instead of using the report "
            "intermediate parameters. The loss surface is flat enough that this "
            "can cause tiny downstream metric drift across scipy versions."
        ),
    )
    return parser.parse_args()


def huber_loss(residual: np.ndarray, delta: float) -> np.ndarray:
    abs_residual = np.abs(residual)
    return np.where(
        abs_residual < delta,
        0.5 * residual**2,
        delta * abs_residual - 0.5 * delta**2,
    )


def momentum_predict(params: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    l0, a, c, alpha = params
    return l0 + a * np.power(np.maximum(s1, 1e-12), -alpha) - c * s2


def fit_momentum_law(
    s1: np.ndarray,
    s2: np.ndarray,
    loss: np.ndarray,
    huber_delta: float,
    maxiter: int,
) -> tuple[np.ndarray, float]:
    def objective(params: np.ndarray) -> float:
        pred = momentum_predict(params, s1, s2)
        if np.any(~np.isfinite(pred)) or np.any(pred <= 0):
            return 1e8
        residual = np.log(loss) - np.log(pred)
        return float(np.sum(huber_loss(residual, huber_delta)))

    starts = [
        (l0, a, c, alpha)
        for l0 in np.linspace(0.5, 2.8, 3)
        for a in np.linspace(0.5, 6.0, 3)
        for c in [0.01, 0.1, 0.5, 1.0]
        for alpha in [0.3, 0.6, 0.9, 1.2]
    ]
    best_params: np.ndarray | None = None
    best_value = math.inf
    for start in starts:
        result = minimize(
            objective,
            np.array(start, dtype=np.float64),
            method="L-BFGS-B",
            bounds=[(0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)],
            options={"maxiter": maxiter, "ftol": 1e-9, "gtol": 1e-6, "eps": 1e-8},
        )
        if np.isfinite(result.fun) and float(result.fun) < best_value:
            best_value = float(result.fun)
            best_params = result.x.astype(np.float64)

    if best_params is None:
        raise RuntimeError("Momentum-law fit failed")
    return best_params, best_value


def main() -> None:
    args = parse_args()
    data_path = resolve_input_path(args.data_path)
    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = load_curves(data_path, SCHEDULES, args.decay_factor)
    train_curve = curves[args.train_run]
    train_idx = np.where(train_curve.step >= args.start_step)[0]
    if train_idx.size == 0:
        raise ValueError(f"No training rows at or after step {args.start_step}")
    if args.refit:
        params, objective_value = fit_momentum_law(
            train_curve.s1[train_idx],
            train_curve.s2[train_idx],
            train_curve.loss[train_idx],
            huber_delta=args.huber_delta,
            maxiter=args.maxiter,
        )
        fit_source = "refit"
    else:
        params = REPORT_MOMENTUM_PARAMS.copy()
        objective_value = REPORT_MOMENTUM_OBJECTIVE
        fit_source = "report_intermediate_parameters"

    frames = []
    for schedule in SCHEDULES:
        curve = curves[schedule]
        frames.append(
            pd.DataFrame(
                {
                    "run": curve.alias,
                    "step": curve.step,
                    "loss": curve.loss,
                    "lr": curve.lr,
                    "s1": curve.s1,
                    "s2": curve.s2,
                    "momentum_s2": np.maximum(momentum_predict(params, curve.s1, curve.s2), 1e-8),
                }
            )
        )

    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    summary = {
        "purpose": (
            "Full-grid momentum-law predictions for residual spline and "
            "event/tail leftover experiments."
        ),
        "contains_mlp_predictions": False,
        "schedules": list(SCHEDULES),
        "rows": int(len(predictions)),
        "columns": list(predictions.columns),
        "data_path": display_path(data_path),
        "output_dir": display_path(output_dir),
        "fit_run": args.train_run,
        "fit_source": fit_source,
        "fit_start_step": args.start_step,
        "fit_points": int(train_idx.size),
        "objective_value": float(objective_value),
        "decay_factor": args.decay_factor,
        "huber_delta": args.huber_delta,
        "params": {
            "L0": float(params[0]),
            "A": float(params[1]),
            "C": float(params[2]),
            "alpha": float(params[3]),
        },
        "fit_initialization": {
            "L0": "np.linspace(0.5, 2.8, 3)",
            "A": "np.linspace(0.5, 6.0, 3)",
            "C": "[0.01, 0.1, 0.5, 1.0]",
            "alpha": "[0.3, 0.6, 0.9, 1.2]",
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(predictions)} rows to {display_path(output_dir / 'predictions.csv')}")


if __name__ == "__main__":
    main()
