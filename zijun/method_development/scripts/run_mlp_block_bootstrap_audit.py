from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
CODE_DIR = REPO_ROOT / "code"

MPL_CACHE_DIR = PROJECT_DIR / "outputs" / ".matplotlib"
XDG_CACHE_DIR = PROJECT_DIR / "outputs" / ".cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from momentum_residual_mlp import (  # noqa: E402
    RUN_ALIASES,
    build_feature_cache,
    fit_momentum_law,
    load_curves,
    metric_dict,
    momentum_predict,
    parse_float_list,
    parse_hidden,
    parse_int_list,
    residual_predictions,
    train_residual_mlp,
)


OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"
DEFAULT_DATA_PATH = REPO_ROOT / "loss curves" / "gpt_loss+lrs.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Block-bootstrap stability audit for the momentum residual MLP."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--output-prefix", default="mlp_block_bootstrap")
    parser.add_argument("--train-run", default="cosine", choices=sorted(RUN_ALIASES))
    parser.add_argument("--validation-run", default="811", choices=sorted(RUN_ALIASES))
    parser.add_argument("--test-run", default="wsd", choices=sorted(RUN_ALIASES))
    parser.add_argument("--start-step", type=int, default=1000)
    parser.add_argument("--momentum-decay", type=float, default=0.999)
    parser.add_argument("--feature-set", default="kernel_summary", choices=["kernel_summary"])
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--target", default="log_residual", choices=["log_residual"])
    parser.add_argument("--hidden", default="64,32")
    parser.add_argument("--activation", default="relu", choices=["identity", "logistic", "tanh", "relu"])
    parser.add_argument("--alpha", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--scaler", default="physics", choices=["standard", "physics"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--huber-delta", type=float, default=1e-3)
    parser.add_argument("--block-sizes", default="128,512,2048")
    parser.add_argument("--n-bootstrap", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=20260608)
    parser.add_argument("--mlp-seed-base", type=int, default=308100)
    parser.add_argument("--shrink-grid", default="0,0.75,1.0,1.25")
    parser.add_argument("--fixed-shrink", type=float, default=1.0)
    parser.add_argument("--acf-max-lag", type=int, default=8192)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def draw_block_bootstrap_indices(
    base_indices: np.ndarray,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if len(base_indices) == 0:
        raise ValueError("Cannot bootstrap an empty training index set")

    block_size = min(block_size, len(base_indices))
    first = int(base_indices[0])
    last = int(base_indices[-1])
    last_start = last - block_size + 1
    n_blocks = int(math.ceil(len(base_indices) / block_size))
    starts = rng.integers(first, last_start + 1, size=n_blocks)
    pieces = [np.arange(start, start + block_size, dtype=np.int64) for start in starts]
    sampled = np.concatenate(pieces)[: len(base_indices)]
    return sampled.astype(np.int64, copy=False)


def residual_target_values(curve, indices: np.ndarray, baseline: np.ndarray, target: str) -> np.ndarray:
    loss = curve.loss[indices]
    pred = baseline[indices]
    if target == "log_residual":
        return np.log(np.maximum(loss, 1e-12)) - np.log(np.maximum(pred, 1e-12))
    raise ValueError(f"Unsupported target: {target}")


def autocorrelation_fft(values: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.array([1.0], dtype=np.float64)
    x = x - float(np.mean(x))
    variance = float(np.var(x))
    if variance <= 0.0:
        return np.ones(min(max_lag, len(x) - 1) + 1, dtype=np.float64)

    max_lag = min(int(max_lag), len(x) - 1)
    n_fft = 1 << (2 * len(x) - 1).bit_length()
    spectrum = np.fft.rfft(x, n=n_fft)
    autocov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n_fft)[: max_lag + 1]
    counts = np.arange(len(x), len(x) - max_lag - 1, -1, dtype=np.float64)
    autocov = autocov / counts
    return autocov / autocov[0]


def build_effective_sample_frame(
    residual_values: np.ndarray,
    max_lag: int,
) -> pd.DataFrame:
    acf = autocorrelation_fft(residual_values, max_lag=max_lag)
    positive = acf[1:]
    non_positive = np.flatnonzero(positive <= 0.0)
    if len(non_positive):
        cutoff_lag = int(non_positive[0] + 1)
    else:
        cutoff_lag = len(acf) - 1

    tau_int = 1.0 + 2.0 * float(np.sum(acf[1 : cutoff_lag + 1]))
    tau_int = max(tau_int, 1.0)
    n = int(len(residual_values))
    ess = float(n / tau_int)
    diagnostic_lags = sorted(
        {
            0,
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            min(len(acf) - 1, max_lag),
        }
    )
    rows = []
    for lag in diagnostic_lags:
        if lag < len(acf):
            rows.append(
                {
                    "series": "cosine_log_residual",
                    "n": n,
                    "max_lag": int(len(acf) - 1),
                    "cutoff_lag_initial_positive": cutoff_lag,
                    "tau_int_initial_positive": tau_int,
                    "ess_initial_positive": ess,
                    "ess_fraction": ess / n,
                    "lag": int(lag),
                    "acf": float(acf[lag]),
                }
            )
    return pd.DataFrame(rows)


def metrics_for_prediction(curves, alias: str, indices: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    metrics = metric_dict(curves[alias].loss[indices], prediction[indices])
    return {
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "endpoint_abs_diff": metrics["endpoint_abs_diff"],
        "max_abs_error": metrics["max_abs_error"],
    }


def evaluate_shrink(
    curves,
    indices: np.ndarray,
    validation_run: str,
    test_run: str,
    predictions: dict[str, np.ndarray],
    baseline_validation_mae: float,
    baseline_test_mae: float,
) -> dict[str, float]:
    validation = metrics_for_prediction(curves, validation_run, indices, predictions[validation_run])
    test = metrics_for_prediction(curves, test_run, indices, predictions[test_run])
    return {
        "validation_mae": validation["mae"],
        "validation_rmse": validation["rmse"],
        "validation_r2": validation["r2"],
        "validation_endpoint_abs_diff": validation["endpoint_abs_diff"],
        "test_mae": test["mae"],
        "test_rmse": test["rmse"],
        "test_r2": test["r2"],
        "test_endpoint_abs_diff": test["endpoint_abs_diff"],
        "test_max_abs_error": test["max_abs_error"],
        "validation_improvement_pct": 100.0
        * (baseline_validation_mae - validation["mae"])
        / baseline_validation_mae,
        "test_improvement_pct": 100.0 * (baseline_test_mae - test["mae"]) / baseline_test_mae,
    }


def summarize_trials(trials: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "train_unique_fraction",
        "selected_shrink",
        "validation_mae",
        "validation_r2",
        "validation_endpoint_abs_diff",
        "validation_improvement_pct",
        "test_mae",
        "test_r2",
        "test_endpoint_abs_diff",
        "test_improvement_pct",
        "n_iter",
        "loss_curve_final",
    ]
    rows = []
    grouped = trials.groupby(["block_size", "selection_mode"], sort=True)
    for (block_size, selection_mode), group in grouped:
        row = {
            "block_size": int(block_size),
            "selection_mode": selection_mode,
            "n_replicates": int(len(group)),
        }
        for metric in metrics:
            values = group[metric].astype(float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_q05"] = float(values.quantile(0.05))
            row[f"{metric}_q50"] = float(values.quantile(0.50))
            row[f"{metric}_q95"] = float(values.quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def shrink_label(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def plot_mae_distribution(trials: pd.DataFrame, output_path: Path) -> None:
    selected = trials[trials["selection_mode"].eq("validation_selected")].copy()
    if selected.empty:
        return

    block_sizes = sorted(selected["block_size"].unique())
    positions = np.arange(len(block_sizes), dtype=np.float64)
    validation_data = [
        selected[selected["block_size"].eq(block_size)]["validation_mae"].to_numpy()
        for block_size in block_sizes
    ]
    test_data = [
        selected[selected["block_size"].eq(block_size)]["test_mae"].to_numpy()
        for block_size in block_sizes
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(
        validation_data,
        positions=positions - 0.18,
        widths=0.28,
        patch_artist=True,
        boxprops={"facecolor": "#56B4E9", "alpha": 0.65},
        medianprops={"color": "#111111"},
    )
    ax.boxplot(
        test_data,
        positions=positions + 0.18,
        widths=0.28,
        patch_artist=True,
        boxprops={"facecolor": "#E69F00", "alpha": 0.65},
        medianprops={"color": "#111111"},
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([str(v) for v in block_sizes])
    ax.set_xlabel("Block size")
    ax.set_ylabel("MAE")
    ax.set_title("Block-bootstrap MLP residual MAE, validation-selected shrink")
    ax.grid(axis="y", alpha=0.25)
    ax.plot([], [], color="#56B4E9", linewidth=8, alpha=0.65, label="811 validation")
    ax.plot([], [], color="#E69F00", linewidth=8, alpha=0.65, label="WSD test")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_selected_shrink(trials: pd.DataFrame, output_path: Path) -> None:
    selected = trials[trials["selection_mode"].eq("validation_selected")].copy()
    if selected.empty:
        return

    counts = (
        selected.groupby(["block_size", "selected_shrink"], sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    block_sizes = sorted(counts["block_size"].unique())
    shrink_values = sorted(counts["selected_shrink"].unique())
    x = np.arange(len(block_sizes), dtype=np.float64)
    width = 0.8 / max(len(shrink_values), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, shrink in enumerate(shrink_values):
        subset = counts[counts["selected_shrink"].eq(shrink)]
        values = []
        for block_size in block_sizes:
            match = subset[subset["block_size"].eq(block_size)]
            values.append(int(match["count"].iloc[0]) if not match.empty else 0)
        offset = (i - (len(shrink_values) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=f"shrink={shrink:g}")

    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in block_sizes])
    ax.set_xlabel("Block size")
    ax.set_ylabel("Selected count")
    ax.set_title("Validation-selected shrink distribution")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    data_path = resolve_path(args.data_path)
    output_dir = resolve_path(args.output_dir)
    figure_dir = resolve_path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_figures:
        figure_dir.mkdir(parents=True, exist_ok=True)

    hidden = parse_hidden(args.hidden)
    block_sizes = parse_int_list(args.block_sizes)
    shrink_grid = parse_float_list(args.shrink_grid)
    if args.fixed_shrink not in shrink_grid:
        shrink_grid = sorted({*shrink_grid, args.fixed_shrink})
    if args.n_bootstrap <= 0:
        raise ValueError("--n-bootstrap must be positive")

    aliases = sorted({args.train_run, args.validation_run, args.test_run})
    curves = load_curves(data_path, aliases, args.momentum_decay)
    end = min(len(curve.step) for curve in curves.values())
    base_indices = np.arange(args.start_step, end, dtype=np.int64)
    if len(base_indices) == 0:
        raise ValueError("No indices after --start-step")

    full_train_indices = {args.train_run: base_indices}
    momentum_params, momentum_objective = fit_momentum_law(
        curves,
        full_train_indices,
        args.huber_delta,
    )
    momentum_predictions = {
        alias: np.maximum(momentum_predict(momentum_params, curve.s1, curve.s2), 1e-8)
        for alias, curve in curves.items()
    }
    feature_cache, feature_names = build_feature_cache(
        curves,
        momentum_predictions,
        args.feature_set,
        args.m,
    )

    baseline_validation = metrics_for_prediction(
        curves,
        args.validation_run,
        base_indices,
        momentum_predictions[args.validation_run],
    )
    baseline_test = metrics_for_prediction(
        curves,
        args.test_run,
        base_indices,
        momentum_predictions[args.test_run],
    )

    train_residual = residual_target_values(
        curves[args.train_run],
        base_indices,
        momentum_predictions[args.train_run],
        args.target,
    )
    effective_sample = build_effective_sample_frame(train_residual, args.acf_max_lag)

    rng = np.random.default_rng(args.random_seed)
    trial_rows = []

    for block_size in block_sizes:
        for replicate in range(args.n_bootstrap):
            sampled_indices = draw_block_bootstrap_indices(base_indices, block_size, rng)
            unique_count = int(np.unique(sampled_indices).size)
            mlp_seed = int(args.mlp_seed_base + 10000 * block_size + replicate)
            trained = train_residual_mlp(
                curves=curves,
                feature_cache=feature_cache,
                train_indices={args.train_run: sampled_indices},
                momentum_predictions=momentum_predictions,
                target=args.target,
                hidden=hidden,
                activation=args.activation,
                alpha=args.alpha,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                max_iter=args.max_iter,
                seed=mlp_seed,
                scaler_name=args.scaler,
            )

            candidate_rows = []
            for shrink in shrink_grid:
                predictions = residual_predictions(
                    trained,
                    momentum_predictions,
                    shrink,
                    args.target,
                )
                metrics = evaluate_shrink(
                    curves,
                    base_indices,
                    args.validation_run,
                    args.test_run,
                    predictions,
                    baseline_validation["mae"],
                    baseline_test["mae"],
                )
                candidate_rows.append(
                    {
                        "shrink": float(shrink),
                        "metrics": metrics,
                    }
                )

            fixed = min(candidate_rows, key=lambda row: abs(row["shrink"] - args.fixed_shrink))
            selected = min(
                candidate_rows,
                key=lambda row: (row["metrics"]["validation_mae"], row["shrink"]),
            )

            base_row = {
                "block_size": int(block_size),
                "replicate": int(replicate),
                "mlp_seed": mlp_seed,
                "n_train": int(len(base_indices)),
                "n_blocks": int(math.ceil(len(base_indices) / min(block_size, len(base_indices)))),
                "train_unique_count": unique_count,
                "train_unique_fraction": unique_count / float(len(base_indices)),
                "target": args.target,
                "feature_set": args.feature_set,
                "feature_count": int(len(feature_names)),
                "m": int(args.m),
                "hidden": "-".join(str(v) for v in hidden),
                "activation": args.activation,
                "scaler": args.scaler,
                "alpha": float(args.alpha),
                "learning_rate": float(args.learning_rate),
                "batch_size": int(args.batch_size),
                "max_iter": int(args.max_iter),
                "n_iter": int(trained.model.n_iter_),
                "loss_curve_final": float(trained.model.loss_),
                "baseline_validation_mae": baseline_validation["mae"],
                "baseline_validation_r2": baseline_validation["r2"],
                "baseline_test_mae": baseline_test["mae"],
                "baseline_test_r2": baseline_test["r2"],
                "momentum_objective": float(momentum_objective),
                "momentum_L0": float(momentum_params[0]),
                "momentum_A": float(momentum_params[1]),
                "momentum_C": float(momentum_params[2]),
                "momentum_alpha": float(momentum_params[3]),
            }

            for mode, chosen in [
                (f"fixed_shrink_{shrink_label(args.fixed_shrink)}", fixed),
                ("validation_selected", selected),
            ]:
                row = dict(base_row)
                row.update(
                    {
                        "selection_mode": mode,
                        "selected_shrink": float(chosen["shrink"]),
                        "candidate_validation_mae_by_shrink": json.dumps(
                            {
                                str(item["shrink"]): item["metrics"]["validation_mae"]
                                for item in candidate_rows
                            },
                            sort_keys=True,
                        ),
                        "candidate_test_mae_by_shrink": json.dumps(
                            {
                                str(item["shrink"]): item["metrics"]["test_mae"]
                                for item in candidate_rows
                            },
                            sort_keys=True,
                        ),
                    }
                )
                row.update(chosen["metrics"])
                trial_rows.append(row)

            print(
                "block_size={block_size} replicate={replicate} "
                "selected_shrink={selected_shrink:g} val_mae={val_mae:.6f} "
                "test_mae={test_mae:.6f}".format(
                    block_size=block_size,
                    replicate=replicate,
                    selected_shrink=selected["shrink"],
                    val_mae=selected["metrics"]["validation_mae"],
                    test_mae=selected["metrics"]["test_mae"],
                )
            )

    trials = pd.DataFrame(trial_rows)
    summary = summarize_trials(trials)

    trials_path = output_dir / f"{args.output_prefix}_trials.csv"
    summary_path = output_dir / f"{args.output_prefix}_summary.csv"
    effective_path = output_dir / f"{args.output_prefix}_effective_sample.csv"
    trials.to_csv(trials_path, index=False)
    summary.to_csv(summary_path, index=False)
    effective_sample.to_csv(effective_path, index=False)

    if not args.skip_figures:
        plot_mae_distribution(
            trials,
            figure_dir / f"{args.output_prefix}_mae_distribution.png",
        )
        plot_selected_shrink(
            trials,
            figure_dir / f"{args.output_prefix}_selected_shrink.png",
        )

    print("\nBaseline:")
    print(
        f"  validation {args.validation_run}: MAE={baseline_validation['mae']:.8f}, "
        f"R2={baseline_validation['r2']:.8f}"
    )
    print(
        f"  test {args.test_run}: MAE={baseline_test['mae']:.8f}, "
        f"R2={baseline_test['r2']:.8f}"
    )
    print("\nBootstrap summary:")
    cols = [
        "block_size",
        "selection_mode",
        "n_replicates",
        "test_mae_mean",
        "test_mae_q05",
        "test_mae_q50",
        "test_mae_q95",
        "test_improvement_pct_mean",
    ]
    print(summary[cols].to_string(index=False))
    print("\nSaved:")
    print(f"  {display_path(trials_path)}")
    print(f"  {display_path(summary_path)}")
    print(f"  {display_path(effective_path)}")


if __name__ == "__main__":
    main()
