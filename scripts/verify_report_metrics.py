from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify headline metrics used in the final report.")
    parser.add_argument("--atol", type=float, default=5e-5)
    parser.add_argument("--rtol", type=float, default=5e-5)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required result file is missing: {path.relative_to(REPO_ROOT)}")
    return path


def check_close(label: str, actual: float, expected: float, atol: float, rtol: float) -> None:
    if not math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol):
        raise AssertionError(
            f"{label}: expected {expected:.12g}, got {float(actual):.12g}"
        )
    print(f"OK  {label:<48} {float(actual):.6f}")


def momentum_metric(run: str, key: str) -> float:
    path = require_file(REPO_ROOT / "results" / "baselines" / "momentum" / "summary.json")
    summary = json.loads(path.read_text())
    for fit in summary["fits"]:
        for row in fit["metrics"]:
            if row["run"] == run:
                return float(row[key])
    raise KeyError(f"Momentum metric not found for run={run!r}, key={key!r}")


def csv_value(path: Path, filters: dict[str, object], key: str) -> float:
    frame = pd.read_csv(require_file(path))
    mask = pd.Series(True, index=frame.index)
    for column, value in filters.items():
        mask &= frame[column].astype(str) == str(value)
    matched = frame.loc[mask]
    if len(matched) != 1:
        raise KeyError(
            f"Expected one row in {path.relative_to(REPO_ROOT)} for {filters}, got {len(matched)}"
        )
    return float(matched.iloc[0][key])


def main() -> None:
    args = parse_args()
    atol = args.atol
    rtol = args.rtol

    check_close("momentum/cosine MAE", momentum_metric("cosine", "mae"), 0.03222419146683912, atol, rtol)
    check_close("momentum/cosine R2", momentum_metric("cosine", "r2"), 0.953200819962247, atol, rtol)
    check_close("momentum/wsd MAE", momentum_metric("wsd", "mae"), 0.03784425393486444, atol, rtol)
    check_close("momentum/wsd R2", momentum_metric("wsd", "r2"), 0.9256687965647892, atol, rtol)

    mpl_metrics = REPO_ROOT / "results" / "baselines" / "multi_power_law" / "metrics.csv"
    check_close("MPL/wsd MAE", csv_value(mpl_metrics, {"run": "wsd"}, "mae"), 0.03602715935195318, atol, rtol)
    check_close("MPL/wsd R2", csv_value(mpl_metrics, {"run": "wsd"}, "r2"), 0.9356762525687805, atol, rtol)
    check_close(
        "MPL/wsd endpoint abs",
        csv_value(mpl_metrics, {"run": "wsd"}, "endpoint_abs_diff"),
        0.13409399735175054,
        atol,
        rtol,
    )

    spline_metrics = REPO_ROOT / "experiments" / "residual_methods" / "outputs" / "spline_stability_selected_metrics.csv"
    spline_filter = {
        "model": "step_spline_811_selected",
        "target_schedule": "wsd",
        "window": "full",
    }
    check_close("selected step spline/wsd MAE", csv_value(spline_metrics, spline_filter, "mae"), 0.020877761799257585, atol, rtol)
    check_close("selected step spline/wsd R2", csv_value(spline_metrics, spline_filter, "r2"), 0.9789902948605604, atol, rtol)

    selected = REPO_ROOT / "experiments" / "residual_methods" / "outputs" / "sujianlin_event_decay_selected_metrics_by_window.csv"
    final_filter = {
        "model_class": "step_plus_event_leftover",
        "selection_rule": "811_full",
        "split": "test",
        "schedule": "wsd",
        "window": "full",
    }
    check_close("final step+event/wsd full MAE", csv_value(selected, final_filter, "mae"), 0.011191, atol, rtol)
    check_close("final step+event/wsd full RMSE", csv_value(selected, final_filter, "rmse"), 0.014137, atol, rtol)
    check_close("final step+event/wsd full R2", csv_value(selected, final_filter, "r2"), 0.993410, atol, rtol)
    check_close(
        "final step+event/wsd endpoint abs",
        csv_value(selected, final_filter, "endpoint_abs_diff"),
        0.016615,
        atol,
        rtol,
    )

    tail_filter = dict(final_filter)
    tail_filter["window"] = "tail_27126_33906"
    check_close("final step+event/wsd tail MAE", csv_value(selected, tail_filter, "mae"), 0.015937, atol, rtol)

    print("\nAll report headline metrics match expected values.")


if __name__ == "__main__":
    main()
