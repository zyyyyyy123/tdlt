from pathlib import Path
import os
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = PROJECT_DIR / "outputs" / ".matplotlib"
XDG_CACHE_DIR = PROJECT_DIR / "outputs" / ".cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.baselines import BaselineFitConfig, fit_and_predict_baselines
from src.data import load_loss_curves
from src.evaluation import evaluate_predictions
from src.plotting import plot_residual_predictions
from src.residual_model import ResidualCorrectionConfig, build_residual_predictions


BASELINE_DIR = PROJECT_DIR / "baseline_results"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = PROJECT_DIR / "figures"


def main() -> None:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    curves = load_loss_curves()

    baseline_predictions = fit_and_predict_baselines(
        curves,
        BASELINE_DIR,
        BaselineFitConfig(train_schedule="cosine", fit_stride=50),
    )

    residual_predictions = build_residual_predictions(
        baseline_predictions,
        ResidualCorrectionConfig(train_schedule="cosine"),
    )
    residual_metrics = evaluate_predictions(residual_predictions)

    residual_predictions.to_csv(OUTPUT_DIR / "residual_predictions.csv", index=False)
    residual_metrics.to_csv(OUTPUT_DIR / "residual_metrics.csv", index=False)

    plot_residual_predictions(
        residual_predictions,
        FIGURE_DIR / "residual_predictions_full.png",
        method="mpl",
        feature_set="drop_momentum",
    )
    plot_residual_predictions(
        residual_predictions,
        FIGURE_DIR / "residual_predictions_20000_30000.png",
        method="mpl",
        feature_set="drop_momentum",
        start_step=20000,
        end_step=30000,
    )

    report = residual_metrics[
        (residual_metrics["schedule"].isin(["wsd", "811"]))
        & (residual_metrics["window"].isin(["full", "steps_20000_30000"]))
    ].sort_values(["method", "schedule", "window", "correction", "feature_set"])
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
