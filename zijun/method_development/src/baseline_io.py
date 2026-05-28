from pathlib import Path

import pandas as pd


def default_momentum_prediction_path() -> Path:
    return Path(__file__).resolve().parents[3] / "results" / "reproduction" / "momentum" / "predictions.csv"


def load_reproduced_momentum_predictions(
    path: str | Path | None = None,
    sampled_only: bool = True,
    min_step: int = 1000,
    max_step: int = 33906,
) -> pd.DataFrame:
    prediction_path = Path(path) if path is not None else default_momentum_prediction_path()
    frame = pd.read_csv(prediction_path)
    frame = frame.rename(columns={"run": "schedule", "prediction": "base_pred_loss"})
    frame["method"] = "momentum_reproduced"

    if sampled_only:
        frame = frame[frame["is_sampled"]].copy()
    frame = frame[(frame["step"] >= min_step) & (frame["step"] <= max_step)].copy()

    return frame[
        [
            "method",
            "schedule",
            "step",
            "lr",
            "loss",
            "base_pred_loss",
            "is_sampled",
            "s1",
            "s2",
        ]
    ].sort_values(["schedule", "step"]).reset_index(drop=True)
