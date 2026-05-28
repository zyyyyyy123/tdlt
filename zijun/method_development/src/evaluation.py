from __future__ import annotations

import pandas as pd

from .metrics import regression_metrics


WINDOWS = {
    "full": (None, None),
    "steps_20000_30000": (20000, 30000),
}


def evaluate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["method", "correction", "feature_set", "schedule"]
    for group_values, group in predictions.groupby(group_cols, sort=False):
        for window_name, (start_step, end_step) in WINDOWS.items():
            window = group
            if start_step is not None:
                window = window[window["step"] >= start_step]
            if end_step is not None:
                window = window[window["step"] <= end_step]
            metrics = regression_metrics(window["loss"], window["pred_loss"])
            rows.append({
                **dict(zip(group_cols, group_values)),
                "window": window_name,
                "n": len(window),
                **metrics,
            })
    return pd.DataFrame(rows)
