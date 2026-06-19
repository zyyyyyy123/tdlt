from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def add_trailing_rolling_loss(
    frame: pd.DataFrame,
    *,
    window: int = 5,
    group_cols: Sequence[str] = ("schedule",),
    source_col: str = "loss",
    target_col: str = "loss_roll5",
    step_col: str = "step",
) -> pd.DataFrame:
    """Add a trailing rolling loss target and drop rows without a full window."""
    if window <= 0:
        raise ValueError("window must be positive")

    required = set(group_cols) | {source_col, step_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for rolling target: {sorted(missing)}")

    pieces = []
    for _, group in frame.groupby(list(group_cols), sort=False, dropna=False):
        group = group.sort_values(step_col).copy()
        group[target_col] = (
            group[source_col]
            .rolling(window=window, min_periods=window)
            .mean()
        )
        group = group[group[target_col].notna()].copy()
        pieces.append(group)

    if not pieces:
        result = frame.copy()
        result[target_col] = pd.Series(dtype=float)
        return result.iloc[0:0].reset_index(drop=True)

    sort_cols = list(group_cols) + [step_col]
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )
