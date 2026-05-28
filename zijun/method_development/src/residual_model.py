from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, add_schedule_features


FEATURE_GROUPS = {
    "step": ["log_step"],
    "lr_cumulative": ["log_step", "log_cum_lr", "log_cum_lr2", "lr"],
    "drop_momentum": FEATURE_COLUMNS,
}


@dataclass
class ResidualCorrectionConfig:
    train_schedule: str = "cosine"
    residual_clip: float = 0.7
    huber_epsilon: float = 1.35
    alpha: float = 1e-4
    max_iter: int = 1000


class ResidualCorrector:
    def __init__(
        self,
        feature_group: str,
        config: ResidualCorrectionConfig | None = None,
    ):
        if feature_group not in FEATURE_GROUPS:
            raise ValueError(f"Unknown feature group: {feature_group}")
        self.feature_group = feature_group
        self.feature_columns = FEATURE_GROUPS[feature_group]
        self.config = config or ResidualCorrectionConfig()
        self.model = make_pipeline(
            StandardScaler(),
            HuberRegressor(
                epsilon=self.config.huber_epsilon,
                alpha=self.config.alpha,
                max_iter=self.config.max_iter,
            ),
        )

    def fit(self, baseline_predictions: pd.DataFrame) -> "ResidualCorrector":
        train = self._add_features(baseline_predictions)
        train = train[train["schedule"] == self.config.train_schedule].copy()
        target = np.log(train["loss"]) - np.log(np.maximum(train["base_pred_loss"], 1e-12))
        self.model.fit(train[self.feature_columns], target)
        return self

    def predict_residual(self, baseline_predictions: pd.DataFrame) -> np.ndarray:
        frame = self._add_features(baseline_predictions)
        residual = self.model.predict(frame[self.feature_columns])
        return np.clip(
            residual,
            -self.config.residual_clip,
            self.config.residual_clip,
        )

    @staticmethod
    def _add_features(frame: pd.DataFrame) -> pd.DataFrame:
        pieces = []
        for _, group in frame.groupby(["method", "schedule"], sort=False):
            pieces.append(add_schedule_features(group))
        return pd.concat(pieces, ignore_index=True)


def build_residual_predictions(
    baseline_predictions: pd.DataFrame,
    config: ResidualCorrectionConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ResidualCorrectionConfig()
    baseline = baseline_predictions.rename(columns={"pred_loss": "base_pred_loss"}).copy()

    frames = []
    for method, method_frame in baseline.groupby("method", sort=False):
        no_correction = method_frame.copy()
        no_correction["correction"] = "none"
        no_correction["feature_set"] = "none"
        no_correction["residual_pred"] = 0.0
        no_correction["pred_loss"] = no_correction["base_pred_loss"]
        frames.append(no_correction)

        for feature_group in FEATURE_GROUPS:
            corrector = ResidualCorrector(feature_group, cfg).fit(method_frame)
            corrected = method_frame.copy()
            corrected["correction"] = "residual"
            corrected["feature_set"] = feature_group
            corrected["residual_pred"] = corrector.predict_residual(method_frame)
            corrected["pred_loss"] = corrected["base_pred_loss"] * np.exp(corrected["residual_pred"])
            frames.append(corrected)

    columns = [
        "method",
        "correction",
        "feature_set",
        "schedule",
        "step",
        "loss",
        "lr",
        "base_pred_loss",
        "residual_pred",
        "pred_loss",
    ]
    return pd.concat(frames, ignore_index=True)[columns]
