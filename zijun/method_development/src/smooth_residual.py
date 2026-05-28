from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline


@dataclass
class SplineResidualConfig:
    train_schedule: str = "cosine"
    smoothing: float = 0.1
    shrink: float = 1.0
    residual_clip: float = 0.25


class SplineResidualCorrector:
    def __init__(self, config: SplineResidualConfig | None = None):
        self.config = config or SplineResidualConfig()
        self.spline_: UnivariateSpline | None = None

    def fit(self, predictions: pd.DataFrame) -> "SplineResidualCorrector":
        train = predictions[predictions["schedule"] == self.config.train_schedule].sort_values("step")
        residual = np.log(train["loss"]) - np.log(np.maximum(train["base_pred_loss"], 1e-12))
        self.spline_ = UnivariateSpline(train["step"], residual, s=self.config.smoothing)
        return self

    def predict_residual(self, predictions: pd.DataFrame) -> np.ndarray:
        if self.spline_ is None:
            raise RuntimeError("SplineResidualCorrector must be fitted before prediction.")

        residual = self.spline_(predictions["step"].to_numpy(dtype=float))
        residual = residual * self.config.shrink
        return np.clip(residual, -self.config.residual_clip, self.config.residual_clip)


def build_none_prediction(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["correction"] = "none"
    frame["feature_set"] = "none"
    frame["residual_pred"] = 0.0
    frame["pred_loss"] = frame["base_pred_loss"]
    return frame


def build_mean_shift_prediction(
    predictions: pd.DataFrame,
    train_schedule: str = "cosine",
) -> pd.DataFrame:
    train = predictions[predictions["schedule"] == train_schedule]
    residual = np.log(train["loss"]) - np.log(np.maximum(train["base_pred_loss"], 1e-12))

    frame = predictions.copy()
    frame["correction"] = "mean_shift"
    frame["feature_set"] = "constant_residual"
    frame["residual_pred"] = float(residual.mean())
    frame["pred_loss"] = frame["base_pred_loss"] * np.exp(frame["residual_pred"])
    return frame


def build_spline_prediction(
    predictions: pd.DataFrame,
    config: SplineResidualConfig,
) -> pd.DataFrame:
    corrector = SplineResidualCorrector(config).fit(predictions)
    frame = predictions.copy()
    frame["correction"] = "smooth_residual"
    frame["feature_set"] = f"spline_s{config.smoothing:g}_shrink{config.shrink:g}"
    frame["residual_pred"] = corrector.predict_residual(frame)
    frame["pred_loss"] = frame["base_pred_loss"] * np.exp(frame["residual_pred"])
    return frame
