from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, add_schedule_features


@dataclass
class FeatureRidgeConfig:
    alpha: float = 1e-3


class FeatureRidgeLossPredictor:
    def __init__(self, config: FeatureRidgeConfig | None = None):
        self.config = config or FeatureRidgeConfig()
        self.coef_: np.ndarray | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None

    def fit(self, train_curve: pd.DataFrame) -> "FeatureRidgeLossPredictor":
        train_features = add_schedule_features(train_curve)
        x_raw = train_features[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = np.log(train_features["loss"].to_numpy(dtype=float))

        self.feature_mean_ = x_raw.mean(axis=0)
        self.feature_scale_ = x_raw.std(axis=0)
        self.feature_scale_[self.feature_scale_ == 0.0] = 1.0

        x = (x_raw - self.feature_mean_) / self.feature_scale_
        x = np.column_stack([np.ones(len(x)), x])

        penalty = self.config.alpha * np.eye(x.shape[1])
        penalty[0, 0] = 0.0
        self.coef_ = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        return self

    def predict(self, curve: pd.DataFrame) -> np.ndarray:
        if self.coef_ is None or self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("The predictor must be fitted before calling predict().")

        features = add_schedule_features(curve)
        x_raw = features[FEATURE_COLUMNS].to_numpy(dtype=float)
        x = (x_raw - self.feature_mean_) / self.feature_scale_
        x = np.column_stack([np.ones(len(x)), x])
        return np.exp(x @ self.coef_)
