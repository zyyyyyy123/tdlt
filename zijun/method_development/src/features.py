import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "log_step",
    "log_cum_lr",
    "log_cum_lr2",
    "drop_mass",
    "drop_momentum_mass",
    "lr",
]


def add_schedule_features(curve: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    frame = curve.sort_values("step").copy()

    lr = frame["lr"].to_numpy(dtype=float)
    step = frame["step"].to_numpy(dtype=float)
    lr_drop = np.maximum(np.r_[0.0, lr[:-1] - lr[1:]], 0.0)

    drop_momentum = np.zeros_like(lr)
    for idx in range(1, len(lr)):
        drop_momentum[idx] = 0.999 * drop_momentum[idx - 1] + lr_drop[idx]

    frame["cum_lr"] = np.cumsum(lr)
    frame["cum_lr2"] = np.cumsum(lr * lr)
    frame["drop_mass"] = np.cumsum(lr_drop)
    frame["drop_momentum_mass"] = np.cumsum(drop_momentum)
    frame["log_step"] = np.log1p(step)
    frame["log_cum_lr"] = np.log(frame["cum_lr"] + eps)
    frame["log_cum_lr2"] = np.log(frame["cum_lr2"] + eps)

    return frame
