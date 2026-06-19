# src/evaluation.py
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import mean_squared_error, r2_score
from .utils import huber_loss

def evaluate_mpl(data: dict, curve_set: list, best_params: list, fig_folder: str):
    """
    Evaluate MPL model on a given dataset and plot results.

    Args:
        data (dict): Dataset with steps, lrs, and losses.
        curve_set (list): List of file names to evaluate.
        best_params (list): Fitted MPL parameters [L0, A, alpha, B, C, beta, gamma].
        fig_folder (str): Directory to save figures.
    """
    logger = logging.getLogger(__name__)
    L0, A, alpha, B, C, beta, gamma = best_params

    # Preprocess data
    for file_name in curve_set:
        lrs = data[file_name]["lrs"]
        step = data[file_name]["step"]
        lr_sum = np.cumsum(lrs)
        lr_gap = np.zeros(len(lrs))
        lr_gap[1:] = np.diff(lrs)
        data[file_name]["S1"] = lr_sum[step]
        data[file_name]["lr_sum"] = lr_sum
        data[file_name]["lr_gap"] = lr_gap

    # Compute predictions and metrics
    for file_name in curve_set:
        step = data[file_name]["step"]
        lr_gap = data[file_name]["lr_gap"]
        S1 = data[file_name]["S1"]
        lrs = data[file_name]["lrs"]
        lr_sum = data[file_name]["lr_sum"]
        loss = data[file_name]["loss"]
        LD = np.zeros(len(step))
        for i, s in enumerate(step):
            LD[i] = np.sum(lr_gap[1:s+1] * (1 - (1 + C * lrs[1:s+1] ** (-gamma) * (lr_sum[s] - lr_sum[:s])) ** (-beta)))
        pred = L0 + A * S1 ** (-alpha) + B * LD
        r = np.log(loss) - np.log(pred)

        data[file_name]["pred"] = pred
        data[file_name]["huber_loss"] = huber_loss(r).sum()
        data[file_name]["mse_loss"] = mean_squared_error(loss, pred)
        data[file_name]["rmse_loss"] = np.sqrt(data[file_name]["mse_loss"])
        data[file_name]["mae_loss"] = np.mean(np.abs(loss - pred))
        data[file_name]["prede"] = np.mean(np.abs(loss - pred) / loss)
        data[file_name]["worste"] = np.max(np.abs(loss - pred) / loss)
        data[file_name]["r2_score"] = r2_score(loss, pred)

        plt.figure()
        file_id = file_name.split(".")[0]
        plt.plot(step, pred, label=f"{file_id}_pred", linestyle="--")
        plt.plot(step, loss, label=f"{file_id}_loss", linestyle="-")
        plt.legend()
        plt.savefig(f"{fig_folder}/{file_id}_mplfit.png")
        plt.close()

        logger.info(file_name)
        for key in ["huber_loss", "mse_loss", "rmse_loss", "mae_loss", "prede", "worste", "r2_score"]:
            logger.info(f"{key}: {data[file_name][key]}")

    # Print average metrics
    metrics = ["huber_loss", "mse_loss", "rmse_loss", "mae_loss", "prede", "worste", "r2_score"]
    for metric in metrics:
        avg = np.mean([data[file_name][metric] for file_name in curve_set])
        logger.info(f"Average {metric.capitalize()}: {avg}")
    logger.info("-" * 50)