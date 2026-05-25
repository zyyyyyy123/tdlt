# src/optimization.py
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
from torch import nn
from tqdm import tqdm
from .config import OPT_PATH, OPT_TOTAL_STEPS, OPT_PEAK_LR, OPT_MIN_LR, OPT_LR, OPT_MAX_STEPS, OPT_WARMUP, OPT_INTERVAL

def optimize_lr_schedule(
    best_params,
    total_steps=OPT_TOTAL_STEPS,
    peak_lr=OPT_PEAK_LR,
    min_lr=OPT_MIN_LR,
    lr=OPT_LR,
    max_steps=OPT_MAX_STEPS,
    warmup=OPT_WARMUP,
    interval=OPT_INTERVAL,
    name="400M"
):
    """
    Optimize the learning rate schedule using the fitted MPL model.

    Args:
        best_params (list): Fitted MPL parameters [L0, A, alpha, B, C, beta, gamma].
        total_steps (int): Total steps in the schedule.
        peak_lr (float): Initial peak learning rate.
        min_lr (float): Minimum learning rate threshold.
        lr (float): Learning rate for optimization.
        max_steps (int): Maximum optimization steps.
        warmup (int): Number of warmup steps.
        interval (int): Logging interval.
        name (str): Identifier for output files.
    
    Returns:
        np.ndarray: Optimized learning rate schedule.
    """
    logger = logging.getLogger(__name__)
    L0, A, alpha, B, C, beta, gamma = best_params

    # Initialize Delta (learnable LR reductions)
    delta = nn.Parameter(torch.zeros(total_steps - warmup, dtype=torch.float64), requires_grad=True)
    warmup_bias = 0.5 * peak_lr * warmup
    optimizer = torch.optim.Adam([delta], lr=lr)

    # Optimization loop
    for opt_step in tqdm(range(max_steps), desc="Optimizing LR Schedule"):
        optimizer.zero_grad()

        # Compute LR schedule from Delta
        eta = peak_lr - torch.cumsum(delta.clamp(min=0), dim=0)
        eta = torch.clamp(eta, min=min_lr)

        lr_sum = torch.cumsum(eta, dim=0) + warmup_bias
        lr_sum = torch.concatenate([torch.tensor([0]), lr_sum], dim=0)
        LD = torch.sum(delta * (1 - (1 + C * eta ** (-gamma) * (lr_sum[-1] - lr_sum[:-1])) ** (-beta)))
        pred = L0 + A * lr_sum[-1] ** (-alpha) - B * LD
        pred.backward()
        optimizer.step()

        # Enforce constraints
        with torch.no_grad():
            delta.clamp_(min=0, max=peak_lr)
            eta = peak_lr - torch.cumsum(delta, dim=0)
            delta.masked_fill_(eta <= min_lr, 0)
            opt_lr = eta.detach().numpy()
            loss = pred.item()

        if opt_step % interval == 0 or opt_step == max_steps - 1:
            logger.info(f"Iteration {opt_step}, Loss: {loss}")
            logger.info(f"First 5 LRs: {opt_lr[:5]}, Last 5 LRs: {opt_lr[-5:]}")
            grad_norm = torch.norm(delta.grad).item() if delta.grad is not None else 0.0
            logger.info(f"Last 5-step gradients: {delta.grad[-5:]}")
            logger.info(f"Gradient norm: {grad_norm}")

    logger.info(f"Final Loss: {loss}")
    os.makedirs(OPT_PATH, exist_ok=True)
    np.save(os.path.join(OPT_PATH, f"{name}.npy"), opt_lr)
    plt.figure()
    plt.plot(np.arange(warmup, total_steps), opt_lr)
    plt.grid(True)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Optimized Learning Rate Schedule ({name})")
    plt.savefig(os.path.join(OPT_PATH, f"{name}.png"))
    plt.close()
    return opt_lr