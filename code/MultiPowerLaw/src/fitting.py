# src/fitting.py
import numpy as np
import torch
import logging
from scipy.optimize import minimize
from scipy.stats import linregress
from itertools import product
from tqdm import tqdm
from .models import MPL
from .utils import huber_loss, preprocess_data, compute_loss, compute_grad_norm, log_step, plot_loss_curve
from .config import FIT_EVAL_INTERVAL, FIT_LR1, FIT_LR2, FIT_MAX_STEPS, FIT_GRAD_NORM_THR, FIT_LOSS_THR, FIT_PATIENCE

def initialize_params(data: dict, train_set: list) -> list:
    """
    Initialize MPL parameters using grid search and L-BFGS-B optimization.

    Args:
        data (dict): Dataset containing steps, learning rates, and losses for each file.
        train_set (list): List of training file names.

    Returns:
        list: Initial parameter estimates [L0, A, alpha, B].
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting parameter initialization")

    min_loss = min(data[file_name]["loss"].min() for file_name in train_set)
    log_y_list, log_x_list = [], []

    for file_name in train_set:
        log_y = np.log(data[file_name]["loss"] - min_loss + 0.01)
        log_x = np.log(np.cumsum(data[file_name]["lrs"])[data[file_name]["step"]])
        log_y_list.append(log_y)
        log_x_list.append(log_x)

    log_y = np.concatenate(log_y_list)
    log_x = np.concatenate(log_x_list)
    slope, intercept, _, _, _ = linregress(log_x, log_y)

    L0_init_set = np.linspace(min_loss - 0.2, min_loss + 0.2, 5)
    A_init_set = np.linspace(np.exp(intercept) - 0.1, np.exp(intercept) + 0.1, 3)
    alpha_init_set = np.linspace(-slope - 0.1, -slope + 0.1, 3)
    B_init_set = np.linspace(100, 1000, 3)

    def loss_fn0(params):
        L0, A, alpha, B = params
        total_loss = 0
        for file_name in train_set:
            lr = data[file_name]["lrs"]
            step = data[file_name]["step"]
            pred = L0 + A * np.cumsum(lr)[step] ** (-alpha) - B * (3e-4 - lr[step])
            loss = data[file_name]["loss"]
            r = np.log(loss) - np.log(pred)
            total_loss += huber_loss(r).sum()
        return total_loss

    init_params = list(product(L0_init_set, A_init_set, alpha_init_set, B_init_set))
    best_loss = float('inf')
    best_params = None

    for init_param in tqdm(init_params, desc="Initializing Parameters"):
        res = minimize(
            loss_fn0, init_param, method='L-BFGS-B', bounds=[(0, np.inf)] * 4,
            options={'maxiter': 100000, 'ftol': 1e-9, 'gtol': 1e-6, 'eps': 1e-8}
        )
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x

    logger.info(f"Initialization completed. Best Loss: {best_loss}, Best Params: {best_params}")
    return best_params

def generate_init_params(init_param: list) -> list:
    """
    Generate initial parameter sets for MPL fitting by extending initial estimates.

    Args:
        init_param (list): Initial parameters [L0, A, alpha, B] from initialize_params.

    Returns:
        list: List of parameter tuples [L0, A, alpha, B, C, beta, gamma].
    """
    L0, A, alpha, B = init_param
    init_C_param = [1.0]       # Initial value for C parameter
    init_beta_param = [0.5]    # Initial value for beta parameter
    init_gamma_param = [0.5]   # Initial value for gamma parameter
    return list(product([L0], [A], [alpha], [B], init_C_param, init_beta_param, init_gamma_param))

def mpl_adam_fit(
    data,
    train_set,
    test_set,
    init_params,
    fig_folder,
    eval_interval=FIT_EVAL_INTERVAL,
    lr1=FIT_LR1,
    lr2=FIT_LR2,
    max_steps=FIT_MAX_STEPS,
    grad_norm_thr=FIT_GRAD_NORM_THR,
    loss_thr=FIT_LOSS_THR,
    patience=FIT_PATIENCE
):
    """
    Fit the MPL model using AdamW.

    Args:
        data (dict): Dataset containing steps, learning rates, and losses.
        train_set (list): List of training file names.
        test_set (list): List of test file names.
        init_params (list): List of initial parameter tuples for MPL.
        fig_folder (str): Directory to save fitting loss curve.
        eval_interval (int): Steps between logging evaluations (default from config).
        lr1 (float): Learning rate for L0, A, B, C parameters (default from config).
        lr2 (float): Learning rate for alpha, beta, gamma parameters (default from config).
        max_steps (int): Maximum training steps (default from config).
        grad_norm_thr (float): Gradient norm threshold for stopping training (default from config).
        loss_thr (float): Loss improvement threshold for stopping training (default from config).
        patience (int): Steps without improvement before stopping training (default from config).

    Returns:
        tuple: Best parameters and best loss value.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting MPL fitting with AdamW")

    torch_data = preprocess_data(data, train_set + test_set)
    best_params, best_loss = None, float('inf')

    for init_param in init_params:
        logger.info(f"Initializing with parameters: {init_param}")
        model = MPL(*init_param)
        optimizer = torch.optim.AdamW([
            {"params": [model.L0, model.A, model.B, model.C], "lr": lr1},
            {"params": [model.alpha, model.beta, model.gamma], "lr": lr2},
        ])

        loss_history, min_loss, steps_no_improve = [], float('inf'), 0

        for step in tqdm(range(max_steps), desc="Training Progress"):
            total_loss = compute_loss(model, torch_data, train_set, optimizer)
            loss_history.append(total_loss.item())

            if total_loss < min_loss - loss_thr:
                min_loss = total_loss.item()
                steps_no_improve = 0
            else:
                steps_no_improve += 1

            if step > patience and steps_no_improve >= patience:
                logger.info(f"Stopping at step {step}: No improvement for {patience} steps.")
                break

            grad_norm = compute_grad_norm(model)
            if grad_norm < grad_norm_thr:
                logger.info(f"Stopping at step {step}: Gradient norm {grad_norm:.2e} < {grad_norm_thr:.2e}")
                break

            if total_loss < best_loss:
                best_loss = total_loss.item()
                best_params = [p.item() for p in model.parameters()]
                logger.info(f"New best loss found: {best_loss}")

            if step % eval_interval == 0:
                log_step(step, total_loss, best_loss, model, grad_norm)

        plot_loss_curve(loss_history, fig_folder)

    logger.info(f"Fitting complete. Best Loss: {best_loss}, Best Params: {best_params}")
    return best_params, best_loss