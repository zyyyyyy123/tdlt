# src/utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.special import huber

def torch_huber(delta: float, r: torch.Tensor) -> torch.Tensor:
    """Compute Huber loss for a tensor."""
    return torch.where(torch.abs(r) < delta, 0.5 * r ** 2, delta * (torch.abs(r) - 0.5 * delta))

def huber_loss(r: np.ndarray, delta: float = 0.001) -> np.ndarray:
    """Compute Huber loss for numpy array."""
    return huber(delta, r)

def preprocess_data(data: dict, file_names: list) -> dict:
    """Preprocess data into torch tensors."""
    torch_data = {}
    for file_name in file_names:
        torch_data[file_name] = {
            "step": torch.tensor(data[file_name]["step"], dtype=torch.int32),
            "lrs": torch.tensor(data[file_name]["lrs"], dtype=torch.float64),
            "loss": torch.tensor(data[file_name]["loss"], dtype=torch.float32),
        }
        lr_sum = torch.cumsum(torch_data[file_name]["lrs"], dim=0, dtype=torch.float64)
        torch_data[file_name]["S1"] = lr_sum[torch_data[file_name]["step"]]
        torch_data[file_name]["lr_sum"] = lr_sum
        lr_gap = torch.zeros_like(torch_data[file_name]["lrs"])
        lr_gap[1:] = torch.diff(torch_data[file_name]["lrs"])
        torch_data[file_name]["lr_gap"] = lr_gap
    return torch_data

def compute_loss(model, torch_data, train_set, optimizer):
    """Compute total loss and perform optimization step."""
    optimizer.zero_grad()
    total_loss = 0.0
    for file_name in train_set:
        args = [torch_data[file_name][key] for key in ["S1", "lrs", "lr_sum", "step", "lr_gap", "loss"]]
        total_loss += model(*args)
    total_loss.backward()
    optimizer.step()
    return total_loss

def compute_grad_norm(model):
    """Compute L2 norm of gradients."""
    grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
    return torch.cat(grads).norm() if grads else torch.tensor(0.0)

def log_step(step, total_loss, best_loss, model, grad_norm):
    """
    Log training progress with current step details.

    Args:
        step (int): Current training step.
        total_loss (float): Loss at the current step.
        best_loss (float): Best loss observed so far.
        model (nn.Module): MPL model instance.
        grad_norm (float): Gradient norm at the current step.
    """
    logger = logging.getLogger(__name__)
    params = {name: param.item() for name, param in model.named_parameters()}
    logger.info(f"Step {step:4d}: Loss={total_loss:.6f}, Best Loss={best_loss:.6f}, Grad Norm={grad_norm:.2e}")
    logger.info(f"Parameters: L0={params['L0']:.4f}, A={params['A']:.4f}, alpha={params['alpha']:.4f}, "
                f"B={params['B']:.4f}, C={params['C']:.4f}, beta={params['beta']:.4f}, gamma={params['gamma']:.4f}")

def plot_loss_curve(loss_history, fig_folder):
    """Plot and save loss curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(loss_history)), loss_history, label="Fitting Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{fig_folder}/loss_monitor.png")
    plt.close()