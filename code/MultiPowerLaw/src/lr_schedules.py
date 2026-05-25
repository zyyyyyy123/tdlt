# src/lr_schedulers.py
import numpy as np

def cosine_lrs(warmup: int, total: int, peak_lr: float, end_lr: float, const_warmup: bool) -> np.ndarray:
    """
    Generate a cosine learning rate schedule with optional constant warmup.

    Args:
        warmup (int): Number of warmup steps.
        total (int): Total number of steps in the schedule.
        peak_lr (float): Peak learning rate after warmup.
        end_lr (float): Final learning rate after decay.
        const_warmup (bool): If True, use constant peak_lr during warmup; otherwise, linear ramp-up.

    Returns:
        np.ndarray: Array of learning rates over total steps.
    """
    step = np.arange(total)[warmup:]
    warmup_lrs = np.linspace(0, peak_lr, warmup) if not const_warmup else np.full(warmup, peak_lr)
    cosine_lrs = end_lr + 0.5 * (peak_lr - end_lr) * (1 + np.cos(np.pi * (step - warmup) / (total - warmup)))
    return np.concatenate((warmup_lrs, cosine_lrs))

def const_lrs(warmup: int, total: int, lr: float, const_warmup: bool) -> np.ndarray:
    """
    Generate a constant learning rate schedule with optional warmup.

    Args:
        warmup (int): Number of warmup steps.
        total (int): Total number of steps in the schedule.
        lr (float): Constant learning rate value.
        const_warmup (bool): If True, use constant lr during warmup; otherwise, linear ramp-up.

    Returns:
        np.ndarray: Array of learning rates over total steps.
    """
    warmup_lrs = np.linspace(0, lr, warmup) if not const_warmup else np.full(warmup, lr)
    return np.concatenate((warmup_lrs, np.full(total - warmup, lr)))

def two_stage_lrs(warmup: int, total: int, lr_a: float, lr_b: float, stage_a: int, const_warmup: bool) -> np.ndarray:
    """
    Generate a two-stage learning rate schedule with optional warmup.

    Args:
        warmup (int): Number of warmup steps.
        total (int): Total number of steps in the schedule.
        lr_a (float): Learning rate for the first stage.
        lr_b (float): Learning rate for the second stage.
        stage_a (int): Number of steps in the first stage (including warmup).
        const_warmup (bool): If True, use constant lr_a during warmup; otherwise, linear ramp-up.

    Returns:
        np.ndarray: Array of learning rates over total steps.
    """
    warmup_lrs = np.linspace(0, lr_a, warmup) if not const_warmup else np.full(warmup, lr_a)
    stage_a_lrs = np.full(stage_a - warmup, lr_a)
    stage_b_lrs = np.full(total - stage_a, lr_b)
    return np.concatenate((warmup_lrs, stage_a_lrs, stage_b_lrs))

def wsd_lrs(warmup: int, total: int, decay: int, peak_lr: float, end_lr: float, const_warmup: bool) -> np.ndarray:
    """
    Generate a WSD (Warmup-Stable-Decay) learning rate schedule.

    Args:
        warmup (int): Number of warmup steps.
        total (int): Total number of steps in the schedule.
        decay (int): Step at which decay begins.
        peak_lr (float): Peak learning rate after warmup.
        end_lr (float): Final learning rate after decay.
        const_warmup (bool): If True, use constant peak_lr during warmup; otherwise, linear ramp-up.

    Returns:
        np.ndarray: Array of learning rates over total steps.
    """
    step = np.arange(total)[decay:]
    warmup_lrs = np.linspace(0, peak_lr, warmup) if not const_warmup else np.full(warmup, peak_lr)
    decay_lrs = peak_lr ** ((total - step) / (total - decay)) * end_lr ** ((step - decay) / (total - decay))
    return np.concatenate((warmup_lrs, np.full(decay - warmup, peak_lr), decay_lrs))

def wsdld_lrs(warmup: int, total: int, decay: int, peak_lr: float, end_lr: float, const_warmup: bool) -> np.ndarray:
    """
    Generate a WSDLD (Warmup-Stable-Decay-Linear-Decay) learning rate schedule.

    Args:
        warmup (int): Number of warmup steps.
        total (int): Total number of steps in the schedule.
        decay (int): Step at which linear decay begins.
        peak_lr (float): Peak learning rate after warmup.
        end_lr (float): Final learning rate after decay.
        const_warmup (bool): If True, use constant peak_lr during warmup; otherwise, linear ramp-up.

    Returns:
        np.ndarray: Array of learning rates over total steps.
    """
    step = np.arange(total)[decay:]
    warmup_lrs = np.linspace(0, peak_lr, warmup) if not const_warmup else np.full(warmup, peak_lr)
    decay_lrs = peak_lr * (1 - (step - decay) / (total - decay)) + end_lr * (step - decay) / (total - decay)
    return np.concatenate((warmup_lrs, np.full(decay - warmup, peak_lr), decay_lrs))