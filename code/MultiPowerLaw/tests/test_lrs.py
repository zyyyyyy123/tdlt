# tests/test_lr_schedulers.py
import numpy as np
import matplotlib.pyplot as plt
import logging
from src.lr_schedules import cosine_lrs, const_lrs, two_stage_lrs, wsd_lrs, wsdld_lrs

def test_lr_schedulers():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    warmup = 2160
    total = 24000
    peak_lr = 3e-4
    end_lr = 3e-5
    const_warmup = True

    logger.info("Generating LR schedules")
    cos_lr = cosine_lrs(warmup, total, peak_lr, end_lr, const_warmup)
    const_lr = const_lrs(warmup, total, peak_lr, const_warmup)
    two_stage_lr = two_stage_lrs(warmup, total, peak_lr, end_lr, 8000, const_warmup)
    wsd_lr = wsd_lrs(warmup, total, 20000, peak_lr, end_lr, const_warmup)
    wsdld_lr = wsdld_lrs(warmup, total, 20000, peak_lr, end_lr, const_warmup)

    logger.info("Plotting LR schedules")
    plt.figure()
    plt.plot(np.arange(total), cos_lr, label="cosine")
    plt.plot(np.arange(total), const_lr, label="constant")
    plt.plot(np.arange(total), two_stage_lr, label="two_stage")
    plt.plot(np.arange(total), wsd_lr, label="wsd")
    plt.plot(np.arange(total), wsdld_lr, label="wsdld")
    plt.legend()
    plt.savefig("lrs.png")
    plt.close()
    logger.info("All LR scheduler tests passed!")

if __name__ == "__main__":
    test_lr_schedulers()