import numpy as np
from src.lr_schedules import cosine_lrs, const_lrs, two_stage_lrs, wsd_lrs, wsdld_lrs


def test_lr_schedulers():
    warmup = 2160
    total = 24000
    peak_lr = 3e-4
    end_lr = 3e-5
    const_warmup = True

    cos_lr = cosine_lrs(warmup, total, peak_lr, end_lr, const_warmup)
    const_lr = const_lrs(warmup, total, peak_lr, const_warmup)
    two_stage_lr = two_stage_lrs(warmup, total, peak_lr, end_lr, 8000, const_warmup)
    wsd_lr = wsd_lrs(warmup, total, 20000, peak_lr, end_lr, const_warmup)
    wsdld_lr = wsdld_lrs(warmup, total, 20000, peak_lr, end_lr, const_warmup)

    schedules = [cos_lr, const_lr, two_stage_lr, wsd_lr, wsdld_lr]
    for lrs in schedules:
        assert lrs.shape == (total,)
        assert np.all(np.isfinite(lrs))
        assert np.all(lrs >= 0)

    assert np.isclose(cos_lr[-1], end_lr, atol=1e-7)
    assert np.isclose(const_lr[-1], peak_lr, atol=1e-7)
    assert np.isclose(two_stage_lr[-1], end_lr, atol=1e-7)
    assert np.isclose(wsd_lr[-1], end_lr, atol=1e-7)
    assert np.isclose(wsdld_lr[-1], end_lr, atol=1e-7)

if __name__ == "__main__":
    test_lr_schedulers()
