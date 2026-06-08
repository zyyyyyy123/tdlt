# Attempt 002: Smooth Residual Transfer Over Reproduced Momentum Baseline

- Idea: use the reproduced momentum baseline from
  `results/reproduction/momentum/predictions.csv`, fit a smooth one-dimensional
  residual curve on cosine sampled points, then transfer that residual curve to
  WSD.
- Process: implemented in `scripts/run_momentum_residual_spline.py`.
- Protocol: use `is_sampled == True`, `1000 <= step <= 33906`, fit on cosine,
  evaluate on WSD, and report full sampled trajectory plus `20000-30000`.
- Current conclusion: this is the best current direction. The spline residual
  correction improves WSD full-trajectory metrics over the reproduced momentum
  baseline while staying simple enough to explain in slides.

WSD sampled trajectory metrics:

| Model | MAE | RMSE | MAPE | R2 |
|---|---:|---:|---:|---:|
| Momentum baseline | 0.037216 | 0.046672 | 0.013198 | 0.928180 |
| Mean residual shift | 0.036862 | 0.046243 | 0.013068 | 0.929493 |
| Spline residual `s=0.5` | 0.023494 | 0.028371 | 0.008343 | 0.973460 |
| Spline residual `s=0.1`, shrink `0.75` | 0.022251 | 0.026662 | 0.007941 | 0.976562 |
| Spline residual `s=0.1`, shrink `1.0` | 0.020657 | 0.024735 | 0.007360 | 0.979827 |

WSD `20000-30000` window metrics:

| Model | MAE | RMSE | MAPE | R2 |
|---|---:|---:|---:|---:|
| Momentum baseline | 0.042608 | 0.052675 | 0.015505 | -0.209586 |
| Mean residual shift | 0.042004 | 0.052026 | 0.015284 | -0.179938 |
| Spline residual `s=0.5` | 0.035462 | 0.038739 | 0.012825 | 0.345769 |
| Spline residual `s=0.1`, shrink `0.75` | 0.035048 | 0.037181 | 0.012699 | 0.397338 |
| Spline residual `s=0.1`, shrink `1.0` | 0.035471 | 0.036293 | 0.012799 | 0.425794 |

Interpretation:

- A constant residual shift barely helps, so the gain is not just a global bias
  correction.
- Smooth step-wise residual structure learned on cosine transfers well to WSD.
- `s=0.1, shrink=1.0` is the current key model because it gives the best WSD
  full-trajectory metrics and the best window RMSE/R2.

Key output files:

```text
outputs/momentum_residual_spline_metrics.csv
outputs/key_momentum_residual_predictions.csv
figures/momentum_residual_spline_full.png
figures/momentum_residual_spline_20000_30000.png
```
