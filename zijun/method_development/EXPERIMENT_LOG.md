# Experiment Log

## Attempt 000: Feature Ridge Sanity Check

- Idea: fit a direct ridge model on schedule features from cosine and predict
  other schedules.
- Process: implemented in `scripts/run_feature_fit.py`.
- Conclusion: useful only as a plumbing check. It fits cosine moderately but
  transfers poorly, so it should not be used as the final method.

## Attempt 001: High-Dimensional Residual Features

- Idea: learn `log(loss) - log(base_prediction)` using step, cumulative LR,
  LR-squared cumulative sum, LR drop mass, and momentum drop mass.
- Process: implemented in `scripts/run_residual_correction.py` using locally
  generated baseline predictions.
- Conclusion: too aggressive. The feature residual model can overfit cosine
  residuals and often hurts WSD transfer. Keep this as a negative ablation.

## Attempt 002: Smooth Residual Transfer Over Reproduced Momentum Baseline

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

## Data Prep 003: Momentum Multi-End Cosine Baseline

- Idea: reproduce the momentum baseline in the setting where only the first
  `sample_end` points are visible, then evaluate the unseen future of the same
  cosine run.
- Process: ran `code/reproduction_momentum.py` with `sample_interval=1`,
  `eval_runs=cosine`, and `sample_end` values from `2000` to `20000` by `2000`.
- Output directory:
  `baseline_results/momentum_multi_end_check/`.
- Conclusion: this produces the baseline CSVs needed for the next residual
  correction task. The full prediction table is local-only because it is large;
  the command is documented in `baseline_results/README.md`.

## Attempt 004: Roll5 Smooth Residual Target

- Idea: fit the smooth residual correction to the trailing 5 sampled-tick rolling
  mean loss instead of raw loss, while preserving raw loss for secondary
  evaluation.
- Process: implemented in
  `scripts/run_momentum_residual_spline_roll5.py`. The target is
  `loss_roll5 = rolling(5, min_periods=5).mean()` within each schedule, so the
  first four sampled points per schedule are dropped.
- Protocol: fit roll5 residuals on cosine, evaluate transfer to WSD, and report
  both roll5-target and raw-loss metrics for full sampled trajectory plus
  `20000-30000`.
- Current conclusion: the best WSD full roll5 MAE comes from
  `roll5_spline_s0.01_shrink1`. It improves full-trajectory roll5 metrics over
  the momentum baseline, but the `20000-30000` window remains mixed and should
  be reported separately.

Key WSD roll5 metrics:

| Model | Window | MAE | RMSE | R2 |
|---|---|---:|---:|---:|
| Momentum baseline | full | 0.024360 | 0.029668 | 0.969567 |
| Roll5 residual `s=0.01`, shrink `1.0` | full | 0.019673 | 0.023798 | 0.980417 |
| Momentum baseline | `20000-30000` | 0.034367 | 0.038544 | -0.480141 |
| Roll5 residual `s=0.01`, shrink `1.0` | `20000-30000` | 0.035483 | 0.035763 | -0.274301 |

Key output files:

```text
outputs/momentum_residual_spline_roll5_metrics.csv
outputs/key_momentum_residual_roll5_predictions.csv
figures/momentum_residual_spline_roll5_full.png
figures/momentum_residual_spline_roll5_20000_30000.png
```
