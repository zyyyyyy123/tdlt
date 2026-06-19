# Attempt 004: Roll5 Smooth Residual Target

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
