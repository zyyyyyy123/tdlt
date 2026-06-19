# Attempt 005: Intrinsic-Time Smooth Residual Spline

- Idea: follow the FSL intrinsic-time viewpoint and replace raw step with
  intrinsic time as the spline input. With fixed batch size and discrete
  `h = 1`, the FSL intrinsic time `t = integral phi(r) dr` corresponds to the
  cumulative learning rate `S1`.
- Process: implemented in
  `scripts/run_momentum_residual_intrinsic_spline.py`. The script compares the
  current step-spline residual against two intrinsic-time variants:
  `intrinsic_s1_raw`, which directly evaluates the cosine-fitted spline at WSD
  `S1`, and `intrinsic_s1_clamp`, which clamps WSD `S1` to the cosine training
  range before evaluating the spline. After a follow-up suggestion, the script
  also evaluates `intrinsic_s1_ratio`, which uses per-schedule progress
  `S1(t) / max_t S1(t)` so all schedules are mapped to roughly `[0, 1]`.
- Protocol: use reproduced momentum predictions with `is_sampled == True`,
  `1000 <= step <= 33906`; fit residuals on cosine only; evaluate WSD full
  sampled trajectory and `20000-30000`.
- Diagnostic detail: cosine sampled `S1` ranges from about `1.000` to `18.649`,
  while WSD sampled `S1` ranges from about `1.001` to `29.777`, so the raw
  intrinsic-time transfer extrapolates far beyond the cosine-fitted domain.

WSD sampled trajectory metrics:

| Model | MAE | RMSE | MAPE | R2 |
|---|---:|---:|---:|---:|
| Momentum baseline | 0.037215 | 0.046671 | 0.013198 | 0.928182 |
| Step spline `s=0.1`, shrink `1.0` | 0.020656 | 0.024734 | 0.007360 | 0.979829 |
| Intrinsic `S1` raw `s=0.5`, shrink `0.5` | 0.398934 | 0.558339 | 0.144731 | -9.278585 |
| Intrinsic `S1` clamp `s=0.5`, shrink `0.5` | 0.044536 | 0.055218 | 0.015851 | 0.899469 |
| Intrinsic `S1` ratio `s=0.5`, shrink `0.5` | 0.052949 | 0.116302 | 0.017693 | 0.554023 |

WSD `20000-30000` window metrics:

| Model | MAE | RMSE | MAPE | R2 |
|---|---:|---:|---:|---:|
| Momentum baseline | 0.042607 | 0.052674 | 0.015504 | -0.209511 |
| Step spline `s=0.1`, shrink `1.0` | 0.035469 | 0.036291 | 0.012799 | 0.425859 |
| Intrinsic `S1` raw `s=0.5`, shrink `0.5` | 0.831274 | 0.832267 | 0.300030 | -300.960747 |
| Intrinsic `S1` clamp `s=0.5`, shrink `0.5` | 0.057221 | 0.067519 | 0.020822 | -0.987360 |
| Intrinsic `S1` ratio `s=0.5`, shrink `0.5` | 0.045473 | 0.056214 | 0.016529 | -0.377556 |

Current conclusion:

- Direct intrinsic-time residual transfer is a negative result. Raw `S1`
  extrapolation is unstable because WSD accumulates much larger `S1` than
  cosine by the end of training.
- Clamping removes the extreme extrapolation failure, but it still underperforms
  the original momentum baseline, suggesting that the residual learned after the
  momentum-law correction is not simply aligned by cumulative LR.
- Per-schedule progress normalization fixes the domain mismatch, but still
  underperforms the original momentum baseline on both full WSD and the
  `20000-30000` window. This suggests that the mismatch is not only caused by
  scale/extrapolation of `S1`.
- For this dataset, the useful residual structure appears more step-aligned than
  intrinsic-time-aligned. A better FSL-inspired next step is likely not replacing
  the spline coordinate with `S1` alone, but adding the FSL noise/convolution
  term or using a two-coordinate residual model involving both step and `S1`.

Key output files:

```text
outputs/momentum_residual_intrinsic_spline_metrics.csv
outputs/key_momentum_residual_intrinsic_spline_predictions.csv
figures/momentum_residual_intrinsic_spline_full.png
figures/momentum_residual_intrinsic_spline_20000_30000.png
```
