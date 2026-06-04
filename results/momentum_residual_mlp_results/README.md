# Momentum-Law Residual MLP

## Protocol

- Train run: `cosine`
- Validation run: `811`
- Test run: `wsd`
- Start step: `1000`
- m: `64`
- Selection rule: lowest validation MAE. The test run is not used to select seed or shrink.
- Shrink grid: `0.0, 0.75, 1.0, 1.25`. The shrink is selected only on the
  validation run; the test run is held out until final reporting.

## Selected Model

- Baseline: `momentum_s2`, formula `L0 + A*S1^-alpha - C*S2`
- Residual target: `log_residual`
- Target definition: `log(loss) - log(momentum_s2)`
- Feature set: `kernel_summary`
- Features: `S1, kernel_ewm_diff_decay_0.9, kernel_ewm_diff_decay_0.99, kernel_ewm_diff_decay_0.995, kernel_ewm_diff_decay_0.999, kernel_diff_sum, kernel_abs_diff_sum, kernel_diff_max, kernel_diff_min, kernel_nonzero_diff_count`
- Hidden layers: `(64, 32)`
- Activation: `relu`
- Scaler: `physics`
- Seed: `3081`
- Shrink: `1.0`

## Test Result

| model | test_run | MAE | R2 | endpoint_abs_diff |
|---|---|---:|---:|---:|
| momentum_s2 | wsd | 0.03784420 | 0.92566899 | 0.02226007 |
| momentum_residual_mlp | wsd | 0.03387169 | 0.94031545 | 0.00930743 |

MAE improvement over momentum baseline: `10.497%`.

## Files

- `summary.json`: protocol, selected model, and headline metrics.
- `metrics.csv`: train/validation/test metrics for baseline and all residual trials.
- `trials.csv`: residual MLP seed/shrink search table.
- `predictions.csv`: true loss, momentum prediction, and selected residual MLP prediction.
- `fit.png`: curve and residual visualization.
