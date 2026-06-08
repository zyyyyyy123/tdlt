# Attempt 006: Momentum-MPL Hybrid Decay Correction

- Idea: test whether a low-degree MPL-inspired nonlinear decay response can
  improve the momentum law without relying on a flexible step-wise residual
  spline. The model keeps the reproduced momentum prediction as the main trend
  and fits a log-residual correction using saturated functions of `S2`, LR drop
  mass, and post-drop tail progress.
- Process: implemented in
  `scripts/run_momentum_mpl_hybrid_decay.py`. The script reads the three-run
  `momentum_s2` prediction table from
  `results/momentum_residual_mlp_results/predictions.csv`, samples
  `1000 <= step <= 33906` every 2 steps, fits only on cosine, selects the
  hybrid decay feature family and shrink using 811 full-trajectory MAE, and
  reports WSD as the held-out transfer target. The reproduced momentum output
  under `results/reproduction/momentum/` does not contain 811, so all numbers
  below should be compared within this experiment's table.
- Selected model:
  `hybrid_decay_progress_scale1_beta0.5_ridge1_shrink1.25`, with features
  `sat_s2`, `sat_drop_times_sat_tail`, and `sat_s2_times_progress`.

WSD sampled trajectory metrics:

| Model | MAE | RMSE | MAPE | R2 | endpoint_abs_diff |
|---|---:|---:|---:|---:|---:|
| Momentum baseline | 0.037720 | 0.047275 | 0.013382 | 0.926313 | 0.045668 |
| Mean residual shift | 0.037456 | 0.046958 | 0.013284 | 0.927298 | 0.046356 |
| Hybrid decay correction | 0.036592 | 0.045901 | 0.012963 | 0.930533 | 0.053108 |
| Step-spline reference | 0.021741 | 0.026068 | 0.007749 | 0.977594 | 0.004744 |

WSD `20000-30000` window metrics:

| Model | MAE | RMSE | MAPE | R2 | endpoint_abs_diff |
|---|---:|---:|---:|---:|---:|
| Momentum baseline | 0.043619 | 0.053750 | 0.015874 | -0.259430 | 0.082613 |
| Mean residual shift | 0.043167 | 0.053271 | 0.015709 | -0.237102 | 0.081906 |
| Hybrid decay correction | 0.041618 | 0.051610 | 0.015141 | -0.161141 | 0.077653 |
| Step-spline reference | 0.037652 | 0.038449 | 0.013586 | 0.355543 | 0.030412 |

Current conclusion:

- The hybrid decay correction gives a real but small WSD improvement over the
  local momentum baseline and is better than a constant residual shift on both
  full trajectory and the `20000-30000` window.
- It does not approach the step-spline residual reference. This suggests that a
  simple saturated transformation of `S2`/drop mass is not enough to explain the
  transferable residual structure.
- The next hybrid-law attempt should be more event-local: approximate MPL's
  drop-event response more directly, or add a very low-rank event basis, instead
  of using only global cumulative decay geometry.
- Use in slides: ablation/secondary research direction, not the main method.

Key output files:

```text
outputs/momentum_mpl_hybrid_decay_metrics.csv
outputs/momentum_mpl_hybrid_decay_all_metrics.csv
outputs/momentum_mpl_hybrid_decay_trials.csv
outputs/key_momentum_mpl_hybrid_decay_predictions.csv
figures/momentum_mpl_hybrid_decay_full.png
figures/momentum_mpl_hybrid_decay_20000_30000.png
figures/momentum_mpl_hybrid_decay_diagnostics.png
```
