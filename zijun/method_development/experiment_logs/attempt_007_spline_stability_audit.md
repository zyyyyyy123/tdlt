# Attempt 007: Smooth Residual Spline Stability Audit

- Idea: stress-test the smooth residual spline mainline before making it the
  central narrative. The audit targets statistical validity, parameter
  reasonableness, and robustness against simple placebo residual templates.
- Process: implemented in `scripts/run_spline_stability_audit.py`. The script
  reads the three-schedule momentum table from
  `results/momentum_residual_mlp_results/predictions.csv`, samples
  `1000 <= step <= 33906` every 2 steps, fits residual splines on source
  schedules, and evaluates held-out schedules.
- Protocol:
  - Main frozen-selection test: train on cosine, choose spline parameters using
    811 full-trajectory MAE, then report WSD once.
  - Stability checks: window sweep, paired block bootstrap on WSD, placebo
    residual controls, single-source transfer matrix, and two-source LOSO.
  - Important limitation: the bootstrap is within-curve over time blocks. It is
    not a schedule-level confidence interval because the dataset has only three
    schedules.

## Frozen 811 Selection

The 811-selected configuration is:

```text
spline_s0.01_shrink1_clip0.15
```

This also ranks best on WSD full MAE inside the audited grid, but the selection
rule is based on 811.

Top validation/test configs:

| split | config | full MAE | hard-window MAE | endpoint |
|---|---|---:|---:|---:|
| 811 validation | `spline_s0.01_shrink1_clip0.15` | 0.016447 | 0.029224 | 0.017371 |
| WSD test | `spline_s0.01_shrink1_clip0.15` | 0.020878 | 0.037672 | 0.004600 |
| WSD test | `spline_s0.05_shrink1_clip0.15` | 0.021281 | 0.037662 | 0.004676 |
| WSD test | `spline_s0.1_shrink1_clip0.15` | 0.021741 | 0.037652 | 0.004744 |

Interpretation: the exact smoothing value changes the WSD full MAE by less than
about `0.002` across the strongest `shrink=1` settings, so the conclusion is
not dependent on only the previously reported `s=0.1`.

## WSD Window Sweep

Frozen 811-selected spline on WSD:

| window | momentum MAE | spline MAE | improvement | relative improvement |
|---|---:|---:|---:|---:|
| full | 0.037720 | 0.020878 | 0.016843 | 44.65% |
| `1000-10000` | 0.033264 | 0.005579 | 0.027685 | 83.23% |
| `10000-20000` | 0.037426 | 0.022589 | 0.014837 | 39.64% |
| `20000-30000` | 0.043619 | 0.037672 | 0.005947 | 13.63% |
| `27126-33906` | 0.038754 | 0.021855 | 0.016898 | 43.60% |
| `30000-33906` | 0.033631 | 0.008762 | 0.024869 | 73.95% |
| last 2048 sampled | 0.033819 | 0.009557 | 0.024261 | 71.74% |

Interpretation: the hard `20000-30000` window improves, but much less than the
early and tail windows. The main claim should mention this unevenness.

## Paired Block Bootstrap

Quantity bootstrapped on WSD:

```text
abs_error(momentum) - abs_error(spline)
```

| window | block size | mean improvement | q05 | q50 | q95 | P(>0) |
|---|---:|---:|---:|---:|---:|---:|
| full | 128 | 0.016843 | 0.015548 | 0.016767 | 0.018288 | 1.0 |
| full | 512 | 0.016843 | 0.013868 | 0.016505 | 0.019355 | 1.0 |
| full | 2048 | 0.016843 | 0.010497 | 0.015308 | 0.020088 | 1.0 |
| `20000-30000` | 128 | 0.005947 | 0.004968 | 0.005872 | 0.006812 | 1.0 |
| `20000-30000` | 512 | 0.005947 | 0.004454 | 0.005692 | 0.006680 | 1.0 |
| `20000-30000` | 2048 | 0.005947 | 0.005274 | 0.006000 | 0.006702 | 1.0 |
| `27126-33906` | 2048 | 0.016898 | 0.010157 | 0.015298 | 0.021047 | 1.0 |

Interpretation: within the WSD curve, paired block bootstrap supports positive
MAE improvement even with large time blocks. This does not solve the
schedule-level sample-size limit.

## Negative Controls

Placebo residual templates on WSD:

| control | full MAE | full improvement | hard-window MAE | hard-window improvement |
|---|---:|---:|---:|---:|
| true step spline | 0.020878 | 0.016843 | 0.037672 | 0.005947 |
| reversed residual | 0.049905 | -0.012185 | 0.052223 | -0.008604 |
| circular shift | 0.050053 | -0.012333 | 0.055138 | -0.011518 |
| block permutation | 0.050067 | -0.012346 | 0.054264 | -0.010645 |
| sign flip | 0.067432 | -0.029712 | 0.069794 | -0.026175 |

Interpretation: the result is not explained by arbitrary smooth residual
magnitude; the step alignment matters.

## LOSO / Transfer Matrix

Full-trajectory MAE improvement is positive for all 9 single-source or
two-source transfer directions. Hard-window MAE improvement is positive for 7
of 9 directions; the two failures are transfers into cosine when WSD is part of
the source. This supports a real shared residual template, while showing that
the hard window is less robust than full-trajectory performance.

## Current Conclusion

- Stronger claim now supported: within this three-schedule dataset, the
  step-aligned residual spline is stable under 811-based parameter selection,
  WSD time-block bootstrap, negative controls, and most transfer directions.
- Claim still not supported: broad schedule-level statistical generalization.
  The independent experimental unit is a schedule/run, and there are only three.
- Recommended wording: "The spline gives a robust descriptive improvement on
  the available cosine/811/WSD curves and reveals a transferable step-aligned
  residual template; larger schedule sets would be needed for schedule-level
  confidence intervals."

Key output files:

```text
outputs/spline_stability_parameter_grid.csv
outputs/spline_stability_selected_metrics.csv
outputs/spline_stability_selected_improvements.csv
outputs/spline_stability_loso_transfer.csv
outputs/spline_stability_block_bootstrap.csv
outputs/spline_stability_negative_controls.csv
```
