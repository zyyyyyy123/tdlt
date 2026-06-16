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
  - Stability checks: full/tail/endpoint sweep, paired block bootstrap on WSD,
    placebo residual controls, single-source transfer matrix, and two-source
    LOSO.
  - The old `20000-30000` window is retained in the raw CSV for continuity, but
    it is no longer treated as a primary test because its motivation is weak.
  - Important limitation: the bootstrap is within-curve over time blocks. It is
    not a schedule-level confidence interval because the dataset has only three
    schedules.

## Frozen 811 Selection

The 811-selected configuration is:

```text
spline_s0.01_shrink1_clip0.15
```

This also ranks best on WSD full MAE inside the audited grid, but the selection
rule is based on 811. That said, this does not by itself prove the parameter is
the best scientific choice; complexity diagnostics below suggest using the
smoother `s=0.1` setting as the conservative report parameter.

Top validation/test configs:

| split | config | full MAE | tail MAE `27126-33906` | endpoint |
|---|---|---:|---:|---:|
| 811 validation | `spline_s0.01_shrink1_clip0.15` | 0.016447 | 0.015154 | 0.017371 |
| WSD test | `spline_s0.01_shrink1_clip0.15` | 0.020878 | 0.021855 | 0.004600 |
| WSD test | `spline_s0.05_shrink1_clip0.15` | 0.021281 | see CSV | 0.004676 |
| WSD test | `spline_s0.1_shrink1_clip0.15` | 0.021741 | see CSV | 0.004744 |

Interpretation: the exact smoothing value changes the WSD full MAE by less than
about `0.001` across the strongest `shrink=1` settings, so the conclusion is
not dependent on the newly selected `s=0.01`.

## WSD Full/Tail/Endpoint Sweep

Frozen 811-selected spline on WSD:

| window | momentum MAE | spline MAE | improvement | relative improvement |
|---|---:|---:|---:|---:|
| full | 0.037720 | 0.020878 | 0.016843 | 44.65% |
| `1000-10000` | 0.033264 | 0.005579 | 0.027685 | 83.23% |
| `10000-20000` | 0.037426 | 0.022589 | 0.014837 | 39.64% |
| `27126-33906` | 0.038754 | 0.021855 | 0.016898 | 43.60% |
| `30000-33906` | 0.033631 | 0.008762 | 0.024869 | 73.95% |
| last 2048 sampled | 0.033819 | 0.009557 | 0.024261 | 71.74% |

Interpretation: the stable parts of the claim are full trajectory, tail/decay
region, and endpoint behavior. The legacy `20000-30000` window is not needed for
the main narrative.

## Parameter Overfit Assessment

Evidence against WSD-specific parameter overfit:

- `spline_s0.01_shrink1_clip0.15` is selected by 811, not WSD.
- Across the parameter grid, validation full-MAE rank and WSD full-MAE rank are
  almost identical: Spearman `0.989286`, Pearson `0.997236`.
- The selected config ranks first on both 811 and WSD full MAE.
- Negative controls fail, so the gain is not reproduced by arbitrary smooth
  residual templates.

Evidence for caution:

- The selected `s=0.01` curve has lower train error, but also higher
  high-frequency variation than `s=0.05` or `s=0.1`.
- The residual clip does not explain the result: raw residuals stay inside
  `[-0.063, 0.052]`, so `clip=0.15` is inactive.
- Since `s=0.05` and `s=0.1` give WSD full MAE `0.021281` and `0.021741`, very
  close to `0.020878`, the main result should be reported as a stable spline
  family rather than as a special property of `s=0.01`.

Recommended reporting choice: use `s=0.1, shrink=1` as the conservative main
parameter, and mention that 811 selection slightly prefers `s=0.01` with nearly
identical WSD behavior.

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
| `27126-33906` | 2048 | 0.016898 | 0.010157 | 0.015298 | 0.021047 | 1.0 |

Interpretation: within the WSD curve, paired block bootstrap supports positive
MAE improvement even with large time blocks. This does not solve the
schedule-level sample-size limit.

## Negative Controls

Placebo residual templates on WSD:

| control | full MAE | full improvement | tail MAE `27126-33906` | tail improvement |
|---|---:|---:|---:|---:|
| true step spline | 0.020878 | 0.016843 | 0.021855 | 0.016898 |
| reversed residual | 0.049905 | -0.012185 | 0.047814 | -0.009061 |
| circular shift | 0.050053 | -0.012333 | 0.051580 | -0.012826 |
| block permutation | 0.050067 | -0.012346 | 0.049095 | -0.010341 |
| sign flip | 0.067432 | -0.029712 | 0.067303 | -0.028549 |

Interpretation: the result is not explained by arbitrary smooth residual
magnitude; the step alignment matters.

## LOSO / Transfer Matrix

Full-trajectory MAE improvement is positive for all 9 single-source or
two-source transfer directions. This supports a real shared residual template.

## Current Conclusion

- Stronger claim now supported: within this three-schedule dataset, the
  step-aligned residual spline family is stable under 811-based parameter
  selection, WSD time-block bootstrap, negative controls, and transfer checks.
- The exact `s=0.01` parameter should not be overinterpreted. It is not WSD
  overfit, but it is higher-complexity than needed; `s=0.1` is a more
  conservative report setting with nearly the same WSD performance.
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
outputs/spline_parameter_overfit_summary.csv
outputs/spline_parameter_overfit_rank_correlation.csv
outputs/spline_parameter_overfit_selected_metrics.csv
outputs/spline_parameter_overfit_curve_complexity.csv
```
