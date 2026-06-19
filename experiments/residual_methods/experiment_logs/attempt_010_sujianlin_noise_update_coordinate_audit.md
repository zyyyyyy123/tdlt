# Attempt 010: Sujianlin Noise/Update Coordinate Audit

- Goal: test whether sujianlin-inspired schedule-only coordinates explain the
  momentum residual better than the existing step/S1 controls.
- Protocol: read `results/momentum_residual_mlp_results/predictions.csv`;
  compute `residual = log(loss) - log(momentum_s2)`; use points
  `1000 <= step <= 33906` with every-2-step sampling; train interpolation
  templates on `cosine`; use `811` only to select coordinate/config; report
  `wsd` as held-out test.
- Model: full-resolution coordinate interpolation from the cosine residual,
  with conservative configs over raw vs `roll501` residual target, shrink
  `{0.5, 0.75, 1.0}`, and residual clip `{0.15, 0.25}`.

## Proxy Construction

- Controls:
  - `step_abs`: absolute step coordinate, the prior task2 reference.
  - `s1_raw`, `s1_ratio`: cumulative LR controls.
- BatchSize/noise-scale inspired proxies:
  - `sqrt_cum_lr2 = sqrt(sum lr^2)`.
  - `noise_ratio = sum lr^2 / (s1^2 + eps)` for several `eps`.
- Adam/update-RMS inspired proxies:
  - `effective_update_time = sum lr / sqrt(EWMA(lr^2; half_life) + eps)`.
  - `softsign_lr_time = sum lr / (eps + sqrt(EWMA(lr^2; half_life)))`.
  - Half-lives tested: `64`, `512`, `4096`; epsilon variants were swept.

These are deliberately only schedule proxies. There is no gradient variance,
true Adam second moment, update RMS, batch-size state, or optimizer state in
the available file.

## Key WSD Results

Selected on `811` full MAE:

| rank | coordinate | selected config | WSD full MAE | WSD tail `27126-33906` | WSD last-2048 | corr | sign agreement |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `step_abs` | `interp_raw_shrink1_clip0.15` | `0.020767` | `0.021707` | `0.009310` | `0.936` | `0.810` |
| 2 | `sqrt_cum_lr2` | `interp_roll501_shrink1_clip0.15` | `0.036435` | `0.037196` | `0.033058` | `0.160` | `0.671` |
| 3 | `s1_raw` | `interp_roll501_shrink1_clip0.15` | `0.036698` | `0.037196` | `0.033058` | `0.122` | `0.639` |
| 4 | `effective_update_hl4096_epslr2_1e-06` | `interp_roll501_shrink1_clip0.15` | `0.036811` | `0.037196` | `0.033058` | `0.098` | `0.626` |
| 5 | `softsign_lr_time_hl4096_epslr_1e-03` | `interp_roll501_shrink1_clip0.15` | `0.036824` | `0.037196` | `0.033058` | `0.095` | `0.625` |
| 23 | `s1_ratio` | `interp_roll501_shrink0.5_clip0.15` | `0.037606` | `0.038539` | `0.033432` | `-0.069` | `0.546` |

Momentum baseline WSD full MAE was `0.037720`. The best sujianlin proxy
(`sqrt_cum_lr2`) improves it by only `0.001286` absolute MAE (`3.41%`), while
`step_abs` improves it by `0.016953` (`44.94%`).

## Support And Warp

| coordinate | WSD outside cosine support | median abs warp | q90 abs warp |
|---|---:|---:|---:|
| `step_abs` | `0.000` | `0` | `0` |
| `sqrt_cum_lr2` | `0.614` | `5119` | `16963` |
| `s1_raw` | `0.464` | `3002` | `12317` |
| `effective_update_hl4096_epslr2_1e-06` | `0.420` | `2529` | `10997` |
| `softsign_lr_time_hl4096_epslr_1e-03` | `0.411` | `2507` | `10781` |
| `s1_ratio` | `0.018` | `3692` | `5713` |

The proxy coordinates mostly behave like smoothed progress clocks with large
support/phase mismatch on WSD. Their residual correlations are far below the
step template, even when MAE is slightly better than the momentum baseline.

## Verdict

- `H1_proxy_coordinates_can_transfer_residual_phase`: mixed. A proxy can beat
  the momentum baseline slightly, but it is not competitive with the step
  template.
- `H2_noise_scale_proxy_is_enough`: negative result. The best noise proxy is
  `sqrt_cum_lr2`, but WSD residual corr is only `0.160` and WSD support
  mismatch is `0.614`.
- `H3_update_rms_or_softsign_time_is_better_than_s1`: mixed. The best
  update-RMS proxy beats `s1_ratio` and the baseline by a tiny amount, but its
  WSD corr is only `0.098` and median warp is about `2529` steps.
- Overall: mixed-to-negative for the sujianlin-inspired coordinates as
  schedule-only replacements. The result does not refute the real
  BatchSize/noise/update-RMS ideas; it only says these black-box proxies do not
  recover the strong task2 step-aligned residual signal without gradient or
  Adam-state information.

## Output Files

```text
scripts/run_sujianlin_noise_update_coordinate_audit.py
outputs/sujianlin_noise_update_validation_grid.csv
outputs/sujianlin_noise_update_coordinate_selection.csv
outputs/sujianlin_noise_update_wsd_window_metrics.csv
outputs/sujianlin_noise_update_residual_alignment.csv
outputs/sujianlin_noise_update_domain_support.csv
outputs/sujianlin_noise_update_warp_summary.csv
outputs/sujianlin_noise_update_coordinate_catalog.csv
outputs/sujianlin_noise_update_hypothesis_summary.csv
```
