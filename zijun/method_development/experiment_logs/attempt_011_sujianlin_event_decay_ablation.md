# Attempt 011: Sujianlin Event-Local Decay Ablation

## Motivation

`reference/sujianlin.pdf` points to Su Jianlin's discussions of optimizer
history as a weight-decay or EMA-like process, especially under dynamic
learning-rate and weight-decay schedules, Cosine/WSD comparisons, and endpoint
loss arguments for linear decay. This repository does not have true weight
norms or weight decay trajectories, so this attempt uses the known LR schedule
as an event-local proxy for history:

- LR drop impulses and cumulative drop mass approximate schedule-driven history.
- Post-drop age, `s1_after_first_drop`, EWMA(drop), and tail saturation
  approximate local memory after a decay event.
- Linear decay progress and endpoint/tail powers target the linear-decay and
  endpoint-loss part of the theory.

## Protocol

- Script: `zijun/method_development/scripts/run_sujianlin_event_decay_ablation.py`
- Input: `results/momentum_residual_mlp_results/predictions.csv`
- Filter: `1000 <= step <= 33906`, every 2 steps from step 1000.
- Target residual: `log(loss) - log(momentum_s2)`.
- Train: `cosine`.
- Validation and selection: `811`.
- Test report only: `wsd`. WSD is not used for config selection.
- Selection rules:
  - `811_full`: sort by 811 full MAE, then endpoint error.
  - `811_full_plus_tail`: `0.5 * full + 0.3 * tail_27126_33906 + 0.2 * last_2048`,
    then full MAE and endpoint error.

## Model Classes

- `event_decay_only`: ridge on event-local features only.
- `step_reference`: conservative one-dimensional step template using smoothed
  spline or rolling interpolation.
- `step_plus_event_leftover`: fit the step template first, then fit event-local
  ridge on the cosine leftover.

Total candidates: 4,624.

## Event Geometry Check

The schedule geometry is meaningfully different across runs:

| schedule | positive drop samples | large drop events | first positive drop | max LR drop | total drop mass |
|---|---:|---:|---:|---:|---:|
| 811 | 2 | 2 | 27126 | 6.837722e-04 | 0.000900 |
| cosine | 16453 | 0 | 1002 | 8.338790e-08 | 0.000898 |
| wsd | 3391 | 0 | 27126 | 6.787657e-07 | 0.000900 |

This matters: cosine supplies continuous small decay, 811 supplies two abrupt
events, and WSD supplies many small linear-tail drops. A true WD-state model
would need information we do not have.

## Selected Configs

| selection | class | config |
|---|---|---|
| 811_full | event_decay_only | `linear_endpoint_ridge10_shrink0.75_clip0.15` |
| 811_full | step_reference | `spline_s0.05_shrink1_clip0.15` |
| 811_full | step_plus_event_leftover | `step[spline_s0.05_shrink0.75_clip0.15]_event[linear_endpoint_ridge0.01_shrink1_clip0.25]` |
| 811_full_plus_tail | event_decay_only | `linear_endpoint_ridge0.01_shrink0.25_clip0.15` |
| 811_full_plus_tail | step_reference | `spline_s0.05_shrink1_clip0.15` |
| 811_full_plus_tail | step_plus_event_leftover | `step[spline_s0.05_shrink0.75_clip0.15]_event[linear_endpoint_ridge0.01_shrink1_clip0.25]` |

The selected event feature family is `linear_endpoint`, not the more direct
drop-impulse or EWMA-drop families. The largest standardized coefficients are
on `linear_decay_progress`, `remaining_frac`/`endpoint_progress`, and
`linear_decay_progress_sq`.

## WSD Results

`811_full` selected WSD metrics:

| model | full MAE | full R2 | tail MAE | last_2048 MAE | endpoint_abs_diff |
|---|---:|---:|---:|---:|---:|
| momentum_baseline | 0.037720 | 0.926313 | 0.038754 | 0.033819 | 0.045668 |
| event_decay_only | 0.032596 | 0.945083 | 0.032285 | 0.031930 | 0.050142 |
| step_reference | 0.021281 | 0.978375 | 0.022268 | 0.010247 | 0.004676 |
| step_plus_event_leftover | 0.011191 | 0.993410 | 0.015937 | 0.011531 | 0.016615 |

`811_full_plus_tail` selected WSD metrics:

| model | full MAE | full R2 | tail MAE | last_2048 MAE | endpoint_abs_diff |
|---|---:|---:|---:|---:|---:|
| momentum_baseline | 0.037720 | 0.926313 | 0.038754 | 0.033819 | 0.045668 |
| event_decay_only | 0.032892 | 0.944072 | 0.034163 | 0.032758 | 0.047281 |
| step_reference | 0.021281 | 0.978375 | 0.022268 | 0.010247 | 0.004676 |
| step_plus_event_leftover | 0.011191 | 0.993410 | 0.015937 | 0.011531 | 0.016615 |

## Verdict

The event-local proxy carries real schedule information beyond the raw momentum
baseline. `event_decay_only` improves WSD full MAE by about 0.0051 under
`811_full`, but it does not beat the step template and it worsens endpoint
absolute error.

`step_plus_event_leftover` does add validation-selected marginal information
beyond the conservative step template on WSD full and decay-tail MAE:

- WSD full MAE: `0.021281 -> 0.011191`.
- WSD `tail_27126_33906` MAE: `0.022268 -> 0.015937`.

However, the gain is not an endpoint-loss win:

- WSD `last_2048` MAE is worse than step reference: `0.010247 -> 0.011531`.
- WSD endpoint absolute difference is worse: `0.004676 -> 0.016615`.

Interpretation: the Sujianlin-inspired history proxy helps as a mid-tail
leftover correction, but the selected signal collapses mostly to linear decay
progress and endpoint/tail coordinates rather than true LR-drop impulse memory.
The likely failure mode is missing state: without actual weight decay, weight
RMS, update RMS, or optimizer EMA state, LR events are only a schedule proxy.
Cosine training also has no abrupt drop support, while 811 validation has two
large drops and WSD has many small linear drops, so direct event impulses do
not transfer as cleanly as a step template plus low-rank linear-tail correction.

## Outputs

- `zijun/method_development/outputs/sujianlin_event_decay_trials.csv`
- `zijun/method_development/outputs/sujianlin_event_decay_selected_metrics_by_window.csv`
- `zijun/method_development/outputs/sujianlin_event_decay_coefficients.csv`
- `zijun/method_development/outputs/sujianlin_event_decay_feature_summary.csv`
- `zijun/method_development/outputs/sujianlin_event_decay_summary_verdict.csv`
