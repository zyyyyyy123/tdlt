# Attempt 012: Step Plus Event Leftover Stability Audit

## Motivation

Attempt 011 found that `step_plus_event_leftover` improves WSD full/tail MAE
over a conservative step template, but worsens endpoint behavior. This audit
tests whether that conclusion is stable and whether the gain really tracks the
hypothesis that a low-rank event/linear-tail residual is being captured.

The audit is deliberately smaller than Attempt 011. It uses 4096-bin step
spline fits with smoothing scaled by the effective point count, so the absolute
MAE values are not meant to replace the full-resolution Attempt 011 headline
numbers. The purpose is mechanism and stability testing:

- Does the aligned event leftover beat step-only on WSD full/tail?
- Do shifted/reversed/zero event-feature controls fail?
- Is the gain mainly driven by `linear_endpoint` features?
- What happens if `811` is used as an extra training source?
- Can validation endpoint constraints fix endpoint degradation?
- Is the WSD within-curve improvement stable under block bootstrap?

## Protocol

- Script:
  `zijun/method_development/scripts/run_step_plus_event_leftover_stability_audit.py`
- Input: `results/momentum_residual_mlp_results/predictions.csv`
- Filter: `1000 <= step <= 33906`, every 2 steps.
- Target residual: `log(loss) - log(momentum_s2)`.
- Main split: train `cosine`, test `wsd`.
- Extra split checks:
  - train `811`, test `wsd`;
  - train `cosine+811`, test `wsd`.
  No WSD loss is used for fitting or selection in the WSD test rows.
- Endpoint-selection grids store validation metrics only; WSD test endpoint
  metrics are written only for validation-selected configs.
- Main step template in this audit:
  - `step_reference`: binned spline `s=0.05`, shrink `1.0`, clip `0.15`.
  - `step_base_for_leftover`: binned spline `s=0.05`, shrink `0.75`, clip `0.15`.
- Main event leftover:
  - `linear_endpoint`, ridge `0.01`, shrink `1.0`.

## Main Stability Results

Using train `cosine`, test `wsd`, aligned `linear_endpoint` leftover:

| model | WSD full MAE | WSD tail `27126-33906` MAE | endpoint `30000-33906` MAE | endpoint abs diff |
|---|---:|---:|---:|---:|
| step reference | `0.035007` | `0.035694` | `0.029582` | `0.024591` |
| step + event leftover | `0.029644` | `0.031012` | `0.029159` | `0.031494` |

The binned audit reproduces the direction of Attempt 011:

- full MAE improves by `0.005363`;
- tail MAE improves by `0.004682`;
- endpoint-region MAE improves slightly in this binned audit;
- endpoint absolute difference still worsens.

This is consistent with the original full-resolution Attempt 011 result:
`step_plus_event_leftover` improves WSD full/tail but is not an endpoint-loss
win.

## 811 As Training Or Extra Evidence

| train source | target | step full MAE | step + event full MAE | verdict |
|---|---|---:|---:|---|
| `cosine` | `wsd` | `0.035007` | `0.029644` | supported |
| `811` | `wsd` | `0.029869` | `0.030152` | not supported |
| `cosine+811` | `wsd` | `0.031382` | `0.029322` | supported |

Interpretation:

- The WSD gain is not simply a generic property of fitting event features from
  any source schedule. Training on `811` alone does not transfer the event
  leftover to WSD.
- Adding `811` to cosine as a second training source still leaves a positive
  WSD margin, so the conclusion does not rely on using `811` only as a
  validation schedule.
- The most plausible reading is that cosine supplies the smooth step residual
  phase, while `811` helps constrain the event/tail correction when included as
  training data.

## Negative Controls

Train `cosine`, test `wsd`, same step base, same linear endpoint feature family:

| feature transform | WSD full MAE | tail MAE | endpoint abs diff |
|---|---:|---:|---:|
| aligned | `0.029644` | `0.031012` | `0.031494` |
| zero event | `0.035032` | `0.035836` | `0.030044` |
| circular shift 25% | `0.035132` | `0.036998` | `0.018553` |
| reverse time | `0.038455` | `0.036350` | `0.031027` |

The aligned features are clearly better on full and tail MAE. This supports the
claim that the improvement is tied to correctly aligned event/tail geometry,
not just adding arbitrary degrees of freedom. The endpoint caveat remains:
shifted features can look better on endpoint absolute difference while being
worse on the curve shape.

## Feature-Family Ablation

Train `cosine`, test `wsd`, `step_plus_event_leftover`:

| feature family | WSD full MAE | WSD tail MAE | comment |
|---|---:|---:|---|
| `linear_endpoint` | `0.029644` | `0.031012` | best full/tail stability |
| `drop_impulse` | `0.033357` | `0.037049` | weaker; direct drop impulses do not explain WSD tail |
| `event_tail_interactions` | `0.034400` | `0.031355` | tail close, full weaker |
| `event_history` | `0.082043` | `0.266411` | fails badly |
| `full_event_local` | `0.085251` | `0.284405` | fails badly |

This confirms the Attempt 011 interpretation: the selected signal is mainly
linear decay / endpoint-tail geometry, not true LR-drop impulse memory or a
generic high-dimensional event feature model.

## Endpoint Constraint Check

The endpoint-constrained validation variants did not resolve the endpoint
tradeoff in a useful way. The selected endpoint-guarded candidate still has
WSD endpoint absolute difference around `0.031491`, with WSD full MAE
`0.029716`.

Conclusion: endpoint should be handled as a separate constraint or separate
model component. Simply adding endpoint into the validation score does not
recover the pure step template endpoint behavior.

## WSD Block Bootstrap

Within-curve block bootstrap compares absolute-error improvement of
`step_plus_event_leftover` over `step_reference` on WSD.

| window | mean improvement | q05, block 2048 | prob positive |
|---|---:|---:|---:|
| full | `0.005363` | `0.003307` | `1.000` |
| tail `27126-33906` | `0.004682` | `0.001656` | `1.000` |
| early decay `27126-30000` | `0.010467` | `0.010467` | `1.000` |
| last 2048 | `0.000563` | `0.000563` | `1.000` |
| last 512 | `-0.000188` | `-0.000188` | `0.000` |

The full and tail gains are robust within the WSD curve. The last-512 result is
negative, matching the endpoint caveat.

## Current Conclusion

The stability audit supports the core part of the Attempt 011 conclusion:

1. `step_plus_event_leftover` captures a real, aligned event/linear-tail
   correction on top of the step template.
2. The gain is not reproduced by zero, shifted, or reversed event features.
3. The useful signal is specifically `linear_endpoint` style tail geometry,
   not direct LR-drop impulse memory.
4. Adding `811` as training data still preserves the WSD full/tail margin.

But the audit also narrows the claim:

1. Training on `811` alone does not transfer the event leftover to WSD.
2. Endpoint behavior remains unstable; the method should not be described as an
   endpoint-loss improvement.
3. The evidence is still within three schedules. Block bootstrap is
   within-curve evidence only, not schedule-level significance.

Use in slides: follow-up stability evidence for the event-leftover direction,
with endpoint caveat shown explicitly.

Key output files:

```text
scripts/run_step_plus_event_leftover_stability_audit.py
outputs/step_plus_event_stability_summary.csv
outputs/step_plus_event_stability_transfer_metrics.csv
outputs/step_plus_event_stability_negative_controls.csv
outputs/step_plus_event_stability_feature_ablation.csv
outputs/step_plus_event_stability_endpoint_grid.csv
outputs/step_plus_event_stability_endpoint_selected.csv
outputs/step_plus_event_stability_block_bootstrap.csv
```
