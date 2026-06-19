# Attempt 013: No-Lookahead Audit for Spline to Event Leftover

## Scope

This audit covers the active chain:

- momentum residual spline mainline;
- spline stability and step-vs-S1 mechanism checks;
- Sujianlin event-local decay ablation;
- final `step_plus_event_leftover` stability audit.

The main question was whether WSD loss can leak into fitting, validation
selection, or full-candidate grids before the WSD test report.

## Issues Found And Fixed

1. `run_sujianlin_event_decay_ablation.py` named an event residual clip inside
   `StepEventConfig`, but the step-plus-event branch only clipped the final
   combined residual. The selected model's event leftover never reached the
   clip (`max_abs_event_leftover = 0.011990835`, clip `0.25`), so headline
   metrics did not change. The implementation now clips the event leftover
   before adding it to the step template.
2. `run_step_plus_event_leftover_stability_audit.py` included extra transfer
   diagnostics with WSD inside the training schedule. These rows were not used
   in the conclusion, but they were removed to avoid any WSD-as-training rows
   in the final event-leftover stability outputs.
3. `sujianlin_event_decay_trials.csv` exposed WSD test metrics for every
   candidate. Selection was still validation-only, but the full candidate grid
   now keeps train/validation metrics only.
4. `step_plus_event_stability_endpoint_grid.csv` exposed WSD test metrics for
   every endpoint-selection candidate. The endpoint grid now keeps validation
   metrics only; WSD test endpoint metrics are written only for validation-
   selected configs in `step_plus_event_stability_endpoint_selected.csv`.

## Rerun Commands

```text
python -m py_compile zijun/method_development/scripts/run_sujianlin_event_decay_ablation.py
python -m py_compile zijun/method_development/scripts/run_step_plus_event_leftover_stability_audit.py
python zijun/method_development/scripts/run_sujianlin_event_decay_ablation.py
python zijun/method_development/scripts/run_step_plus_event_leftover_stability_audit.py
```

## Output Checks

- `sujianlin_event_decay_trials.csv` has no `test_*` columns.
- `step_plus_event_stability_endpoint_grid.csv` has no `test_*` columns.
- `step_plus_event_stability_endpoint_selected.csv` keeps selected-config
  `test_*` columns.
- `step_plus_event_stability_transfer_metrics.csv` has training schedules only
  `811`, `cosine`, and `cosine+811`; no row has WSD in the training schedule.
- Attempt 011 and Attempt 012 docs now state the no-lookahead reporting rule.

## Final WSD Test Results

Full-resolution Attempt 011, `811_full` selection, test schedule `wsd`, window
`full`:

| model | WSD full MAE | RMSE | R2 | endpoint abs diff |
|---|---:|---:|---:|---:|
| momentum_baseline | `0.037720` | `0.047275` | `0.926313` | `0.045668` |
| step_reference | `0.021281` | `0.025610` | `0.978375` | `0.004676` |
| step_plus_event_leftover | `0.011191` | `0.014137` | `0.993410` | `0.016615` |

The no-lookahead fixes did not change the selected configs or the final WSD
test metrics.

## Current Conclusion

The event-leftover chain is now clean under the audited protocol:

- fit on `cosine`;
- select event/step configs using `811`;
- report WSD only after selection;
- do not expose WSD test metrics in all-candidate selection grids.

The substantive conclusion remains unchanged: `step_plus_event_leftover`
improves WSD full/tail over the step reference, but endpoint behavior remains
a caveat and should be reported separately.
