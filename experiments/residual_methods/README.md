# Residual Method Experiments

This directory contains the method-development part used in the final report:
baseline-calibrated residual modeling for cross-schedule loss-curve prediction.

## Role in the Report

The report pipeline is:

1. Fit the reproduced momentum-law baseline on cosine.
2. Model the post-momentum log residual with an absolute-step spline.
3. Add a low-dimensional event/tail leftover correction.
4. Select using 8-1-1 where applicable and report WSD as held-out transfer.

## Layout

```text
experiments/residual_methods/
├── scripts/          # runnable experiment and audit entry points
├── src/              # reusable data, metric, feature, and residual helpers
├── outputs/          # generated CSV metrics and selected predictions
├── figures/          # generated diagnostic and slide figures
├── experiment_logs/  # chronological experiment notes
└── baseline_results/ # older copied baseline diagnostics
```

## Main Commands

Run from the repository root.

```bash
python experiments/residual_methods/scripts/run_momentum_residual_spline.py
python experiments/residual_methods/scripts/run_spline_stability_audit.py --selected-only
python experiments/residual_methods/scripts/run_sujianlin_event_decay_ablation.py --selected-only
```

The selected-only commands regenerate the CSV rows checked by the final report
verifier. The complete robustness/candidate-grid audits are still available:

```bash
python experiments/residual_methods/scripts/run_spline_stability_audit.py
python experiments/residual_methods/scripts/run_sujianlin_event_decay_ablation.py
python experiments/residual_methods/scripts/run_step_plus_event_leftover_stability_audit.py
```

The first two selected-only commands produce the step-residual results. The
event/tail selected-only command produces the final WSD table used in the
slides.

## Inputs

Raw course data:

```text
data/raw/gpt_loss+lrs.pkl
```

Momentum baseline predictions:

```text
results/baselines/momentum/predictions.csv
```

Three-schedule momentum/MLP intermediate used by the event/tail audits:

```text
results/baselines/momentum_residual_mlp/predictions.csv
```

## Key Outputs

```text
experiments/residual_methods/outputs/momentum_residual_spline_metrics.csv
experiments/residual_methods/outputs/spline_stability_selected_metrics.csv
experiments/residual_methods/outputs/sujianlin_event_decay_selected_metrics_by_window.csv
experiments/residual_methods/outputs/step_plus_event_stability_summary.csv
```

Use the repository-level checker to verify headline numbers:

```bash
python scripts/verify_report_metrics.py
```
