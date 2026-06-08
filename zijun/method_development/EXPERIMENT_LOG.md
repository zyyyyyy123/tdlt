# Experiment Log

This file is now an index. Detailed records are split under
`experiment_logs/` so future reads can load only the relevant attempt.

## Protocol Reminder

- Main task: predict full LLM pretraining loss curves across learning-rate
  schedules.
- Default split: fit on `cosine`, use `811` only as validation/auxiliary when
  available, and report `wsd` as the main transfer target.
- Default evaluation: sampled points with `step >= 1000`, plus the hard
  `20000-30000` window.
- Every nontrivial experiment should get its own Markdown file in
  `experiment_logs/` and one concise index row below.

## Attempt Index

| id | file | role | current status |
|---|---|---|---|
| 000 | [Feature Ridge Sanity Check](experiment_logs/attempt_000_feature_ridge.md) | direct feature baseline | Negative; useful only as plumbing check. |
| 001 | [High-Dimensional Residual Features](experiment_logs/attempt_001_high_dimensional_residual.md) | residual feature model | Negative; overfits cosine and transfers poorly. |
| 002 | [Smooth Residual Transfer](experiment_logs/attempt_002_smooth_residual_transfer.md) | main method | Best current direction; WSD MAE `0.037216 -> 0.020657`. |
| 003 | [Momentum Multi-End Baseline](experiment_logs/data_prep_003_momentum_multi_end.md) | data prep | Baseline assets for partial-trajectory checks. |
| 004 | [Roll5 Smooth Residual](experiment_logs/attempt_004_roll5_smooth_residual.md) | noise robustness | Full roll5 improves; hard-window result remains mixed. |
| 005 | [Intrinsic-Time Spline](experiment_logs/attempt_005_intrinsic_time_spline.md) | FSL-inspired coordinate test | Negative; `S1`-aligned residual transfer fails. |
| 006 | [Momentum-MPL Hybrid Decay](experiment_logs/attempt_006_momentum_mpl_hybrid_decay.md) | hybrid decay law | Small WSD improvement over momentum; not competitive with step spline. |
| 007 | [Spline Stability Audit](experiment_logs/attempt_007_spline_stability_audit.md) | statistical/robustness audit | 811-selected spline remains strong on WSD; block CI and placebo controls support the descriptive claim. |

## Current Working Conclusions

- The strongest method remains baseline-informed smooth residual transfer over
  the reproduced momentum law.
- Constant residual shift, direct feature fitting, and high-dimensional residual
  features are not sufficient.
- Intrinsic-time replacement by `S1` alone is a useful negative result.
- Momentum-MPL hybrid decay has a weak signal, but the next version should be
  event-local rather than another global cumulative-decay feature model.
- The spline mainline is now backed by frozen 811 selection, WSD block bootstrap,
  negative controls, and LOSO transfer checks, but still cannot claim
  schedule-level statistical significance with only three schedules.

## Log Template

Use this template for new files under `experiment_logs/`:

```text
# Attempt XXX: Name

- Idea:
- Process:
- Protocol:
- Metrics:
- Compared to:
- Failure mode:
- Conclusion:
- Use in slides: main / ablation / appendix / omit

Key output files:
```
