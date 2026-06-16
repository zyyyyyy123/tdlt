# Attempt 009: Sujianlin EMA Memory Audit

- Idea: `reference/sujianlin.pdf` highlights the sliding-average view of
  optimizers: Adam/EMA style states make the effective update depend on
  historical learning-rate mass, not only the current learning rate. This audit
  tests whether that history-memory view explains the residual left after the
  momentum baseline.
- Protocol: read `results/momentum_residual_mlp_results/predictions.csv`; keep
  `1000 <= step <= 33906` with every-2-step sampling; define
  `residual = log(loss) - log(momentum_s2)`; train residual ridge models on
  `cosine`; choose candidates by `811` full-trajectory MAE; report WSD only
  after selection. Features are schedule-only: current LR controls, normalized
  `s1/s2`, and EMA/history features of LR, LR^2, positive LR drops, and absolute
  LR changes across half-lives `[64, 256, 1024, 4096, 8192, 16384]`. No WSD loss history or
  scheduler label is used as a model input.
- Selection rule: global grid winner and best EMA-only winner are both selected
  by `811` full MAE. The step spline is included only as a reference, also
  selected on `811`, with config `spline_s0.01_shrink1_clip0.15`.

## Selected Candidates

- Global grid winner: `current_lr_ewma_lr2_h1024_ridge0p01_shrink1p25_clip0p05`
  (`ema_memory`, features
  `lr_norm, ewma_lr2_h1024`), 811 full MAE
  `0.034792`.
- Best EMA-only winner: `current_lr_ewma_lr2_h1024_ridge0p01_shrink1p25_clip0p05` (features
  `lr_norm, ewma_lr2_h1024`), 811 full MAE
  `0.034792`.

Top 5 validation candidates:

| validation_full_rank | candidate_id | feature_group | feature_set | validation_full_mae | validation_tail_27126_33906_mae | train_full_mae |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | current_lr_ewma_lr2_h1024_ridge0p01_shrink1p25_clip0p05 | ema_memory | current_lr_ewma_lr2 | 0.034792 | 0.033359 | 0.032345 |
| 2 | current_lr_ewma_lr2_h1024_ridge0p01_shrink1p25_clip0p1 | ema_memory | current_lr_ewma_lr2 | 0.034792 | 0.033359 | 0.032345 |
| 3 | current_lr_ewma_lr2_h1024_ridge0p01_shrink1p25_clip0p15 | ema_memory | current_lr_ewma_lr2 | 0.034792 | 0.033359 | 0.032345 |
| 4 | current_lr_ewma_lr2_h1024_ridge0p01_shrink1p25_clip0p25 | ema_memory | current_lr_ewma_lr2 | 0.034792 | 0.033359 | 0.032345 |
| 5 | current_lr_ewma_lr2_h1024_ridge0p1_shrink1p25_clip0p05 | ema_memory | current_lr_ewma_lr2 | 0.034792 | 0.033359 | 0.032345 |

## Key WSD Numbers

| model | full MAE | tail `27126-33906` MAE | improvement vs momentum | gap vs step |
|---|---:|---:|---:|---:|
| momentum baseline | 0.037720 | 0.038754 | 0.000000 | 0.016843 |
| global grid winner | 0.036327 | 0.035942 | 0.001393 | 0.015450 |
| best EMA-only | 0.036327 | 0.035942 | 0.001393 | 0.015450 |
| step spline reference | 0.020878 | 0.021855 | 0.016843 | 0.000000 |

## Conclusion

- The EMA/history grid improves the WSD full-trajectory momentum baseline under the frozen 811 selection rule.
- It does not approach the step-spline reference: the WSD full MAE gap is `0.015450` for the global winner and `0.015450` for the best EMA-only model.
- Failure mode: the selected low-dimensional schedule-memory features are too
  coarse to recover the step-aligned residual phase found in the spline audits.
  They can encode smoothed LR history and drop memory, but not the sharper
  absolute-step residual template that transfers from cosine to WSD in this
  three-schedule dataset.

Key output files:

```text
outputs/sujianlin_ema_memory_candidate_grid.csv
outputs/sujianlin_ema_memory_selected_metrics.csv
outputs/sujianlin_ema_memory_feature_summary.csv
```
