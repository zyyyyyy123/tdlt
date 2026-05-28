# Zijun Project Workspace

This folder contains Zijun's work for Task 2: predicting LLM pretraining loss
curves across learning-rate schedules.

## Current Navigation

```text
zijun/
├── ReadMe.md
├── agent.md
└── method_development/
    ├── README.md
    ├── EXPERIMENT_LOG.md
    ├── scripts/
    ├── src/
    ├── outputs/
    └── figures/
```

## Baseline Inputs

The reproduced momentum-law baseline from the shared repository is the current
main baseline:

```text
results/reproduction/momentum/predictions.csv
results/reproduction/momentum/metrics.csv
results/reproduction/momentum/summary.json
```

Its prediction file uses sampled evaluation points marked by `is_sampled`.
For comparable experiments, residual models should use `is_sampled == True`
and `1000 <= step <= 33906`.

## Current Main Experiment

Run from the repository root:

```bash
python zijun/method_development/scripts/run_momentum_residual_spline.py
```

This fits residual corrections on the cosine run and evaluates transfer to WSD:

```text
residual = log(loss) - log(momentum_prediction)
corrected_prediction = momentum_prediction * exp(predicted_residual)
```

The current best lightweight method is a smooth spline residual transfer:

```text
feature_set = spline_s0.1_shrink1
```

Current WSD sampled metrics:

```text
momentum baseline:       MAE 0.037216, RMSE 0.046672, R2 0.928180
smooth residual spline:  MAE 0.020657, RMSE 0.024735, R2 0.979827
```

For the `20000-30000` window:

```text
momentum baseline:       MAE 0.042608, RMSE 0.052675
smooth residual spline:  MAE 0.035471, RMSE 0.036293
```

## Tracked Outputs

Keep these small or high-value outputs in Git:

```text
zijun/method_development/outputs/momentum_residual_spline_metrics.csv
zijun/method_development/outputs/key_momentum_residual_predictions.csv
zijun/method_development/figures/momentum_residual_spline_full.png
zijun/method_development/figures/momentum_residual_spline_20000_30000.png
```

Large scratch prediction tables should stay ignored unless they are the single
key model sequence needed for slides or comparison.
