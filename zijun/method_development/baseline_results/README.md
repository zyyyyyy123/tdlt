# Baseline Results

The shared repository now contains the reproduced momentum-law baseline in:

```text
results/reproduction/momentum/
```

Current files:

```text
metrics.csv
predictions.csv
summary.json
momentum_fit_prediction.png
momentum_s1_s2.png
```

For new residual experiments, prefer reading
`results/reproduction/momentum/predictions.csv` directly instead of copying it
here. The normalized loader is:

```text
zijun/method_development/src/baseline_io.py
```

This local `baseline_results/` folder should only contain small summaries or
parameters specific to Zijun's experiments. Large copied prediction tables should
remain ignored.
