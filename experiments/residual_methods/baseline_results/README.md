# Baseline Results

The shared repository now contains the reproduced momentum-law baseline in:

```text
results/baselines/momentum/
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
`results/baselines/momentum/predictions.csv` directly instead of copying it
here. The normalized loader is:

```text
experiments/residual_methods/src/baseline_io.py
```

This local `baseline_results/` folder should only contain small summaries or
parameters specific to Zijun's experiments. Large copied prediction tables should
remain ignored.

## Momentum Multi-End Check

The cosine self-prediction check with multiple fitting endpoints is stored
locally in:

```text
experiments/residual_methods/baseline_results/momentum_multi_end_check/
```

Command:

```bash
python scripts/reproduce_momentum.py \
  --sample-interval 1 \
  --output-dir experiments/residual_methods/baseline_results/momentum_multi_end_check \
  --sample-end 2000,4000,6000,8000,10000,12000,14000,16000,18000,20000 \
  --eval-runs cosine
```

Generated CSV files:

```text
momentum_multi_end_check/metrics.csv
momentum_multi_end_check/predictions.csv
```

`predictions.csv` is kept local and ignored because it is about 44 MB. Use
`metrics.csv` and `summary.json` for quick checks, and regenerate the prediction
table when needed.
