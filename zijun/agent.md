# Agent Notes

## Working Rules

- Work from the repository root: `/Users/john/Desktop/深度学习理论选讲/tdlt`.
- Keep new method code under `zijun/method_development/`.
- Do not edit reproduced baseline code unless explicitly requested.
- Prefer using reproduced baseline outputs from `results/reproduction/` over
  refitting baseline models inside the method-development folder.
- Keep generated cache files and large scratch prediction tables out of Git.

## Experiment Protocol

- Fit residual models on cosine only.
- Evaluate on WSD as the main transfer target.
- Use sampled points from the momentum reproduction:
  `is_sampled == True` and `1000 <= step <= 33906`.
- Report both full sampled trajectory metrics and the `20000-30000` window.
- Log every nontrivial attempt in
  `zijun/method_development/EXPERIMENT_LOG.md` with idea, process, metrics, and
  conclusion.

## Large File Policy

Ignored by default:

```text
zijun/method_development/outputs/*_predictions.csv
zijun/method_development/baseline_results/*_predictions.csv
```

Allowed tracked exception:

```text
zijun/method_development/outputs/key_momentum_residual_predictions.csv
```

This file should contain only the key model prediction sequence, not every
ablation attempt.
