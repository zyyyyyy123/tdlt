# Data Prep 003: Momentum Multi-End Cosine Baseline

- Idea: reproduce the momentum baseline in the setting where only the first
  `sample_end` points are visible, then evaluate the unseen future of the same
  cosine run.
- Process: ran `code/reproduction_momentum.py` with `sample_interval=1`,
  `eval_runs=cosine`, and `sample_end` values from `2000` to `20000` by `2000`.
- Output directory:
  `baseline_results/momentum_multi_end_check/`.
- Conclusion: this produces the baseline CSVs needed for the next residual
  correction task. The full prediction table is local-only because it is large;
  the command is documented in `baseline_results/README.md`.
