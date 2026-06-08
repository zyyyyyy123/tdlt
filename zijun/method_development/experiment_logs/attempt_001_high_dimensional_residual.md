# Attempt 001: High-Dimensional Residual Features

- Idea: learn `log(loss) - log(base_prediction)` using step, cumulative LR,
  LR-squared cumulative sum, LR drop mass, and momentum drop mass.
- Process: implemented in `scripts/run_residual_correction.py` using locally
  generated baseline predictions.
- Conclusion: too aggressive. The feature residual model can overfit cosine
  residuals and often hurts WSD transfer. Keep this as a negative ablation.
