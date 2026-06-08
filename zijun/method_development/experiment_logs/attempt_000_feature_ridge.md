# Attempt 000: Feature Ridge Sanity Check

- Idea: fit a direct ridge model on schedule features from cosine and predict
  other schedules.
- Process: implemented in `scripts/run_feature_fit.py`.
- Conclusion: useful only as a plumbing check. It fits cosine moderately but
  transfers poorly, so it should not be used as the final method.
