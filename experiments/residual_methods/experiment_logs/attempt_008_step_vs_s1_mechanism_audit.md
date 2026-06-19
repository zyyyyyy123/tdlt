# Attempt 008: Step vs S1 Mechanism Audit

- Idea: answer why the absolute-step residual template works better than an
  `S1` intrinsic-time residual template after the momentum-law baseline.
- Process: used three subagent critiques to turn the explanation into
  falsifiable diagnostics: no-leak selection, coordinate support, common
  support, residual phase alignment, time warp, smoothing stress test, and
  additive step/S1 ablation.
- Protocol: load `results/momentum_residual_mlp_results/predictions.csv`;
  define residual as `log(loss) - log(momentum_s2)`; train templates on
  `cosine`; use `811` only for family/config selection; report `wsd` held out.
  Sampled points are `1000 <= step <= 33906`, every 2 steps. The spline
  mechanism audit uses at most 4096 binned cosine support points with smoothing
  scaled by effective point count, so it should not be read as the final
  full-resolution step-spline performance.

## Key Results

Full-resolution coordinate interpolation selected on `811` gives the cleanest
phase test:

| coordinate | WSD full MAE | residual corr vs WSD residual | sign agreement |
|---|---:|---:|---:|
| absolute step | `0.020767` | `0.936` | `0.810` |
| `S1` raw | `0.045414` | `-0.123` | `0.382` |
| `S1` clamp | `0.045414` | `-0.123` | `0.382` |
| `S1` ratio | `0.039812` | `-0.008` | `0.500` |

This matches the full-resolution step-spline stability audit, where the
811-selected step spline reaches WSD full MAE `0.020878`, WSD tail
`27126-33906` MAE `0.021855`, and last-2048 MAE `0.009557`.

Binned spline mechanism audit selected on `811`:

| coordinate | selected config | WSD full MAE | WSD tail `27126-33906` MAE | verdict |
|---|---|---:|---:|---|
| absolute step | `s0.1_shrink1_clip0.15` | `0.035180` | `0.035798` | best held-out reference |
| `S1` clamp | `s0.5_shrink0.5_clip0.15` | `0.038881` | `0.040673` | negative transfer |
| `S1` ratio | `s0.5_shrink0.5_clip0.15` | `0.045606` | `0.038622` | negative transfer |
| `S1` raw | `s0.5_shrink0.5_clip0.15` | `0.240873` | `0.464558` | negative transfer |

Support and warp diagnostics:

| diagnostic | WSD result |
|---|---:|
| `S1` raw fraction outside cosine support | `0.464` |
| `S1` common-support WSD MAE, step | `0.031719` |
| `S1` common-support WSD MAE, raw S1 | `0.035162` |
| median absolute warp, step | `0` steps |
| median absolute warp, `S1` clamp | `3002` steps |
| median absolute warp, `S1` ratio | `3692` steps |

## Hypothesis Outcomes

| hypothesis | result | interpretation |
|---|---|---|
| H1: step advantage is not WSD selection leakage | supported | `811`-selected step templates beat `S1` variants on held-out WSD. |
| H2: raw `S1` fails partly because of support extrapolation | supported | WSD spends about `46.4%` of sampled points beyond the cosine `S1` support. |
| H3: extrapolation is not the whole story | supported | On the shared raw-`S1` support, step still beats raw `S1`; clamp and ratio also fail to recover the step result. |
| H4: transferable residual phase is step-aligned | supported | Full-resolution interpolation gives step residual corr `0.936`, while `S1` variants are near zero or negative. |
| H5: `S1` clamp/ratio induce phase warp | supported | `S1` coordinates map WSD points to cosine phases thousands of steps away. |
| H6: step advantage is not merely low-frequency smoothing | mixed | Heavy roll501 smoothing collapses templates toward baseline and can erase the step edge; the strong evidence is the raw residual phase alignment, not a generic low-frequency trend. |
| H7: step has larger marginal value than `S1` | supported | `step + S1 ratio` stays at `0.035180`, while `S1 ratio + step` remains `0.045118` in the conservative binned additive audit. |

## Current Explanation

The momentum law already uses `S1` and `S2` to explain the main learning-rate
progress trend. After that correction, the remaining transferable residual is
not primarily a one-dimensional `S1` functional law. It is a step-aligned
template: the same absolute training step maps to a similar residual phase
across these schedules, while the same `S1` value maps WSD to the wrong cosine
training phase. Raw `S1` also suffers from large WSD support extrapolation, but
clamping or ratio normalization only fixes scale/support; it does not restore
the residual phase.

Use in slides: main mechanism explanation, with H6 as an honest caveat.

Key output files:

```text
scripts/run_step_vs_s1_spline_mechanism_audit.py
outputs/step_vs_s1_hypothesis_summary.csv
outputs/step_vs_s1_interpolation_selection.csv
outputs/step_vs_s1_interpolation_alignment.csv
outputs/step_vs_s1_selection.csv
outputs/step_vs_s1_domain_support.csv
outputs/step_vs_s1_common_support.csv
outputs/step_vs_s1_time_warp.csv
outputs/step_vs_s1_smoothing_robustness.csv
outputs/step_vs_s1_additive_ablation.csv
```
