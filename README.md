# TDLT Task 2: Loss-Curve Prediction from Learning-Rate Schedules

本仓库对应课程 Task 2：在 cosine 学习率曲线上拟合，并预测 held-out WSD 学习率曲线下的 LLM pretraining loss curve。最终报告见 `results/presentation/slide.pdf`。

## Project Layout

```text
.
├── data/
│   └── raw/                         # 课程给定 loss/lr 曲线和清洗脚本
├── scripts/                         # 老师复现实验时优先运行这里
│   ├── reproduce_report_results.py   # 一键重跑报告主线结果
│   ├── verify_report_metrics.py      # 校验重跑指标是否对齐报告数值
│   ├── reproduce_momentum.py         # Tissue et al. momentum-law baseline
│   ├── reproduce_multi_power_law.py  # Luo et al. Multi-Power Law baseline
│   └── build_three_schedule_momentum_table.py
├── experiments/
│   └── residual_methods/             # residual spline 和 event/tail leftover 实验
├── results/
│   ├── baselines/                    # baseline 复现输出
│   ├── intermediates/                # 主线复现需要的中间表
│   ├── diagnostics/                  # 辅助诊断结果
│   └── presentation/                 # slide.tex, slide.pdf, references.bib, figures/
├── src/
│   └── baselines/                    # baseline 参考实现代码
├── docs/
│   ├── assignment/                   # 课程说明文件
│   ├── literature/                   # 参考论文
│   └── research_notes/               # 文献和实验过程笔记
└── external/
    └── MultiPowerLaw/                # vendored external reference implementation
```

## Environment

Python 3.11 is recommended.

```bash
pip install -r requirements.txt
```

## One-Command Reproduction

From the repository root:

```bash
python scripts/reproduce_report_results.py --device cpu
```

This reruns the report mainline in order:

1. Momentum-law baseline.
2. Multi-Power Law baseline.
3. Step-aligned residual spline.
4. 8-1-1-selected spline metrics used in the report checker.
5. Sujianlin-inspired event/tail leftover selected metrics.
6. Metric verification against the report headline numbers.

The default command regenerates the selected report tables, not every
bootstrap/placebo/LOSO robustness grid. To rerun those heavier audits too:

```bash
python scripts/reproduce_report_results.py --device cpu --full-audit
```

The event/tail scripts read a three-schedule momentum-baseline table in
`results/intermediates/three_schedule_momentum/predictions.csv`. This table
contains true loss, LR features, and the fitted momentum-law prediction
`momentum_s2` for cosine, 8-1-1, and WSD. To regenerate that intermediate table:

```bash
python scripts/reproduce_report_results.py --device cpu --refresh-momentum-table
```

For exact report reproduction, this table is rebuilt from the report-locked
momentum calibration parameters. To refit that calibration from scratch for
diagnosis, run `python scripts/build_three_schedule_momentum_table.py --refit`;
small downstream metric drift is possible because the momentum objective is
nearly flat around the optimum.

If compute time is tight, skip the Multi-Power Law rerun and verify the rest:

```bash
python scripts/reproduce_report_results.py --skip-mpl
```

The final line should be:

```text
All report headline metrics match expected values.
```

## Individual Commands

Momentum law:

```bash
python scripts/reproduce_momentum.py
```

Output:

```text
results/baselines/momentum/
```

Multi-Power Law:

```bash
python scripts/reproduce_multi_power_law.py --device cpu
```

Output:

```text
results/baselines/multi_power_law/
```

Residual spline:

```bash
python experiments/residual_methods/scripts/run_momentum_residual_spline.py
python experiments/residual_methods/scripts/run_spline_stability_audit.py --selected-only
```

Step + event/tail leftover:

```bash
python experiments/residual_methods/scripts/run_sujianlin_event_decay_ablation.py --selected-only
```

Full robustness audits:

```bash
python experiments/residual_methods/scripts/run_spline_stability_audit.py
python experiments/residual_methods/scripts/run_sujianlin_event_decay_ablation.py
python experiments/residual_methods/scripts/run_step_plus_event_leftover_stability_audit.py
```

Verify report numbers:

```bash
python scripts/verify_report_metrics.py
```

## Expected Headline Metrics

The verification script checks the key values used in `results/presentation/slide.tex`:

```text
Momentum WSD MAE                  0.037844
Multi-Power Law WSD MAE           0.036027
Step spline selected WSD MAE      0.020878
Final step+event WSD full MAE     0.011191
Final step+event WSD full R2      0.993410
Final step+event WSD tail MAE     0.015937
```

Small floating-point differences are tolerated; large deviations cause `verify_report_metrics.py` to fail.

## Presentation

The final deck is self-contained under:

```text
results/presentation/
```

To compile:

```bash
cd results/presentation
xelatex slide.tex
bibtex slide
xelatex slide.tex
xelatex slide.tex
```
