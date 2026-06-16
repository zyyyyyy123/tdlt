# 研究进展与文献核心思想笔记

更新时间：2026-06-08  
资料来源：本仓库 `final.pdf`、`README.md`、`loss curves/` 数据说明、`results/` 与 `zijun/method_development/` 实验输出、`reference/` 下论文。

This file is now an index. Detailed notes are split under `research_notes/` so
future reads can load only the relevant section.

## Reading Index

| file | scope | when to read |
|---|---|---|
| [task_and_protocol.md](research_notes/task_and_protocol.md) | task framing, data, default split, metrics protocol | Start of any new research thread. |
| [current_progress.md](research_notes/current_progress.md) | baselines, residual MLP, smooth residual spline, non-main attempts | When comparing methods or writing results. |
| [literature_core_ideas.md](research_notes/literature_core_ideas.md) | scaling law, Chinchilla, momentum law, MPL, FSL, NCPL summaries | When designing theory-motivated methods or slides. |
| [synthesis_and_next_steps.md](research_notes/synthesis_and_next_steps.md) | current judgment, next-step themes, one-line summary | When deciding priorities or preparing the final report. |

## Current High-Level State

- Task: predict full LLM pretraining loss curves across learning-rate schedules.
- Main protocol: fit/train on `cosine`, use `811` as validation/auxiliary when
  available, report `wsd` as the main transfer target.
- Completed baselines: momentum law and Multi-Power Law.
- Current strongest method: momentum-law residual transfer with smooth step-wise
  spline; WSD full sampled MAE `0.037216 -> 0.020657`, R2
  `0.928180 -> 0.979827`.
- Spline robustness audit: the spline family remains strong on WSD under
  811-based selection, placebo residual controls fail, and paired WSD block
  bootstrap has positive within-curve improvement. The exact `s=0.01` selection
  is higher-complexity than necessary, so the conservative report setting should
  remain `s=0.1`; schedule-level significance is still limited by having only
  three schedules.
- Important caveat: direct schedule-feature fitting and high-dimensional
  residual fitting transfer poorly; baseline-informed correction is more stable.

## Maintenance Rule

- Keep this file short.
- Add durable knowledge to one of the `research_notes/*.md` files.
- Add per-experiment details to
  `zijun/method_development/experiment_logs/*.md`, then update
  `zijun/method_development/EXPERIMENT_LOG.md`.
