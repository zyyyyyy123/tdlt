# 任务理解与实验协议

资料来源：本仓库 `final.pdf`、`README.md`、`loss curves/` 数据说明、`results/` 与 `zijun/method_development/` 实验输出、`reference/` 下论文。

## 1. 研究任务理解

本项目对应课程 final 的 Task 2：Predicting Loss Curve of LLM Pretraining。核心问题是研究 LLM 预训练中的 scaling laws，尤其是 learning-rate schedule, LRS, 如何影响完整训练过程中的 loss curve。

任务要求可以概括为三点：

1. 问题介绍：说明为什么 loss curve 预测重要，以及 LRS 为什么是 scaling law 中不可忽略的变量。
2. 实验复现：复现 Tissue et al. 的 learning-rate annealing momentum law 和 Luo et al. 的 Multi-Power Law，并按照课程要求在 cosine LRS 上拟合，在 WSD LRS 上评估。
3. 方法开发：分析 baseline 的局限，提出自己的 fitting 或 prediction 方法，并与复现 baseline 对比。

仓库当前聚焦的问题不是传统的只预测 final loss，而是预测每个 step 的训练 loss：

```text
L_t = L_Theta(eta_1, eta_2, ..., eta_t)
```

其中 `eta_t` 是第 `t` 步学习率。也就是说，模型输入不是一个标量训练步数，而是一段学习率历史。项目的关键难点在于：LRS 是高维函数输入，不同 schedule 的累计学习率、annealing 位置、decay 幅度和尾部形状都会影响 loss dynamics。

## 2. 数据与实验协议

### 数据

课程提供的数据在：

```text
loss curves/gpt_loss+lrs.pkl
loss curves/gpt_loss+lrs.csv
loss curves/gpt_loss_lrs_all_runs.xlsx
```

数据包含 3 条训练曲线，每条曲线是一个 `DataFrame`，字段包括：

```text
step
Metrics/loss
lr
```

3 条曲线对应：

| alias | run name |
|---|---|
| `cosine` | `M:100M_gpt_D:20B_scheduler:cosine_rope` |
| `wsd` | `M:100M_gpt_D:20B_scheduler:wsd_rope` |
| `811` | `M:100M_gpt_D:20B_scheduler:811_rope` |

数据已经清洗为连续 step `0..33907`，共 `33908` 个点。原始缺失点使用 forward fill 补齐：

| schedule | missing step | fill source |
|---|---:|---:|
| `wsd_rope` | `20815` | `20814` |
| `cosine_rope` | `22493` | `22492` |

### 主协议

当前主要协议为：

| 阶段 | 设置 |
|---|---|
| fit/train | cosine |
| validation/auxiliary | 811 |
| test/main transfer | wsd |
| evaluation start | step `1000` |
| 重点窗口 | full sampled trajectory 和 `20000-30000` |

这个协议符合课程要求：用 cosine LRS 拟合模型，再评估其对 WSD LRS 的预测能力。
