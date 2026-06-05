# 研究进展与文献核心思想笔记

更新时间：2026-06-05  
资料来源：本仓库 `final.pdf`、`README.md`、`loss curves/` 数据说明、`results/` 与 `zijun/method_development/` 实验输出、`reference/` 下 6 篇 PDF。

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

## 3. 当前研究进度

### 3.1 Momentum law baseline

实现位置：

```text
code/reproduction_momentum.py
results/reproduction/momentum/
```

模型来自 Tissue et al. 的 learning-rate annealing scaling law：

```text
L(s) = L0 + A * S1(s)^(-alpha) - C * S2(s)
```

其中：

| 符号 | 含义 |
|---|---|
| `S1` | 累计学习率，近似表示参数更新总量 |
| `S2` | 学习率下降的 momentum/annealing area |
| `L0, A, C, alpha` | 从 loss curve 中拟合的参数 |

当前复现设置为 cosine 拟合、cosine 和 WSD 评估。WSD 上的主要结果为：

| model | eval | MAE | R2 | endpoint_abs_diff |
|---|---|---:|---:|---:|
| momentum law | WSD | `0.037844` | `0.925669` | `0.022260` |

结论：momentum law 已经可以较好描述跨 schedule 的整体趋势，但残差仍然具有可迁移结构，尤其在部分中后期窗口仍有改进空间。

### 3.2 Multi-Power Law baseline

实现位置：

```text
code/reproduction_multi-power_law/main.py
results/reproduction/multi_power_law/
code/MultiPowerLaw/
```

Multi-Power Law, MPL, 采用更复杂的 schedule-aware 形式：

```text
L(t) = L0 + A * S1(t)^(-alpha) - LD(t)
```

其中 `LD(t)` 是学习率下降带来的额外 loss reduction，使用多个 decay event 的非线性 power-law 叠加表示。它比 momentum law 的线性 `S2` 项更灵活。

当前复现实验在 cosine 上拟合，在 WSD 上验证。WSD sampled metric 结果为：

| model | eval | MAE | RMSE | R2 | endpoint_abs_diff |
|---|---|---:|---:|---:|---:|
| Multi-Power Law | WSD | `0.036027` | `0.045796` | `0.935676` | `0.134094` |

与 sampled momentum baseline 对比：

| model | WSD MAE | WSD R2 | endpoint_abs_diff |
|---|---:|---:|---:|
| sampled momentum | `0.037216` | `0.928180` | `0.045104` |
| Multi-Power Law | `0.036027` | `0.935676` | `0.134094` |

结论：MPL 在平均误差和 R2 上略优于 momentum baseline，但 endpoint 误差明显更差。当前本地复现为了算力做了简化，例如采样间隔和训练步数少于论文设置，因此还不能直接等同于论文完整效果。

### 3.3 Momentum residual MLP

实现与结果：

```text
code/momentum_residual_mlp.py
results/momentum_residual_mlp_results/
```

方法思路是在 momentum law 基础上学习 residual：

```text
residual = log(loss) - log(momentum_prediction)
prediction = momentum_prediction * exp(predicted_residual)
```

默认协议：

| split | schedule |
|---|---|
| train | cosine |
| validation | 811 |
| test | wsd |

默认模型使用 10 个 schedule-derived 特征，包括 `S1` 和最近 `m=64` 步学习率差分窗口的统计量，不直接使用测试集 loss history 或 scheduler label。

WSD 测试结果：

| model | WSD MAE | WSD R2 | endpoint_abs_diff |
|---|---:|---:|---:|
| momentum_s2 | `0.037844` | `0.925669` | `0.022260` |
| momentum residual MLP | `0.033872` | `0.940315` | `0.009307` |

结论：residual MLP 在 held-out WSD 上相对 momentum baseline 有约 `10.497%` MAE 改进，并且 endpoint 误差也降低。它说明 residual correction 是有效方向，但 MLP 的可解释性弱于 spline residual。

### 3.4 Zijun smooth residual spline

实现与结果：

```text
zijun/method_development/scripts/run_momentum_residual_spline.py
zijun/method_development/outputs/momentum_residual_spline_metrics.csv
zijun/method_development/figures/momentum_residual_spline_full.png
zijun/method_development/figures/momentum_residual_spline_20000_30000.png
```

方法思路：

```text
residual = log(loss) - log(momentum_prediction)
corrected_prediction = momentum_prediction * exp(predicted_residual)
```

与 MLP 不同，这里只在 cosine sampled points 上拟合一维 smooth residual curve，然后把 residual 结构迁移到 WSD。当前关键模型为：

```text
feature_set = spline_s0.1_shrink1
```

WSD full sampled trajectory：

| model | MAE | RMSE | MAPE | R2 |
|---|---:|---:|---:|---:|
| momentum baseline | `0.037216` | `0.046672` | `0.013198` | `0.928180` |
| mean residual shift | `0.036862` | `0.046243` | `0.013068` | `0.929493` |
| smooth residual spline | `0.020657` | `0.024735` | `0.007360` | `0.979827` |

WSD `20000-30000` window：

| model | MAE | RMSE | MAPE | R2 |
|---|---:|---:|---:|---:|
| momentum baseline | `0.042608` | `0.052675` | `0.015505` | `-0.209586` |
| mean residual shift | `0.042004` | `0.052026` | `0.015284` | `-0.179938` |
| smooth residual spline | `0.035471` | `0.036293` | `0.012799` | `0.425794` |

结论：

- constant residual shift 几乎没有提升，说明收益不是简单 bias correction。
- smooth step-wise residual 结构能从 cosine 迁移到 WSD。
- 当前最强结果是 smooth residual spline，WSD full sampled MAE 达到 `0.020657`，R2 达到 `0.979827`。

### 3.5 已尝试但不是主线的方法

| attempt | idea | current conclusion |
|---|---|---|
| feature ridge sanity check | 用 schedule features 直接 ridge 拟合 loss | 只能检查流程，迁移到 WSD/811 很差 |
| high-dimensional residual features | 用 step、累计 LR、LR drop 等高维特征学习 residual | 容易过拟合 cosine residual，迁移不稳 |
| roll5 smooth residual target | 用 trailing 5 sampled-tick rolling mean loss 作为目标 | full roll5 指标有提升，但 `20000-30000` window 表现 mixed |
| WSD partial residual forecast | 只看 WSD prefix 后预测未来 | 当前 residual prefix 修正明显不稳定，不适合作为主结论 |

## 4. 文献核心思想

### 4.1 Scaling Laws for Neural Language Models

文件：

```text
reference/Scaling Laws for Neural Language Models.pdf
```

核心思想：

- 语言模型 cross-entropy loss 随模型参数量 `N`、数据量 `D`、训练 compute `C` 呈平滑 power-law 下降。
- 在合理范围内，模型宽深等架构细节影响弱于整体 scale。
- 大模型具有更高 sample efficiency；在固定 compute 下，compute-optimal training 倾向于训练更大的模型，并在未完全收敛时 early stop。
- 论文主要研究 final loss 和 compute allocation，而不是显式建模 LRS 对完整 loss curve 的影响。

与本项目关系：

- 提供 scaling law 的基本背景和幂律建模范式。
- 本项目是在它的基础上扩展：从 final loss 走向 full loss curve，从 `N/D/C` 标量输入走向 LRS 函数输入。

### 4.2 Training Compute-Optimal Large Language Models

文件：

```text
reference/Training Compute-Optimal Large Language Models.pdf
```

核心思想：

- Chinchilla 重新估计 compute-optimal frontier，认为很多当时的大模型过大且训练 token 不足。
- 在固定 compute budget 下，模型参数量和训练 tokens 应近似等比例扩展。
- 70B Chinchilla 用与 280B Gopher 相近的 compute，但训练更多 tokens，最终在大量 benchmark 上更好。

与本项目关系：

- 强调 scaling law 的实际价值是节省昂贵大规模训练的决策成本。
- 该文仍主要关注模型大小和数据量的最终性能分配；本项目进一步研究给定模型和 token budget 下如何通过 LRS 改变 loss curve。

### 4.3 Scaling Law with Learning Rate Annealing

文件：

```text
reference/Scaling law with learning rate.pdf
```

核心思想：

- 提出 full loss curve 的 learning-rate-aware scaling law：

```text
L(s) = L0 + A * S1^(-alpha) - C * S2
```

- `S1` 是 forward area，即累计学习率，刻画训练推进量。
- `S2` 是 annealing area，刻画学习率下降带来的额外 loss drop。
- 该公式能用一条或两条训练曲线拟合，并预测 unseen LRS 的完整 loss curve。
- 论文认为 loss 下降有两种来源：消耗更多训练步带来的 power-law 下降，以及 LR annealing 带来的额外下降。

与本项目关系：

- 是本项目第一个核心 baseline。
- 当前 residual correction 方法也直接建立在该模型之上：先用 momentum law 给出物理启发的主趋势，再学习残差。

### 4.4 A Multi-Power Law for Loss Curve Prediction Across Learning Rate Schedules

文件：

```text
reference/A multi-power law for loss curve prediction across learning rate schedules.pdf
```

核心思想：

- 提出 Multi-Power Law, MPL：

```text
L(t) = L0 + A * (S1(t) + SW)^(-alpha) - LD(t)
```

- 第一项仍是累计学习率驱动的 power law。
- `LD(t)` 用多个 power-law 项描述学习率下降事件产生的 loss reduction。
- 相比 momentum law 的线性 `S2`，MPL 对 decay shape 的刻画更细，尤其适合 discontinuous 或复杂 LRS。
- 论文通过拟合少量 LRS 后预测 unseen schedule，并用拟合公式优化 LRS，得到类似 WSD 但更优的 schedule。
- 论文也指出对高 peak LR、长 horizon、warmup 简化等设置仍有偏差。

与本项目关系：

- 是第二个核心 baseline。
- 它提示 residual 不应只看累计量，还应关注 decay event 的局部结构和非线性饱和效应。
- 当前 smooth residual spline 可以视作更轻量的经验修正；后续可把 MPL 的 decay event 特征加入 residual 模型。

### 4.5 Functional Scaling Laws in Kernel Regression: Loss Dynamics and Learning Rate Schedules

文件：

```text
reference/Functional Scaling Laws in Kernel Regression Loss Dynamics and Learning Rate Schedules.pdf
```

核心思想：

- 在 power-law kernel regression 中理论分析 SGD loss dynamics，提出 Functional Scaling Law, FSL。
- 关键概念是 intrinsic time：相比原始 step，它更准确表示训练进度。
- LRS 的影响通过 convolutional functional 进入 loss dynamics，包含 signal learning、noise accumulation 和 forgetting kernel。
- 理论比较 constant、exponential decay、WSD-like schedule，结论是 WSD 在 scaling efficiency 上最好，其次是 exponential decay，constant 最差。
- 论文还在 LLM 预训练实验中验证 FSL 能拟合并预测 cosine、WSD、8-1-1 等 schedule 的 loss curves。

与本项目关系：

- 为 momentum law 和 MPL 提供理论解释方向。
- 它支持一个重要判断：WSD 优势不是偶然经验现象，即便在二次/核回归近似中也能出现。
- 后续方法可以把 residual correction 从单纯 step spline 推向 intrinsic-time 或 convolution-kernel residual。

### 4.6 Configuration-to-Performance Scaling Law with Neural Ansatz

文件：

```text
reference/Configuration-to-Performance Scaling Law with Neural Ansatz.pdf
```

核心思想：

- 传统 scaling law 假设除 `N` 和 `D` 外的超参数都已接近最优，但现实中训练配置往往复杂且未充分调参。
- 论文提出 Configuration-to-Performance Scaling Law, CPL：从完整训练配置 `C` 映射到性能 `P`。
- 因为完整配置空间难以写成简单 closed-form，作者用 LLM/neural network 作为 neural ansatz，得到 NCPL。
- 输入包括模型结构、数据规模、optimizer、peak LR、LRS、batch size、weight decay、warmup ratio 等。
- NCPL 可预测 final loss，也可通过查询中间 step 预测完整 loss curve。

与本项目关系：

- 提供一个更数据驱动的扩展方向：当手工公式不足以覆盖复杂配置时，可以让神经网络学习 configuration-to-loss 的映射。
- 当前项目数据只有 3 条曲线，不适合直接训练大 NCPL；但 residual MLP 已经是一个小型 neural ansatz，可以作为轻量版本。

## 5. 当前综合判断

1. 项目已经完成课程要求中的两个 baseline 复现：momentum law 和 Multi-Power Law。
2. 当前自研方法中，residual correction 是最有效主线。
3. smooth residual spline 是目前最强且最容易解释的方法：它利用 momentum law 解释主趋势，再利用 cosine residual 的平滑结构修正 WSD。
4. residual MLP 也有效，但可解释性弱，适合作为辅助对比。
5. 直接从 schedule features 拟合 loss 容易迁移失败，说明 baseline-informed correction 比从零拟合更稳。
6. WSD 中后期窗口仍是难点，尤其 `20000-30000` window 的 baseline R2 很低，说明局部 dynamics 还没有被 momentum law 完整捕捉。

## 6. 下一步建议

### 主线一：完善 smooth residual spline 叙事

- 将方法表述为 baseline-informed residual transfer。
- 强调其只使用 cosine residual 拟合，不使用 WSD loss 选参。
- 对比 mean residual shift，证明提升来自 step-wise residual shape，而不是常数 bias。
- 在 slides 中展示 full trajectory 和 `20000-30000` window 两张图。

### 主线二：引入 FSL 解释

- 用 FSL 的 intrinsic time 解释为什么原始 step 上的 residual 可能不是最自然坐标。
- 尝试把 spline 自变量从 step 改为 `S1`、intrinsic time 或与 LRS convolution 相关的坐标。
- 目标是让 residual spline 从经验修正变成更有理论动机的 functional correction。

### 主线三：轻量 schedule-aware residual 特征

- 从 MPL 中提取 decay event 特征，例如 LR drop mass、tail cumulative LR、局部 decay saturation。
- 不直接训练过强模型，避免高维 residual features 的过拟合问题。
- 可以优先尝试 ridge/spline/low-dimensional GAM 风格模型，而不是大 MLP。

### 主线四：明确负结果

- feature ridge 和 high-dimensional residual correction 应作为负结果保留。
- partial WSD prefix forecast 当前明显不稳定，不建议放在主结论中。
- roll5 方法可作为噪声鲁棒性探索，但不应替代 raw-loss 主指标。

## 7. 可用于汇报的一句话总结

本项目复现了 learning-rate-aware scaling law 的两个代表性 baseline，并发现最有效的改进不是重新从 LRS 直接拟合 loss，而是在 momentum law 的物理主趋势上学习可迁移 residual；当前 smooth residual spline 在 WSD full sampled trajectory 上将 MAE 从 `0.037216` 降至 `0.020657`，R2 从 `0.928180` 提升至 `0.979827`。
