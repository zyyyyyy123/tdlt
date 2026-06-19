# 文献核心思想

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
