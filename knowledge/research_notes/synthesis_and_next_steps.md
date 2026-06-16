# 综合判断与下一步建议

## 5. 当前综合判断

1. 项目已经完成课程要求中的两个 baseline 复现：momentum law 和 Multi-Power Law。
2. 当前自研方法中，residual correction 是最有效主线。
3. smooth residual spline 是目前最强且最容易解释的方法：它利用 momentum law 解释主趋势，再利用 cosine residual 的平滑结构修正 WSD。
4. residual MLP 也有效，但可解释性弱，适合作为辅助对比。
5. 直接从 schedule features 拟合 loss 容易迁移失败，说明 baseline-informed correction 比从零拟合更稳。
6. WSD 中后期和 tail dynamics 仍需单独检查，但不再把 `20000-30000` 作为主检测窗口；主报告优先看 full trajectory、tail/decay window 和 endpoint。
7. smooth residual spline 已通过初步稳定性审计：811 冻结选参后在 WSD 仍显著改善，负对照 residual 模板失败，WSD paired block bootstrap 的 within-curve 改进为正。但 `s=0.01` 虽由 811 选中，复杂度偏高；主报告更适合使用保守的 `s=0.1`，并把结论写成 spline family 稳定而非单个参数最优。由于只有三条 schedule，仍不能声称 schedule-level 统计显著泛化。
8. `S1` intrinsic-time residual spline 目前是明确负结果。机制审计显示，失败不只是 raw `S1` 外推；即使用 clamp/ratio 修正 support，`S1` 也会把 WSD residual 映射到错误的 cosine phase。当前更稳妥的解释是：momentum law 已经吸收了主要 `S1/S2` 进度项，剩余可迁移 residual 是 absolute-step aligned。

## 6. 下一步建议

### 主线一：完善 smooth residual spline 叙事

- 将方法表述为 baseline-informed residual transfer。
- 强调其只使用 cosine residual 拟合，不使用 WSD loss 选参。
- 对比 mean residual shift，证明提升来自 step-wise residual shape，而不是常数 bias。
- 在 slides 中展示 full trajectory、tail/decay window 和 endpoint behavior。
- 把结论措辞限定为当前三曲线数据中的稳定 descriptive improvement，而不是广义 schedule-level 置信结论。

### 主线二：引入 FSL 解释

- 用 FSL 的 intrinsic time 解释为什么原始 step 上的 residual 可能不是最自然坐标。
- 直接把 spline 自变量从 step 改为 `S1` 已经失败，不应继续作为主线。
- 下一步如果继续 FSL 方向，应研究 noise/convolution kernel 或 event-local decay features，而不是 `S1` single-coordinate replacement。
- 目标是解释为什么 post-momentum residual 会呈现 step-aligned phase，并判断它是训练时间/token-time 结构、LR history kernel，还是 measurement residual。

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
