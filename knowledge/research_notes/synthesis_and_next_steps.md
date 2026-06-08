# 综合判断与下一步建议

## 5. 当前综合判断

1. 项目已经完成课程要求中的两个 baseline 复现：momentum law 和 Multi-Power Law。
2. 当前自研方法中，residual correction 是最有效主线。
3. smooth residual spline 是目前最强且最容易解释的方法：它利用 momentum law 解释主趋势，再利用 cosine residual 的平滑结构修正 WSD。
4. residual MLP 也有效，但可解释性弱，适合作为辅助对比。
5. 直接从 schedule features 拟合 loss 容易迁移失败，说明 baseline-informed correction 比从零拟合更稳。
6. WSD 中后期窗口仍是难点，尤其 `20000-30000` window 的 baseline R2 很低，说明局部 dynamics 还没有被 momentum law 完整捕捉。
7. smooth residual spline 已通过初步稳定性审计：811 冻结选参后在 WSD 仍显著改善，负对照 residual 模板失败，WSD paired block bootstrap 的 within-curve 改进为正。但由于只有三条 schedule，仍不能声称 schedule-level 统计显著泛化。

## 6. 下一步建议

### 主线一：完善 smooth residual spline 叙事

- 将方法表述为 baseline-informed residual transfer。
- 强调其只使用 cosine residual 拟合，不使用 WSD loss 选参。
- 对比 mean residual shift，证明提升来自 step-wise residual shape，而不是常数 bias。
- 在 slides 中展示 full trajectory 和 `20000-30000` window 两张图。
- 把结论措辞限定为当前三曲线数据中的稳定 descriptive improvement，而不是广义 schedule-level 置信结论。

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
