# 当前研究进度

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

历史诊断窗口 WSD `20000-30000`：

| model | MAE | RMSE | MAPE | R2 |
|---|---:|---:|---:|---:|
| momentum baseline | `0.042608` | `0.052675` | `0.015505` | `-0.209586` |
| mean residual shift | `0.042004` | `0.052026` | `0.015284` | `-0.179938` |
| smooth residual spline | `0.035471` | `0.036293` | `0.012799` | `0.425794` |

结论：

- constant residual shift 几乎没有提升，说明收益不是简单 bias correction。
- smooth step-wise residual 结构能从 cosine 迁移到 WSD。
- 当前最强结果是 smooth residual spline，WSD full sampled MAE 达到 `0.020657`，R2 达到 `0.979827`。
- 机制审计进一步解释了为什么 absolute-step spline 比 `S1`
  intrinsic-time spline 好：full-resolution coordinate interpolation 中，
  WSD residual 与 cosine residual 按 step 对齐的相关性为 `0.936`，而
  `S1` raw/clamp 为 `-0.123`，`S1` ratio 为 `-0.008`。Raw `S1` 还有
  `46.4%` WSD sampled points 超出 cosine `S1` support；clamp/ratio 修掉
  support 问题后仍会把 WSD 映射到错误的 cosine residual phase。

### 3.5 已尝试但不是主线的方法

| attempt | idea | current conclusion |
|---|---|---|
| feature ridge sanity check | 用 schedule features 直接 ridge 拟合 loss | 只能检查流程，迁移到 WSD/811 很差 |
| high-dimensional residual features | 用 step、累计 LR、LR drop 等高维特征学习 residual | 容易过拟合 cosine residual，迁移不稳 |
| roll5 smooth residual target | 用 trailing 5 sampled-tick rolling mean loss 作为目标 | full roll5 指标有提升，但旧 `20000-30000` 诊断窗口表现 mixed |
| WSD partial residual forecast | 只看 WSD prefix 后预测未来 | 当前 residual prefix 修正明显不稳定，不适合作为主结论 |
| `S1` intrinsic-time residual spline | 用累计 LR 或 normalized `S1` 替代 absolute step | 负结果；momentum law 后的 residual 不是纯 `S1` 单变量结构，主要问题是 phase alignment 和 support mismatch |
