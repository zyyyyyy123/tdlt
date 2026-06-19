# Sujianlin Task2 改进点综合

## 1. 目标与读法

本笔记只服务 Task2：预测 LLM pretraining full loss curve across
learning-rate schedules。`reference/sujianlin.pdf` 不是一个可直接复现的
单一模型，而是一组与 schedule/optimizer dynamics 有关的理论启发：

- 滑动平均/EMA 视角：AdamW/SGDM/RMSProp/Muon 等更新受历史状态影响，不能只看当前学习率。
- Batch Size 与 LR/noise scale：有效训练噪声与 `lr^2`、batch size、训练步数相关。
- Adam epsilon/update RMS：Adam 型优化器的有效步长受 epsilon 和 update RMS 影响。
- AdamW weight RMS/动态 weight decay：LR schedule 与 weight decay 共同决定权重尺度。
- SGD 平均损失/终点损失：平均 loss 与 endpoint loss 可能由不同 schedule 机制控制。
- 线性/decay 策略：linear decay、Cosine、WSD 的 tail/endpoint 行为应单独看。

当前基线背景来自 `current_progress.md` 和 attempt 008：momentum law 已用
`S1/S2` 解释主趋势，但 post-momentum residual 在当前三条曲线中更像
absolute-step aligned phase，而不是单变量 `S1` intrinsic time。下面的实验
A/B/C 是对 Sujianlin 启发点的 schedule-only 审计，不应解释为真实 optimizer
state 已被验证。

## 2. 理论点到实验的映射

| Sujianlin 启发点 | 已跑实验 | 当前结论 | 备注 |
|---|---|---|---|
| 滑动平均/EMA 视角 | A: EMA/history memory | 弱正向 | 小幅优于 momentum，但远弱于 step template。 |
| Batch Size 与 LR/noise scale | B: `sqrt_cum_lr2` 等 noise proxy | mixed-to-negative | 最好 proxy 略优于 baseline，但 support/phase mismatch 很大。 |
| Adam epsilon/update RMS | B: effective update / softsign proxy | mixed-to-negative | 没有真实 Adam state，只能说明 schedule-only proxy 不够。 |
| AdamW weight RMS/动态 WD | C: event-decay/leftover proxy | positive with caveat | event-only 有效；step+event leftover 的 full/tail MAE 稳定更好，但 endpoint 变差。 |
| SGD 平均/终点损失 | C: endpoint/tail 指标 | mixed | full/tail 与 endpoint 指标冲突，必须分开报告。 |
| 线性/decay 策略 | C: linear endpoint/event-local features | 正向但需限定 | 选中的 event family 更像 linear decay progress，而不是真实 WD state。 |

## 3. 实验 A：EMA/history memory

文件：

```text
zijun/method_development/scripts/run_sujianlin_ema_memory_audit.py
zijun/method_development/experiment_logs/attempt_009_sujianlin_ema_memory_audit.md
zijun/method_development/outputs/sujianlin_ema_memory_*.csv
```

运行：

```text
python zijun/method_development/scripts/run_sujianlin_ema_memory_audit.py
```

811 选中模型：

```text
current_lr_ewma_lr2_h1024_ridge0p01_shrink1p25_clip0p05
```

WSD 结果：

| model | full MAE | 27126-33906 MAE | 30000-33906 MAE | last-2048 MAE |
|---|---:|---:|---:|---:|
| momentum baseline | `0.037720` | `0.038754` | `0.033631` | `0.033819` |
| EMA memory | `0.036327` | `0.035942` | `0.032667` | `0.032721` |
| step spline reference | `0.020878` | `0.021855` | `0.008762` | `0.009557` |

判断：弱正向。EMA/history features 符合“历史学习率记忆”的理论直觉，并在
frozen 811 selection 下小幅改善 momentum baseline；但它没有恢复 attempt 008
看到的 step-aligned residual phase，距离 step reference 仍很远。

## 4. 实验 B：noise/update coordinate

文件：

```text
zijun/method_development/scripts/run_sujianlin_noise_update_coordinate_audit.py
zijun/method_development/experiment_logs/attempt_010_sujianlin_noise_update_coordinate_audit.md
zijun/method_development/outputs/sujianlin_noise_update_*.csv
```

运行：

```text
python zijun/method_development/scripts/run_sujianlin_noise_update_coordinate_audit.py
```

WSD 结果：

| coordinate/model | full MAE | tail MAE | last-2048 MAE | residual corr | extra diagnostic |
|---|---:|---:|---:|---:|---|
| momentum baseline | `0.037720` | - | - | - | reference baseline |
| `step_abs` | `0.020767` | `0.021707` | `0.009310` | `0.936` | step phase control |
| best noise proxy: `sqrt_cum_lr2` | `0.036435` | `0.037196` | `0.033058` | `0.160` | WSD support outside `0.614` |
| best update-RMS proxy: `effective_update_hl4096_epslr2_1e-06` | `0.036811` | `0.037196` | `0.033058` | `0.098` | median abs warp `2529` steps |

判断：mixed-to-negative。`sqrt_cum_lr2` 和 update-RMS proxy 比 momentum
baseline 略好，但远弱于 `step_abs`，而且 residual correlation、support、
time-warp 诊断都不支持它们解释主 residual phase。这个结果不能否定真实
BatchSize/noise/update-RMS 理论；它只说明在当前数据中，用 LR schedule
构造的黑盒 proxy 不足以替代 step template。

## 5. 实验 C：event-decay/leftover

文件：

```text
zijun/method_development/scripts/run_sujianlin_event_decay_ablation.py
zijun/method_development/scripts/run_step_plus_event_leftover_stability_audit.py
zijun/method_development/experiment_logs/attempt_011_sujianlin_event_decay_ablation.md
zijun/method_development/experiment_logs/attempt_012_step_plus_event_leftover_stability_audit.md
zijun/method_development/experiment_logs/attempt_013_no_lookahead_event_leftover_audit.md
zijun/method_development/outputs/sujianlin_event_decay_*.csv
zijun/method_development/outputs/step_plus_event_stability_*.csv
```

运行：

```text
python zijun/method_development/scripts/run_sujianlin_event_decay_ablation.py
```

WSD 按 `811_full` 选择的结果：

| model | full MAE | full R2 | endpoint_abs_diff |
|---|---:|---:|---:|
| momentum_baseline | `0.037720` | `0.926313` | `0.045668` |
| event_decay_only | `0.032596` | `0.945083` | `0.050142` |
| step_reference | `0.021281` | `0.978375` | `0.004676` |
| step_plus_event_leftover | `0.011191` | `0.993410` | `0.016615` |

补充窗口：

| model | tail/full message | last-2048/endpoint message |
|---|---|---|
| event_decay_only | full MAE 优于 momentum | endpoint 比 momentum 更差 |
| step_plus_event_leftover | WSD full `0.021281 -> 0.011191`，tail `0.022268 -> 0.015937` | last-2048 `0.010247 -> 0.011531` 变差，endpoint `0.004676 -> 0.016615` 变差 |

判断：mixed-positive。event-local proxy 是三组 sujianlin-inspired 实验中
最有用的下一步候选：它在 step template 之外提供了 WSD full/tail 的边际信息。
但它不是 endpoint-loss win，也不是真实 weight decay 状态验证。当前选中的信号
主要像 linear decay progress / endpoint-tail coordinate，而不是可解释为
真实 AdamW weight RMS 或 WD state。

Attempt 012 对这个结论做了稳定性审计。它使用 4096-bin step spline
近似，不替代 Attempt 011 的 full-resolution headline 数字，但用于检验机制：

| check | result | interpretation |
|---|---:|---|
| train `cosine`, test `wsd` | step full `0.035007` -> step+event `0.029644` | full/tail gain 方向稳定。 |
| train `811`, test `wsd` | step full `0.029869`, step+event `0.030152` | 811 单独训练不能转移到 WSD。 |
| train `cosine+811`, test `wsd` | step full `0.031382` -> step+event `0.029322` | 811 作为额外训练源不破坏结论。 |
| negative controls | aligned `0.029644`; zero/shift/reverse `0.035032/0.035132/0.038455` | 收益依赖正确对齐的 event/tail 几何。 |
| feature ablation | `linear_endpoint` best; `drop_impulse` second `0.033357` | 主要信号是线性 tail/endpoint 坐标，不是 drop impulse memory。 |
| WSD block bootstrap | full q05 improvement `0.003307`, prob positive `1.000` | WSD 曲线内 full/tail 改进稳定。 |
| endpoint check | endpoint abs diff 仍约 `0.031491` | endpoint 问题没有被 validation guard 解决。 |

更新判断：`step_plus_event_leftover` 可以作为 step template 之后的低维
event/linear-tail 修正来讲；但不能讲成 endpoint-loss 改进，也不能讲成
真实 optimizer state 或 WD state 已被识别。

Attempt 013 做了 no-lookahead/output 审计：修正了 step+event 分支的 event
clip 语义，移除了最终稳定性输出中的 WSD-as-training 诊断行，并且不再在
全候选 grids 中写出 WSD test 指标。重跑后 `811_full` 选择的 WSD full
结果不变：momentum `0.037720`，step_reference `0.021281`，
step_plus_event_leftover `0.011191`。

## 6. 总结表

| 方向 | 结果标签 | 相对 momentum | 相对 step template | 是否建议继续 |
|---|---|---|---|---|
| EMA/history memory | 弱正向 | full MAE `0.037720 -> 0.036327` | 明显更差 | 低优先级，除非拿到真实 optimizer state。 |
| noise scale proxy | mixed-to-negative | full MAE `0.037720 -> 0.036435` | 明显更差，corr `0.160` | 暂不作为主线。 |
| update-RMS proxy | mixed-to-negative | full MAE `0.037720 -> 0.036811` | 明显更差，corr `0.098` | 暂不作为主线。 |
| event_decay_only | mixed-positive | full MAE `0.037720 -> 0.032596` | 不如 step，endpoint 更差 | 可作为附录/候选特征。 |
| step_plus_event_leftover | 正向但有 endpoint caveat | 明显优于 baseline | full/tail 优于 step，endpoint/last-2048 不如 step；稳定性审计支持 aligned linear-tail 机制 | 继续作为轻量修正主候选。 |

## 7. 下一步建议

1. 主报告仍以 momentum residual step spline 作为最稳主线。attempt 008 的
   phase-alignment 解释仍成立，Sujianlin proxy 没有推翻它。
2. `step_plus_event_leftover` 已经通过 alignment negative controls、
   feature-family ablation、811 augmentation 和 WSD within-curve bootstrap
   审计。下一步不是继续加高维特征，而是单独处理 endpoint：可以把 endpoint
   作为独立校准项，或把 full/tail 模型和 final-point 模型分开报告。
3. event-decay 方向的当前证据更支持 linear decay progress / endpoint-tail
   coordinate，而不是 event memory。若继续扩展，应保持低维并优先增加
   schedule-level 数据，而不是从三条曲线上训练更强模型。
4. EMA/history 和 noise/update proxy 目前只适合作为弱/负结果写入文档。
   若未来能拿到真实 gradient variance、Adam second moment、update RMS、
   weight RMS 或 weight decay state，再重新测试 Sujianlin 的 optimizer-state
   解释。
5. 汇报措辞应保持克制：这些实验支持“schedule history/tail geometry 有边际
   信息”，不支持“仅靠 Sujianlin proxy 已解释 residual 机制”。
