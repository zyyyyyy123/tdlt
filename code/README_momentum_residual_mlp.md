# Momentum Residual MLP 使用说明

这个脚本用于复现基于 momentum law 的 residual MLP：

- 脚本：`code/momentum_residual_mlp.py`
- 默认数据：`loss curves/gpt_loss+lrs.pkl`
- 默认输出：`results/momentum_residual_mlp_results/`
- 默认协议：`cosine` 训练，`811` 验证，`wsd` 测试

脚本只读取原始数据，不会修改 `loss curves/` 中的数据文件。

## 运行环境

在仓库根目录执行：

```powershell
conda activate tdlt
python -B code\momentum_residual_mlp.py
```

也可以不用手动 activate：

```powershell
conda run --no-capture-output -n tdlt python -B code\momentum_residual_mlp.py
```

`-B` 用于避免生成 `__pycache__`，不影响训练结果。

## 默认模型

默认恢复的是之前效果较好的 10 特征 MLP 版本：

| 参数 | 默认值 |
|---|---|
| baseline | `momentum_s2` |
| residual target | `log(loss) - log(momentum_s2)` |
| feature set | `kernel_summary` |
| m | `64` |
| hidden layers | `64,32` |
| activation | `relu` |
| scaler | `physics` |
| alpha/L2 | `0.001` |
| learning rate | `0.002` |
| seed | `3081` |
| shrink candidates | `0,0.75,1.0,1.25` |

10 个输入特征是：

```text
S1
kernel_ewm_diff_decay_0.9
kernel_ewm_diff_decay_0.99
kernel_ewm_diff_decay_0.995
kernel_ewm_diff_decay_0.999
kernel_diff_sum
kernel_abs_diff_sum
kernel_diff_max
kernel_diff_min
kernel_nonzero_diff_count
```

这些特征只使用累计学习率 `S1(t)` 和最近 `m=64` 步学习率一阶差分窗口的统计量，不使用 loss history、显式 scheduler label 或测试集信息作为 MLP 输入。

## 默认结果

在当前数据和默认设置下，脚本会用 `811` 验证集选择 shrink，然后在 `wsd` 上报告测试结果。已复现的默认结果为：

| model | WSD MAE | WSD R2 | endpoint_abs_diff |
|---|---:|---:|---:|
| momentum_s2 | `0.03784420` | `0.92566899` | `0.02226007` |
| momentum residual MLP | `0.03387169` | `0.94031545` | `0.00930743` |

MAE 相对 momentum baseline 提升约 `10.497%`。

## 输出文件

运行后会生成：

```text
results/reproduction/momentum_residual_mlp_results/
  summary.json
  README.md
  metrics.csv
  trials.csv
  predictions.csv
  fit.png
```

各文件含义：

| 文件 | 含义 |
|---|---|
| `summary.json` | 协议、baseline 参数、选中的 MLP 配置、核心测试指标 |
| `README.md` | 本次运行自动生成的简短结果说明 |
| `metrics.csv` | baseline 和各 residual trial 在 train/validation/test 上的指标 |
| `trials.csv` | seed/shrink 搜索表 |
| `predictions.csv` | 每个 step 的真实 loss、momentum 预测、MLP 修正预测 |
| `fit.png` | 拟合曲线和误差曲线 |

## 常用参数

指定输出目录：

```powershell
python -B code\momentum_residual_mlp.py --output-dir results\reproduction\my_residual_mlp_results
```

改最近差分窗口长度：

```powershell
python -B code\momentum_residual_mlp.py --m 256
```

改 MLP 结构：

```powershell
python -B code\momentum_residual_mlp.py --hidden 128,64 --activation relu
```

对比直接 residual 而不是 log residual：

```powershell
python -B code\momentum_residual_mlp.py --target residual --activation relu --hidden 32,32
```

尝试全历史 10 特征版本：

```powershell
python -B code\momentum_residual_mlp.py --feature-set full_history_10 --target log_residual --activation tanh --hidden 64,32 --alpha 0.0001 --learning-rate 0.001
```

## 注意事项

- 默认选择规则只看 `811` 验证集 MAE，`wsd` 测试集不参与选参。
- 如果 residual MLP 没有超过 momentum baseline，脚本会以非零状态退出，并提示查看 `trials.csv`。
- `predictions.csv` 体积较大，推送前可以根据仓库策略决定是否保留。
- 当前仓库里 `code/reproduction_momentum.py` 是原 baseline/复现实验文件；这个 README 对应的是新的 `momentum_residual_mlp.py`。
