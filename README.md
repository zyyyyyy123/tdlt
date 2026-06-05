# 深度学习理论选讲 final

本仓库用于整理课程 final 作业相关材料，包括题目说明、参考文献、loss curve 数据，以及两个参考实现方向的代码。

## 目录结构

```text
.
├── .gitignore
├── final.pdf
├── README.md
├── code/
│   ├── requirements.txt
│   ├── momentum.py
│   ├── reproduction_momentum.py
│   ├── reproduction_multi-power_law/
│   │   ├── check.py
│   │   └── main.py
│   └── MultiPowerLaw/
│       ├── main.py
│       ├── requirements.txt
│       ├── src/
│       ├── tests/
│       └── optimized_schedules/
├── zijun/
│   └── method_development/
├── loss curves/
│   ├── gpt_loss+lrs.pkl
│   ├── gpt_loss+lrs.csv
│   ├── gpt_loss_lrs_all_runs.xlsx
│   ├── pkl_to_csv.py
│   └── Readme.txt
├── results/
│   └── reproduction/
│       ├── momentum/
│       │   ├── summary.json
│       │   ├── metrics.csv
│       │   ├── predictions.csv
│       │   └── momentum_fit_prediction.png
│       └── multi_power_law/
│           ├── summary.json
│           ├── metrics.csv
│           ├── predictions.csv
│           ├── training_history.csv
│           ├── mpl_fit_prediction.png
│           └── loss_monitor.png
└── reference/
    ├── A multi-power law for loss curve prediction across learning rate schedules.pdf
    ├── Configuration-to-Performance Scaling Law with Neural Ansatz.pdf
    ├── Functional Scaling Laws in Kernel Regression Loss Dynamics and Learning Rate Schedules.pdf
    └── Scaling law with learning rate.pdf
```

## 文件说明

### `final.pdf`

课程 final 作业说明文件。

### `reference/`

作业相关参考文献。当前包含 learning rate schedule、loss curve prediction、scaling law 等方向的论文材料。

### `loss curves/`

包含作业给定的 loss curve 数据和辅助查看脚本。

- `gpt_loss+lrs.pkl`：原始数据文件，使用 `pandas.read_pickle()` 读取。
- `gpt_loss+lrs.csv`：由 `gpt_loss+lrs.pkl` 转换得到的 CSV 文件，便于用表格工具或文本工具快速检查。
- `gpt_loss_lrs_all_runs.xlsx`：已转换出的表格版本，便于直接查看。
- `pkl_to_csv.py`：将 `gpt_loss+lrs.pkl` 转换为 `gpt_loss+lrs.csv` 的辅助脚本。
- `Readme.txt`：原始数据目录中的说明文件。

`gpt_loss+lrs.pkl` 顶层是一个 `dict`，包含 3 条训练曲线。每条曲线是一个 `DataFrame`，列为：

```text
step
Metrics/loss
lr
```

3 条曲线分别对应：

```text
M:100M_gpt_D:20B_scheduler:811_rope
M:100M_gpt_D:20B_scheduler:wsd_rope
M:100M_gpt_D:20B_scheduler:cosine_rope
```

### `code/`

代码目录。

- `requirements.txt`：当前代码需要的 Python 依赖。
- `momentum.py`：参考 `Scaling law with learning rate` 的基础实现代码，当前主要作为参考代码保留。
- `MultiPowerLaw/`：参考 `A multi-power law for loss curve prediction across learning rate schedules` 的实现代码和结果文件。
- `reproduction_multi-power_law/`: 论文 `A multi-power law for loss curve prediction across learning rate schedules` 的复现代码。

### `zijun/method_development/`

邓子钧的个人方法开发目录。该目录默认假设两个 baseline 复现结果已经完成，后续主要放新的 fitting / prediction 方法、实验入口、对比指标和图表。

## 环境配置

建议使用 conda 创建独立环境。当前仓库已在 Python 3.11 下验证：

```bash
conda create -n tdlt python=3.11
conda activate tdlt
pip install -r code/requirements.txt
```

`code/requirements.txt` 当前包含：

```text
numpy
torch
scipy
matplotlib
tqdm
scikit-learn
pandas
```

## 复现实验运行说明

### Momentum reproduction

运行 learning-rate momentum：

```bash
python code/reproduction_momentum.py
```

默认会读取：

```text
loss curves/gpt_loss+lrs.pkl
```

并将结果输出到：

```text
results/reproduction/momentum/
```

主要输出文件包括：

```text
results/reproduction/momentum/summary.json
results/reproduction/momentum/metrics.csv
results/reproduction/momentum/predictions.csv
results/reproduction/momentum/momentum_fit_prediction.png
```

### Multi-Power Law reproduction

运行 `A multi-power law for loss curve prediction across learning rate schedules` 的复现实验：

```bash
python code/reproduction_multi-power_law/main.py --device cuda
```

默认设置为：

- 读取 `loss curves/gpt_loss+lrs.pkl`
- 在 `cosine` loss curve 上拟合 Multi-Power Law 参数
- 在 `cosine` 和 `wsd` loss curve 上评估
- 读取 `results/reproduction/momentum/summary.json`，并在输出中给出和 momentum baseline 的指标对比

结果会输出到：

```text
results/reproduction/multi_power_law/
```

主要输出文件包括：

```text
results/reproduction/multi_power_law/summary.json
results/reproduction/multi_power_law/metrics.csv
results/reproduction/multi_power_law/predictions.csv
results/reproduction/multi_power_law/training_history.csv
results/reproduction/multi_power_law/mpl_fit_prediction.png
results/reproduction/multi_power_law/loss_monitor.png
```

如果希望避免覆盖已有结果，可以指定新的输出目录：

```bash
python code/reproduction_multi-power_law/main.py \
  --device cuda \
  --output-dir results/reproduction/multi_power_law_test
```

如果没有可用 GPU，也可以使用 CPU 运行：

```bash
python code/reproduction_multi-power_law/main.py --device cpu
```
