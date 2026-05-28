# 深度学习理论选讲 final

本仓库用于整理课程 final 作业相关材料，包括题目说明、参考文献、loss curve 数据，以及两个参考实现方向的代码。

## 目录结构

```text
.
├── final.pdf
├── README.md
├── code/
│   ├── requirements.txt
│   ├── momentum.py
│   └── MultiPowerLaw/
├── zijun/
│   └── method_development/
├── loss curves/
│   ├── gpt_loss+lrs.pkl
│   ├── gpt_loss_lrs_all_runs.xlsx
│   └── Readme.txt
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
- `gpt_loss_lrs_all_runs.xlsx`：已转换出的表格版本，便于直接查看。

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

### `zijun/method_development/`

邓子钧的个人方法开发目录。该目录默认假设两个 baseline 复现结果已经完成，后续主要放新的 fitting / prediction 方法、实验入口、对比指标和图表。

## 环境配置

建议使用 conda 创建独立环境：

```bash
conda create -n tdlt python=3.14
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
