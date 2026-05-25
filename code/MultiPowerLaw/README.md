# Multi-Power Law Repository

This repository provides a framework for fitting, predicting, and optimizing learning rate (LR) schedules using Multi-Power Law (MPL) models. Designed for researchers and practitioners in machine learning optimization, it enables analysis of training loss dynamics and derivation of optimized LR schedules in large language models. This work supports research into efficient training strategies for large language models. For more details, check out our paper: [arXiv:2503.12811](https://arxiv.org/abs/2503.12811).  

## Results

The tables below present updated evaluation metrics and best parameters for the MPL model fitted to datasets for 25M, 100M, and 400M parameter models. Note that results in our associated paper may differ due to the computational cost of rerunning all experiments.

### Formulation

The Multi-Power Law (MPL) model is formulated as:

$$
L(t) = L_0 + A \cdot (S_1(t) + S_W)^{-\alpha} - LD(t)
$$ 

Where:
- $L(t)$: Predicted loss at step $t$.
- $S_1(t) = \sum_{\tau=1}^{t} \eta_{\tau}$: Cumulative sum of learning rates up to step $t$.
- $S_W$: Cumulative LR during warmup (fixed offset).
- $LD(t) = B \sum_{k=1}^{t} (\eta_{k-1} - \eta_k) \cdot G(\eta_k^{-\gamma} S_k(t))$: Loss drop term.
- $S_k(t) = \sum_{\tau=k}^{t} \eta_{\tau}$: Partial cumulative LR from step $k$ to $t$.
- $G(x) = 1 - (C x + 1)^{-\beta}$: Power function as a non-linear transformation.

### Evaluation Metrics

| Model | $R^2$   | MAE     | RMSE    | PredE   | WorstE  |
|-------|---------|---------|---------|---------|---------|
| 25M   | 0.9988  | 0.00376 | 0.00465 | 0.00110 | 0.00409 |
| 100M  | 0.9983  | 0.00435 | 0.00592 | 0.00142 | 0.00583 |
| 400M  | 0.9978  | 0.00484 | 0.00730 | 0.00168 | 0.00995 |

- **$R^2$**: Coefficient of Determination, measuring goodness of fit.
- **MAE**: Mean Absolute Error, average absolute prediction error.
- **RMSE**: Root Mean Squared Error, standard deviation of residuals.
- **PredE**: Average relative prediction error.
- **WorstE**: Maximum relative prediction error.

### Best Parameters

| Model | $L_0$ | $A$   | $\alpha$ | $B$      | $C$   | $\beta$ | $\gamma$ |
|-------|-------|-------|----------|----------|-------|---------|----------|
| 25M   | 3.040 | 0.525 | 0.508    | 363.788  | 2.066 | 0.583   | 0.641    |
| 100M  | 2.651 | 0.601 | 0.453    | 437.946  | 2.132 | 0.598   | 0.655    |
| 400M  | 2.375 | 0.654 | 0.429    | 523.425  | 2.025 | 0.594   | 0.635    |

- **Parameters**: Coefficients for the MPL model, optimized to minimize Huber loss.

## Features
- **LR Schedulers**: Supports cosine, constant, two-stage, WSD, and WSDLD schedules.
- **Optimization**: Derives optimized LR schedules with non-increasing constraints using fitted MPL models.
- **Evaluation**: Generates metrics (e.g., MSE, $R^2$, Huber loss) and visualizations comparing predicted vs. actual loss.
- **Testing**: Includes unit tests for reliability of core components.

## Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/thu-yao-01-luo/MultiPowerLaw.git
   cd MultiPowerLaw
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Required packages: `numpy`, `torch`, `scipy`, `matplotlib`, `tqdm`, `sklearn`.

3. (Optional) Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Running the Main Script
The `main.py` script executes the full pipeline: data loading, model fitting, evaluation, and LR schedule optimization. Run it with:
```bash
python -u main.py --folder_path 400
```
- `--folder_path` or `-f`: Model size (`25`, `100`, or `400`). Default: `400`.

For optimization only, use precomputed parameters from `config.py`:
```bash
python main.py --opt_only --folder_path 400
```
- `--opt_only` or `-o`: Runs optimization standalone.

For a easy start, we provide `run_all.sh` to sequentially run test and main scripts across model sizes:
```bash
bash run_all.sh
```

**Outputs**:
- **Fitted Model Evaluation**: Plots in `./<model_size>M/fit/` (e.g., `./400M/fit/cosine_24000_mplfit.png`).
- **Optimized LR Schedule**: Saved as `./optimized_schedules/<model_size>.npy` and plotted in `./optimized_schedules/<model_size>.png`.
- **Logs**: Training progress, metrics, and optimization details in `logs/<model_size>.log`.

### Running Tests
Unit tests are in `tests/`. Execute them from the root directory:
```bash
python -m tests.test_lr_schedulers
python -m tests.test_data_loader --folder_path 400
```
- Outputs LR schedule visualizations in `lrs.png`.

## Project Structure
```
MultiPowerLaw/
├── src/                # Core source code
│   ├── __init__.py     # Package marker
│   ├── config.py       # Constants and configurations
│   ├── data_loader.py  # Data loading and preprocessing
│   ├── lr_schedulers.py# LR scheduler implementations
│   ├── models.py       # MPL and MultiPower models
│   ├── fitting.py      # Model fitting logic
│   ├── evaluation.py   # Evaluation and plotting
│   ├── optimization.py # LR schedule optimization
│   └── utils.py        # Utility functions
├── tests/              # Unit tests
│   ├── __init__.py     # Package marker
│   ├── test_lr_schedulers.py
│   └── test_data_loader.py
├── logs/               # Log files
├── main.py             # Entry point
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

### Key Components
- **`config.py`**: Defines datasets, paths (e.g., `OPT_PATH`), and precomputed parameters.
- **`lr_schedulers.py`**: Implements LR schedules used in training data.
- **`models.py`**: Contains `MPL` (core model) and `MultiPower` (deprecated).
- **`fitting.py`**: Fits MPL to training data using AdamW with early stopping.
- **`optimization.py`**: Optimizes LR schedules with the fitted MPL model.
- **`evaluation.py`**: Provides metrics and visualizations.

## Data Requirements
- **Format**: CSV files with `step`, `lr`, `loss` columns (e.g., `0,0.0003,2.0`).
- **Location**: Specified in `FOLDER_PATHS` (e.g., `./csv_400/`).
- **Names**: Must match `TRAIN_SET` and `TEST_SET` in `config.py` (e.g., `cosine_24000.csv`).

## Customization
- **Add Schedulers**: Extend `lr_schedulers.py` and update `data_loader.py`.
- **Modify Models**: Adjust `models.py` for alternative formulations.
- **Tune Hyperparameters**: Edit `fitting.py` or `optimization.py` parameters via `main.py`.

## Contributing
1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature`.
3. Commit: `git commit -m "Add your feature"`.
4. Push: `git push origin feature/your-feature`.
5. Submit a pull request.

Include tests and documentation updates with contributions.

## License
MIT License (see `LICENSE` file).

## Acknowledgments
- Developed for deep learning optimization research.
- Built with PyTorch, NumPy, and other open-source tools.
- Optimization script is credited to [Kaifeng Lyu](https://github.com/vfleaking).

## Contact
For questions or issues, file a GitHub issue or email `luokr2002@outlook.com`.

Optimize your training with Multi-Power Law schedules!
