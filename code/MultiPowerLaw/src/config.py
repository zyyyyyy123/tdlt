# src/config.py
"""Constants and configurations for the Learning Rate Scheduler project."""

# Training dataset file names
TRAIN_SET = [
    "cosine_24000.csv",
    "constant_24000.csv",
    "wsdcon_9.csv",
]

# Test dataset file names
TEST_SET = [
    "constant_72000.csv",
    "cosine_72000.csv",
    "wsd_20000_24000.csv",
    "wsdld_20000_24000.csv",
    "wsdcon_3.csv",
    "wsdcon_18.csv",
]

# Combined list of all files for data loading
FILES = TRAIN_SET + TEST_SET

# Directory paths for different model sizes
FOLDER_PATHS = {
    "25": "./loss_curve_repo/csv_25",
    "100": "./loss_curve_repo/csv_100",
    "400": "./loss_curve_repo/csv_400",
}

# Precomputed best parameters for each model size [L0, A, alpha, B, C, beta, gamma]
PARAMS = {
    "25": [3.04045406, 0.52468604, 0.50786857, 363.78751622, 2.06560812, 0.58279013, 0.64142257],
    "100": [2.6514477, 0.60115152, 0.45295811, 437.9464276, 2.13245612, 0.59785199, 0.65523644],
    "400": [2.37474466, 0.65421216, 0.42878731, 523.42464371, 2.02462735, 0.59350493, 0.63472457],
}

# Directory for saving optimized LR schedules
OPT_PATH = "./optimized_schedules/"

# Huber loss delta parameter (controls transition from quadratic to linear loss)
HUBER_DELTA = 0.001

# Default hyperparameters for optimization
OPT_TOTAL_STEPS = 24000  # Total steps in the LR schedule
OPT_PEAK_LR = 3e-4       # Peak learning rate
OPT_MIN_LR = 1e-10       # Minimum learning rate threshold
OPT_LR = 5e-9            # Learning rate for Adam optimizer in optimization
OPT_MAX_STEPS = 10000    # Maximum optimization steps
OPT_WARMUP = 2160        # Warmup steps for LR schedule
OPT_INTERVAL = 1000      # Logging interval during optimization

# Default hyperparameters for fitting
FIT_EVAL_INTERVAL = 5    # Steps between evaluations during fitting
FIT_LR1 = 5e-2           # Learning rate for L0, A, B, C parameters
FIT_LR2 = 5e-3           # Learning rate for alpha, beta, gamma parameters
FIT_MAX_STEPS = 200      # Maximum fitting steps
FIT_GRAD_NORM_THR = 1e-5 # Gradient norm threshold for stopping training
FIT_LOSS_THR = 0.0      # Loss improvement threshold for stopping training
FIT_PATIENCE = 20       # Steps without improvement before stopping training