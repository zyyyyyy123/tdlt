# main.py
import os
import argparse
import logging
from src.config import FOLDER_PATHS, TRAIN_SET, TEST_SET, PARAMS
from src.data_loader import load_data
from src.fitting import initialize_params, generate_init_params, mpl_adam_fit
from src.evaluation import evaluate_mpl
from src.optimization import optimize_lr_schedule

def main():
    # Argument parsing with detailed help text
    parser = argparse.ArgumentParser(description="Fit and optimize learning rate schedules using MPL models.")
    parser.add_argument(
        "--folder_path", "-f",
        type=str,
        default="400",
        choices=["25", "100", "400"],
        help="Model size folder path (25M, 100M, or 400M parameters)."
    )
    parser.add_argument(
        "--opt_only", "-o",
        action="store_true",
        help="Skip fitting and evaluation, optimize LR schedule using precomputed parameters."
    )
    args = parser.parse_args()

    # Set up logging with custom format and overwrite mode
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f"{log_dir}/{args.folder_path}.log",
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",  # Custom format: INFO:src.module:message
        filemode="w"  # Overwrite log file for each new run
    )
    logger = logging.getLogger(__name__)

    folder_path = FOLDER_PATHS[args.folder_path]
    fig_folder = f"./{args.folder_path}M/fit/"
    os.makedirs(fig_folder, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {folder_path}")
    data = load_data(folder_path)

    if args.opt_only:
        best_params = PARAMS[args.folder_path]
        logger.info("Using precomputed parameters for optimization-only mode")
    else:
        # Fit model
        logger.info("Initializing parameters")
        init_param = initialize_params(data, TRAIN_SET)
        init_params = generate_init_params(init_param)
        logger.info("Starting MPL model fitting")
        best_params, best_loss = mpl_adam_fit(data, TRAIN_SET, TEST_SET, init_params, fig_folder)

        # Evaluate
        logger.info("Evaluating on training set")
        evaluate_mpl(data, TRAIN_SET, best_params, fig_folder)
        logger.info("Evaluating on test set")
        evaluate_mpl(data, TEST_SET, best_params, fig_folder)
        logger.info(f"Best Loss: {best_loss}")

    logger.info(f"Best Parameters: {best_params}")

    # Optimize LR schedule
    logger.info("Optimizing learning rate schedule")
    opt_eta = optimize_lr_schedule(best_params, name=args.folder_path)
    logger.info("Optimized Learning Rate Schedule:")
    logger.info(f"First 5: {opt_eta[:5]}, Last 5: {opt_eta[-5:]}")

if __name__ == "__main__":
    main()