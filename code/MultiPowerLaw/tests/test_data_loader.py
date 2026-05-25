import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from src.config import FOLDER_PATHS, FILES
from src.data_loader import load_data

def test_data_loader():
    parser = argparse.ArgumentParser(description="Learning Rate Scheduler Fitting")
    parser.add_argument("--folder_path", "-f", type=str, default="400", choices=["25", "100", "400"],
                        help="Model size folder path")
    args = parser.parse_args()
    
    folder_path = FOLDER_PATHS[args.folder_path]
    fig_folder = f"./{args.folder_path}M/"
    os.makedirs(fig_folder, exist_ok=True)
    os.makedirs(f"{fig_folder}/lrs", exist_ok=True)
    os.makedirs(f"{fig_folder}/loss", exist_ok=True)
    
    # Load and visualize data
    data = load_data(folder_path)
    for file_name in FILES:
        step, lrs, loss = data[file_name]["step"], data[file_name]["lrs"], data[file_name]["loss"]
        file_id = file_name.split(".")[0]
        plt.plot(np.arange(len(lrs)), lrs, label=file_id)
        plt.legend()
        plt.savefig(f"{fig_folder}/lrs/{file_id}_lrs.png")
        plt.close()
        plt.plot(step, loss, label=file_id)
        plt.legend()
        plt.savefig(f"{fig_folder}/loss/{file_id}_loss.png")
        plt.close()

if __name__ == "__main__":
    test_data_loader()