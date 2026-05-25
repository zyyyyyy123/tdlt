import os
import numpy as np
from .lr_schedules import cosine_lrs, const_lrs, two_stage_lrs, wsd_lrs, wsdld_lrs
from .config import FILES

def load_data(folder_path: str) -> dict:
    """
    Load and preprocess data from CSV files.

    Args:
        folder_path (str): Path to the directory containing CSV files.

    Returns:
        dict: Dictionary with file names as keys and data (steps, loss, lrs) as values.
    """
    data = {}
    for file_name in FILES:
        # file_path = folder_path + file_name
        file_path = os.path.join(folder_path, file_name)
        file_data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
        data[file_name] = {
            "step": file_data[:, 0].astype(int),
            "loss": file_data[:, 2].astype(float),
        }
        
        # Truncate to 24000 steps if applicable
        if data[file_name]["step"].max() == 24000:
            mask = data[file_name]["step"] < 24000
            data[file_name]["step"] = data[file_name]["step"][mask]
            data[file_name]["loss"] = data[file_name]["loss"][mask]

        # Assign learning rate schedules
        if "cosine" in file_name:
            total = int(file_name.split("_")[1].split(".")[0])
            data[file_name]["lrs"] = cosine_lrs(2160, total, 3e-4, 3e-5, False)
        elif "constant" in file_name:
            total = int(file_name.split("_")[1].split(".")[0])
            data[file_name]["lrs"] = const_lrs(2160, total, 3e-4, False)
        elif "wsdcon" in file_name:
            total = 16000
            lr_b = int(file_name.split("_")[1].split(".")[0]) * 1e-5
            data[file_name]["lrs"] = two_stage_lrs(2160, total, 3e-4, lr_b, 8000, False)
        elif "wsdld" in file_name:
            data[file_name]["lrs"] = wsdld_lrs(2160, 24000, 20000, 3e-4, 3e-5, False)
        elif "wsd" in file_name:
            data[file_name]["lrs"] = wsd_lrs(2160, 24000, 20000, 3e-4, 3e-5, False)
        else:
            raise ValueError(f"Invalid learning rate type for file: {file_name}")
    
    return data