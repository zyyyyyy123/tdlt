from pathlib import Path

import numpy as np
from src.config import FOLDER_PATHS, FILES
from src.data_loader import load_data


PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_data_loader():
    folder_path = PROJECT_DIR / FOLDER_PATHS["400"]
    data = load_data(str(folder_path))

    assert set(data) == set(FILES)
    for file_name in FILES:
        step = data[file_name]["step"]
        lrs = data[file_name]["lrs"]
        loss = data[file_name]["loss"]

        assert len(step) > 0
        assert len(step) == len(loss)
        assert len(lrs) >= int(step.max())
        assert np.all(np.isfinite(loss))
        assert np.all(np.isfinite(lrs))
        assert np.all(lrs >= 0)

if __name__ == "__main__":
    test_data_loader()
