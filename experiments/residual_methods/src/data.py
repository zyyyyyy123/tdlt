from pathlib import Path

import pandas as pd


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_data_path() -> Path:
    return repo_root() / "data" / "raw" / "gpt_loss+lrs.pkl"


def parse_schedule(run_name: str) -> str:
    raw = run_name.split("scheduler:", maxsplit=1)[-1].replace("_rope", "")
    if raw == "811":
        return "811"
    return raw


def load_loss_curves(path: str | Path | None = None) -> pd.DataFrame:
    data_path = Path(path) if path is not None else default_data_path()
    raw_curves = pd.read_pickle(data_path)

    frames = []
    for run_name, curve in raw_curves.items():
        frame = curve.copy()
        frame = frame.rename(columns={"Metrics/loss": "loss"})
        frame["run_name"] = run_name
        frame["schedule"] = parse_schedule(run_name)
        frames.append(frame[["run_name", "schedule", "step", "loss", "lr"]])

    return pd.concat(frames, ignore_index=True)
