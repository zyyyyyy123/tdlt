from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_PKL = Path(__file__).resolve().parents[1] / "loss curves" / "gpt_loss+lrs.pkl"
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "loss curves" / "gpt_loss_lrs_csv"


def safe_filename(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\s]+', "_", name.strip())
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unnamed"


def convert_pkl_to_csv(pkl_path: Path, out_dir: Path) -> list[Path]:
    data = pd.read_pickle(pkl_path)
    if not isinstance(data, dict):
        raise TypeError(f"期望 pkl 顶层是 dict，实际是 {type(data).__name__}")

    out_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []
    combined_frames: list[pd.DataFrame] = []

    for run_name, value in data.items():
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"{run_name} 对应对象不是 DataFrame，实际是 {type(value).__name__}")

        df = value.copy()
        csv_path = out_dir / f"{safe_filename(run_name)}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        output_files.append(csv_path)

        combined = df.copy()
        combined.insert(0, "run", run_name)
        combined_frames.append(combined)

    combined_path = out_dir / "gpt_loss_lrs_all_runs.csv"
    pd.concat(combined_frames, ignore_index=True).to_csv(
        combined_path,
        index=False,
        encoding="utf-8-sig",
    )
    output_files.append(combined_path)

    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(description="把 gpt_loss+lrs.pkl 转换为 CSV 文件。")
    parser.add_argument("--pkl", type=Path, default=DEFAULT_PKL, help=f"输入 pkl 路径，默认: {DEFAULT_PKL}")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help=f"输出目录，默认: {DEFAULT_OUT_DIR}")
    args = parser.parse_args()

    output_files = convert_pkl_to_csv(args.pkl, args.out_dir)
    print("已生成 CSV 文件:")
    for path in output_files:
        print(path)


if __name__ == "__main__":
    main()
