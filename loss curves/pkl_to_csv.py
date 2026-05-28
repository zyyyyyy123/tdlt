from pathlib import Path

import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "gpt_loss+lrs.pkl"
    output_path = base_dir / "gpt_loss+lrs.csv"

    data = pd.read_pickle(input_path)

    if isinstance(data, pd.DataFrame):
        result = data
    elif isinstance(data, dict):
        frames = []
        for run_name, frame in data.items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    f"Expected DataFrame for key {run_name!r}, got {type(frame).__name__}"
                )
            frames.append(frame.assign(run=run_name))
        result = pd.concat(frames, ignore_index=True)
    else:
        raise TypeError(f"Expected DataFrame or dict, got {type(data).__name__}")

    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} rows to {output_path}")


if __name__ == "__main__":
    main()
