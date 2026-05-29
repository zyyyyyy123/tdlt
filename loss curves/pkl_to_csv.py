from pathlib import Path

import pandas as pd


STEP_COLUMN = "step"
VALUE_COLUMNS = ("Metrics/loss", "lr")
STEP_START = 0
STEP_END = 33907
EXPECTED_STEPS = pd.RangeIndex(STEP_START, STEP_END + 1, name=STEP_COLUMN)


def clean_frame(run_name: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    required_columns = (STEP_COLUMN, *VALUE_COLUMNS)
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"{run_name}: missing required columns {missing_columns}")

    working = frame.copy()
    if working[STEP_COLUMN].isna().any():
        raise ValueError(f"{run_name}: step column contains NaN")

    working[STEP_COLUMN] = working[STEP_COLUMN].astype(int)
    duplicate_steps = working.loc[working[STEP_COLUMN].duplicated(), STEP_COLUMN].tolist()
    if duplicate_steps:
        raise ValueError(f"{run_name}: duplicate steps {duplicate_steps[:10]}")

    working = working.sort_values(STEP_COLUMN)
    if working[STEP_COLUMN].min() != STEP_START or working[STEP_COLUMN].max() != STEP_END:
        raise ValueError(
            f"{run_name}: expected step range {STEP_START}-{STEP_END}, "
            f"got {working[STEP_COLUMN].min()}-{working[STEP_COLUMN].max()}"
        )

    indexed = working.set_index(STEP_COLUMN)
    missing_steps = EXPECTED_STEPS.difference(indexed.index).astype(int).tolist()
    cleaned = indexed.reindex(EXPECTED_STEPS)
    cleaned.loc[:, VALUE_COLUMNS] = cleaned.loc[:, VALUE_COLUMNS].ffill()

    remaining_null_columns = [
        column for column in VALUE_COLUMNS if cleaned[column].isna().any()
    ]
    if remaining_null_columns:
        raise ValueError(f"{run_name}: remaining NaN columns {remaining_null_columns}")

    cleaned = cleaned.reset_index()
    cleaned[STEP_COLUMN] = cleaned[STEP_COLUMN].astype(int)
    return cleaned, missing_steps


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir / "gpt_loss+lrs.pkl"
    output_path = base_dir / "gpt_loss+lrs.csv"

    data = pd.read_pickle(input_path)

    if isinstance(data, pd.DataFrame):
        cleaned_frame, filled_steps = clean_frame("single_run", data)
        cleaned_data = cleaned_frame
        result = cleaned_frame
        filled_by_run = {"single_run": filled_steps}
    elif isinstance(data, dict):
        cleaned_data = {}
        frames = []
        filled_by_run = {}
        for run_name, frame in data.items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    f"Expected DataFrame for key {run_name!r}, got {type(frame).__name__}"
                )
            cleaned_frame, filled_steps = clean_frame(run_name, frame)
            cleaned_data[run_name] = cleaned_frame
            filled_by_run[run_name] = filled_steps
            frames.append(cleaned_frame.assign(run=run_name))
        result = pd.concat(frames, ignore_index=True)
    else:
        raise TypeError(f"Expected DataFrame or dict, got {type(data).__name__}")

    pd.to_pickle(cleaned_data, input_path)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} rows to {output_path}")
    for run_name, filled_steps in filled_by_run.items():
        if filled_steps:
            print(f"{run_name}: forward-filled missing steps {filled_steps}")


if __name__ == "__main__":
    main()
