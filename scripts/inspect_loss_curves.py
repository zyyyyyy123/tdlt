from pathlib import Path

import pandas as pd


def format_steps(steps: list[int], limit: int = 20) -> str:
    if len(steps) <= limit:
        return ", ".join(str(step) for step in steps)

    head = ", ".join(str(step) for step in steps[:limit])
    return f"{head}, ... ({len(steps)} total)"


def report_missing_steps(name: str, frame: pd.DataFrame) -> int:
    if "step" not in frame.columns:
        raise KeyError(f"DataFrame {name!r} does not contain a 'step' column")

    steps = frame["step"].dropna()
    if steps.empty:
        print(f"[{name}] rows={len(frame)}, missing_steps=unknown, reason=no step values")
        return 1

    if not (steps == steps.astype(int)).all():
        raise ValueError(f"DataFrame {name!r} contains non-integer step values")

    unique_steps = set(steps.astype(int).tolist())
    min_step = min(unique_steps)
    max_step = max(unique_steps)
    expected_steps = set(range(min_step, max_step + 1))
    missing_steps = sorted(expected_steps - unique_steps)
    duplicate_count = int(frame["step"].duplicated().sum())

    print(
        f"[{name}] rows={len(frame)}, step_range={min_step}-{max_step}, "
        f"unique_steps={len(unique_steps)}, missing_steps={len(missing_steps)}, "
        f"duplicate_steps={duplicate_count}"
    )
    if missing_steps:
        print(f"missing step values: {format_steps(missing_steps)}")

    return len(missing_steps)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "data" / "raw" / "gpt_loss+lrs.pkl"

    data = pd.read_pickle(input_path)

    total_missing_steps = 0
    if isinstance(data, pd.DataFrame):
        total_missing_steps += report_missing_steps("gpt_loss+lrs", data)
    elif isinstance(data, dict):
        for name, frame in data.items():
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    f"Expected DataFrame for key {name!r}, got {type(frame).__name__}"
                )
            total_missing_steps += report_missing_steps(str(name), frame)
    else:
        raise TypeError(f"Expected DataFrame or dict, got {type(data).__name__}")

    if total_missing_steps:
        print(f"Found {total_missing_steps} missing step values.")
        return 1

    print("No missing step values found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
