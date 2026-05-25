from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PKL = Path(__file__).resolve().parents[1] / "loss curves" / "gpt_loss+lrs.pkl"


def _format_float(value: float) -> str:
    return f"{value:.12g}"


def _summarize_dataframe(name: str, df: pd.DataFrame) -> dict[str, Any]:
    if "step" not in df.columns:
        raise ValueError(f"{name} 缺少 step 列，实际列为: {list(df.columns)}")

    step = df["step"]
    min_step = int(step.min())
    max_step = int(step.max())
    expected_steps = set(range(min_step, max_step + 1))
    actual_steps = set(map(int, step.dropna().tolist()))
    missing_steps = sorted(expected_steps - actual_steps)

    numeric_summary: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col]
            numeric_summary[col] = {
                "count": int(series.count()),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
                "min": float(series.min()),
                "max": float(series.max()),
                "first": float(series.iloc[0]),
                "last": float(series.iloc[-1]),
            }

    return {
        "name": name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "step_min": min_step,
        "step_max": max_step,
        "unique_steps": int(step.nunique(dropna=True)),
        "duplicated_steps": int(step.duplicated().sum()),
        "missing_steps": missing_steps,
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "numeric_summary": numeric_summary,
    }


def analyze_pkl(path: Path) -> dict[str, Any]:
    data = pd.read_pickle(path)
    if not isinstance(data, dict):
        raise TypeError(f"期望 pkl 顶层为 dict，实际为 {type(data).__name__}")

    run_summaries = []
    all_steps = []
    for name, value in data.items():
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"{name} 对应对象不是 DataFrame，实际为 {type(value).__name__}")
        run_summaries.append(_summarize_dataframe(name, value))
        all_steps.append(value["step"])

    step_sets = {
        name: set(map(int, value["step"].dropna().tolist()))
        for name, value in data.items()
    }
    union_steps = set().union(*step_sets.values()) if step_sets else set()
    common_steps = set.intersection(*step_sets.values()) if step_sets else set()

    return {
        "path": path,
        "top_type": type(data).__name__,
        "num_runs": len(data),
        "run_summaries": run_summaries,
        "total_rows": sum(item["rows"] for item in run_summaries),
        "sum_unique_steps_per_run": sum(item["unique_steps"] for item in run_summaries),
        "union_unique_steps": len(union_steps),
        "union_step_min": min(union_steps) if union_steps else None,
        "union_step_max": max(union_steps) if union_steps else None,
        "common_steps": len(common_steps),
        "common_step_min": min(common_steps) if common_steps else None,
        "common_step_max": max(common_steps) if common_steps else None,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"文件: {summary['path']}")
    print(f"顶层类型: {summary['top_type']}")
    print(f"曲线/实验数量: {summary['num_runs']}")
    print(f"跨所有曲线的记录总数: {summary['total_rows']}")
    print(f"各曲线 unique step 数量求和: {summary['sum_unique_steps_per_run']}")
    print(
        "所有曲线 step 取值并集: "
        f"{summary['union_unique_steps']} 个 "
        f"({summary['union_step_min']} 到 {summary['union_step_max']})"
    )
    print(
        "三条曲线共同拥有的 step: "
        f"{summary['common_steps']} 个 "
        f"({summary['common_step_min']} 到 {summary['common_step_max']})"
    )

    print("\n逐条曲线统计:")
    for item in summary["run_summaries"]:
        print(f"\n- {item['name']}")
        print(f"  行数: {item['rows']}")
        print(f"  列: {item['columns']}")
        print(f"  类型: {item['dtypes']}")
        print(
            "  step: "
            f"{item['step_min']} 到 {item['step_max']}, "
            f"unique={item['unique_steps']}, "
            f"重复={item['duplicated_steps']}, "
            f"缺失step数={len(item['missing_steps'])}"
        )
        if item["missing_steps"]:
            print(f"  缺失step: {item['missing_steps']}")
        print(f"  各列缺失值: {item['missing_values']}")

        for col in ("Metrics/loss", "lr"):
            if col not in item["numeric_summary"]:
                continue
            stat = item["numeric_summary"][col]
            print(
                f"  {col}: "
                f"count={stat['count']}, missing={stat['missing']}, "
                f"unique={stat['unique']}, "
                f"min={_format_float(stat['min'])}, "
                f"max={_format_float(stat['max'])}, "
                f"first={_format_float(stat['first'])}, "
                f"last={_format_float(stat['last'])}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用 pandas 解析 gpt_loss+lrs.pkl，并输出 step、loss、lr 等统计信息。"
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=DEFAULT_PKL,
        help=f"pkl 文件路径，默认: {DEFAULT_PKL}",
    )
    args = parser.parse_args()

    summary = analyze_pkl(args.pkl)
    print_summary(summary)


if __name__ == "__main__":
    main()
