from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun the report mainline experiments.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-mpl", action="store_true", help="Skip the Multi-Power Law rerun.")
    parser.add_argument(
        "--refresh-mlp-input",
        action="store_true",
        help="Regenerate results/baselines/momentum_residual_mlp before event/tail experiments.",
    )
    parser.add_argument(
        "--full-audit",
        action="store_true",
        help="Also run the slower robustness audits: spline bootstrap/placebo/LOSO and event-leftover stability.",
    )
    parser.add_argument("--skip-verify", action="store_true")
    return parser.parse_args()


def run_step(label: str, command: list[str]) -> None:
    print(f"\n==> {label}")
    print(" ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def sync_presentation_figures() -> None:
    figure_names = [
        "momentum_residual_spline_contrast.png",
        "momentum_residual_spline_edge_zoom.png",
        "step_plus_event_leftover_contrast.png",
        "step_plus_event_leftover_edge_zoom.png",
    ]
    source_dir = REPO_ROOT / "experiments" / "residual_methods" / "figures"
    target_dir = REPO_ROOT / "results" / "presentation" / "figures"
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in figure_names:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, target_dir / name)
            copied.append(name)
    print(f"\n==> Synced presentation figures ({len(copied)}/{len(figure_names)})")


def main() -> None:
    args = parse_args()
    py = sys.executable

    run_step("Momentum-law baseline", [py, "scripts/reproduce_momentum.py"])

    if args.refresh_mlp_input:
        run_step("Three-schedule momentum/MLP intermediate", [py, "scripts/train_momentum_residual_mlp.py"])

    if not args.skip_mpl:
        run_step(
            "Multi-Power Law baseline",
            [py, "scripts/reproduce_multi_power_law.py", "--device", args.device],
        )

    run_step(
        "Step-aligned residual spline",
        [py, "experiments/residual_methods/scripts/run_momentum_residual_spline.py"],
    )
    spline_command = [py, "experiments/residual_methods/scripts/run_spline_stability_audit.py"]
    if not args.full_audit:
        spline_command.append("--selected-only")
    run_step("Spline stability audit" if args.full_audit else "811-selected spline metrics", spline_command)

    event_command = [py, "experiments/residual_methods/scripts/run_sujianlin_event_decay_ablation.py"]
    if not args.full_audit:
        event_command.append("--selected-only")
    run_step("Event/tail leftover ablation", event_command)

    if args.full_audit:
        run_step(
            "Event/tail leftover stability audit",
            [py, "experiments/residual_methods/scripts/run_step_plus_event_leftover_stability_audit.py"],
        )

    sync_presentation_figures()

    if not args.skip_verify:
        run_step("Verify report metrics", [py, "scripts/verify_report_metrics.py"])


if __name__ == "__main__":
    main()
