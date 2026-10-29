#!/usr/bin/env python3
"""
Orchestration script: run full ML pipeline, execute notebook, generate report.

This is the single command for the complete workflow.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def run_command(cmd, description, cwd=None, check=True):
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if check and result.returncode != 0:
        print(f"FAILED: {description}")
        sys.exit(result.returncode)

    print(f"SUCCESS: {description}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Build all research artifacts")
    parser.add_argument("--n-samples", type=int, default=10000,
                        help="Number of samples for pipeline (default: 10000)")
    parser.add_argument("--skip-pipeline", action="store_true",
                        help="Skip ML pipeline, use existing models/results")
    parser.add_argument("--skip-notebook", action="store_true",
                        help="Skip notebook execution")
    parser.add_argument("--skip-report", action="store_true",
                        help="Skip report generation")
    parser.add_argument("--output", type=str,
                        help="Custom output PDF path")
    args = parser.parse_args()

    print("=" * 60)
    print("QBER-ML Research Artifact Builder")
    print("=" * 60)

    # Step 1: Run ML pipeline (if not skipped)
    if not args.skip_pipeline:
        run_command([
            "uv", "run", "python", "scripts/run_pipeline.py",
            "--n-samples", str(args.n_samples)
        ], f"ML Pipeline (n={args.n_samples})")
    else:
        print("\nSkipping ML pipeline (using existing models/results)")

    # Step 2: Execute notebook (if not skipped)
    if not args.skip_notebook:
        run_command([
            "uv", "run", "python", "-m", "nbconvert",
            "--execute", "--to", "notebook",
            "--output", "notebooks/qad_experiments.ipynb",
            "--ExecutePreprocessor.timeout=600",
            "notebooks/qad_experiments.ipynb"
        ], "Notebook Execution")
    else:
        print("\nSkipping notebook execution")

    # Step 3: Generate report (if not skipped)
    if not args.skip_report:
        report_cmd = ["uv", "run", "python", "scripts/generate_report.py"]
        if args.output:
            report_cmd.extend(["--output", args.output])
        run_command(report_cmd, "Report Generation")
    else:
        print("\nSkipping report generation")

    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED SUCCESSFULLY")
    print("=" * 60)

    # Summary
    print("\nGenerated artifacts:")
    artifacts = [
        "models/*.joblib",
        "results/*.json",
        "results/*.png",
        "report/generated/main.pdf",
    ]
    for pattern in artifacts:
        for f in PROJECT_ROOT.glob(pattern):
            print(f"  {f.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()