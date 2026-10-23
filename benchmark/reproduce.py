"""Run the benchmark and verify all newly computed solutions."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "runs/unified")
    args = parser.parse_args()
    output = args.output.resolve()
    for script, flag in [
        ("run_unified.py", "--output"),
        ("summarize_unified.py", "--input"),
    ]:
        subprocess.run(
            [
                sys.executable,
                "-X",
                "utf8",
                str(root / "benchmark" / script),
                flag,
                str(output),
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(root / "benchmark/plot_report.py"),
            "--input",
            str(output),
            "--output",
            str(output / "figures"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
