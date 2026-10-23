"""Plot runtime, iteration counts, and objective accuracy."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASE_PATTERN = re.compile(r"^M(?P<M>\d+)_N(?P<N>\d+)_")
SIZE_ORDER = [(3, 5), (5, 10), (10, 20)]
SIZE_LABELS = ["M=3, N=5", "M=5, N=10", "M=10, N=20"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=root / "results" / "runs.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results" / "figures" / "tableau_results_overview.png",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("result CSV contains no rows")
    required = {
        "case_id", "status", "runtime_seconds", "phase1_iterations",
        "phase2_iterations", "iterations", "primal_residual_inf",
        "relative_duality_gap", "relative_objective_error",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"result CSV is missing fields: {sorted(missing)}")
    return rows


def case_size(case_id: str) -> tuple[int, int]:
    match = CASE_PATTERN.match(case_id)
    if match is None:
        raise ValueError(f"unrecognized case id: {case_id}")
    return int(match.group("M")), int(match.group("N"))


def aggregate(rows: list[dict[str, str]]) -> dict[tuple[int, int], dict[str, list[float]]]:
    failed = [row for row in rows if row["status"] != "optimal"]
    if failed:
        raise ValueError(
            f"cannot plot successful-run comparison: {len(failed)} rows are not optimal"
        )

    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)

    result: dict[tuple[int, int], dict[str, list[float]]] = {
        size: defaultdict(list) for size in SIZE_ORDER
    }
    for case_id, case_rows in by_case.items():
        size = case_size(case_id)
        if size not in result:
            continue
        target = result[size]
        target["runtime_ms"].append(
            1000.0
            * statistics.median(float(row["runtime_seconds"]) for row in case_rows)
        )
        target["phase1"].append(float(case_rows[0]["phase1_iterations"]))
        target["phase2"].append(float(case_rows[0]["phase2_iterations"]))
        target["total"].append(float(case_rows[0]["iterations"]))
        target["primal"].append(
            max(float(row["primal_residual_inf"]) for row in case_rows)
        )
        target["objective"].append(
            max(float(row["relative_objective_error"]) for row in case_rows)
        )
        target["duality"].append(
            max(float(row["relative_duality_gap"]) for row in case_rows)
        )
    return result


def make_figure(
    data: dict[tuple[int, int], dict[str, list[float]]], output: Path, dpi: int
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#3c4653",
            "axes.labelcolor": "#20262e",
            "xtick.color": "#3c4653",
            "ytick.color": "#3c4653",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    positions = np.arange(len(SIZE_ORDER))

    runtime = [data[size]["runtime_ms"] for size in SIZE_ORDER]
    box = axes[0].boxplot(
        runtime,
        tick_labels=SIZE_LABELS,
        patch_artist=True,
        widths=0.58,
        medianprops={"color": "#17212b", "linewidth": 2},
        whiskerprops={"color": "#52606d"},
        capprops={"color": "#52606d"},
    )
    for patch, color in zip(box["boxes"], ("#4c9f70", "#e09f3e", "#3d5a80")):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Median runtime per case (ms, log scale)")
    axes[0].set_title("A. Runtime scaling")
    axes[0].grid(axis="y", which="both", alpha=0.22)

    phase1 = [statistics.median(data[size]["phase1"]) for size in SIZE_ORDER]
    phase2 = [statistics.median(data[size]["phase2"]) for size in SIZE_ORDER]
    total = [statistics.median(data[size]["total"]) for size in SIZE_ORDER]
    width = 0.24
    axes[1].bar(
        positions - width, phase1, width=width, color="#3d5a80", label="Phase I"
    )
    axes[1].bar(
        positions, phase2, width=width, color="#e09f3e", label="Phase II"
    )
    axes[1].bar(
        positions + width, total, width=width, color="#4c9f70", label="Total"
    )
    for x, value in zip(positions + width, total):
        axes[1].text(
            x, value + max(total) * 0.025, f"{value:g}",
            ha="center", va="bottom", fontsize=9,
        )
    axes[1].set_xticks(positions, SIZE_LABELS)
    axes[1].set_ylabel("Median pivot count")
    axes[1].set_title("B. Iterations by phase")
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].grid(axis="y", alpha=0.22)

    metrics = [
        ("primal", "Primal residual", "#3d5a80", "o"),
        ("objective", "Relative objective error", "#b23a48", "s"),
        ("duality", "Relative duality gap", "#4c9f70", "^"),
    ]
    floor = np.finfo(float).tiny
    for key, label, color, marker in metrics:
        values = [max(max(data[size][key]), floor) for size in SIZE_ORDER]
        axes[2].plot(
            positions, values, label=label, color=color, marker=marker,
            linewidth=2, markersize=7,
        )
    axes[2].axhline(
        1e-7, color="#52606d", linestyle="--", linewidth=1.4,
        label="Proposed threshold (1e-7)",
    )
    axes[2].set_yscale("log")
    axes[2].set_xticks(positions, SIZE_LABELS)
    axes[2].set_ylabel("Maximum error across cases (log scale)")
    axes[2].set_title("C. Numerical accuracy")
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")
    axes[2].grid(axis="y", which="both", alpha=0.22)

    fig.suptitle(
        "Two-Phase Tableau Simplex — 54 Cases, 10 Timed Cold Starts per Case",
        fontsize=14,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {output}")
    print(f"wrote {pdf_output}")


def main() -> None:
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    make_figure(aggregate(load_rows(args.input)), args.output, args.dpi)


if __name__ == "__main__":
    main()
