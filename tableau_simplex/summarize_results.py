"""Summarize raw tableau benchmark rows without third-party data libraries."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


CASE_PATTERN = re.compile(r"^M(?P<M>\d+)_N(?P<N>\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--case-output", type=Path)
    parser.add_argument("--size-output", type=Path)
    return parser.parse_args()


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise ValueError("input has no result rows")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw:
        grouped[row["case_id"]].append(row)

    case_rows: list[dict[str, object]] = []
    for case_id, rows in grouped.items():
        match = CASE_PATTERN.match(case_id)
        if match is None:
            raise ValueError(f"unrecognized case id: {case_id}")
        case_rows.append(
            {
                "case_id": case_id,
                "M": int(match.group("M")),
                "N": int(match.group("N")),
                "runs": len(rows),
                "optimal_runs": sum(row["status"] == "optimal" for row in rows),
                "median_runtime_seconds": statistics.median(
                    float(row["runtime_seconds"]) for row in rows
                ),
                "iterations": int(rows[0]["iterations"]),
                "phase1_iterations": int(rows[0]["phase1_iterations"]),
                "phase2_iterations": int(rows[0]["phase2_iterations"]),
                "max_primal_residual_inf": max(
                    float(row["primal_residual_inf"]) for row in rows
                ),
                "max_relative_objective_error": max(
                    float(row["relative_objective_error"]) for row in rows
                ),
            }
        )
    case_rows.sort(key=lambda row: str(row["case_id"]))

    by_size: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for row in case_rows:
        by_size[(int(row["M"]), int(row["N"]))].append(row)
    size_rows: list[dict[str, object]] = []
    for (m, n), rows in sorted(by_size.items()):
        runtimes = [float(row["median_runtime_seconds"]) for row in rows]
        iterations = [int(row["iterations"]) for row in rows]
        size_rows.append(
            {
                "M": m,
                "N": n,
                "cases": len(rows),
                "runs": sum(int(row["runs"]) for row in rows),
                "optimal_runs": sum(int(row["optimal_runs"]) for row in rows),
                "median_case_runtime_seconds": statistics.median(runtimes),
                "min_case_median_runtime_seconds": min(runtimes),
                "max_case_median_runtime_seconds": max(runtimes),
                "median_iterations": statistics.median(iterations),
                "min_iterations": min(iterations),
                "max_iterations": max(iterations),
                "max_primal_residual_inf": max(
                    float(row["max_primal_residual_inf"]) for row in rows
                ),
                "max_relative_objective_error": max(
                    float(row["max_relative_objective_error"]) for row in rows
                ),
            }
        )

    case_output = args.case_output or args.input.with_name("tableau_case_summary.csv")
    size_output = args.size_output or args.input.with_name("tableau_size_summary.csv")
    _write(case_output, case_rows)
    _write(size_output, size_rows)
    print(f"wrote {len(case_rows)} case summaries to {case_output}")
    print(f"wrote {len(size_rows)} size summaries to {size_output}")


if __name__ == "__main__":
    main()
