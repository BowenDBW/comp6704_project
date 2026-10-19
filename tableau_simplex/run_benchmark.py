"""Protocol-aligned benchmark runner for Two-Phase Tableau Simplex."""

from __future__ import annotations

import os

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from tableau_simplex import solve


HERE = Path(__file__).resolve().parent
ALGORITHM = "Tableau Simplex"
PROTOCOL_VERSION = "v1"
HISTORY_CASES = {
    "M03_N05_a030_seed2026_base",
    "M10_N20_a090_seed2028_base",
    "M10_N20_a090_seed2028_demand_perturbed",
}
ACCEPTANCE_LIMITS = {
    "primal_residual_inf": 1e-7,
    "min_z": -1e-8,
    "dual_residual_inf": 1e-7,
    "min_dual_slack": -1e-7,
    "relative_duality_gap": 1e-7,
    "relative_objective_error": 1e-7,
}


def default_benchmark_root() -> Path:
    """Support both the GitHub layout and this local development workspace."""
    candidates = (
        HERE.parent / "shared",
        HERE.parent / "COMP6704_Benchmark_v1_validated",
    )
    for candidate in candidates:
        if (candidate / "benchmark_v1" / "manifest.csv").is_file():
            return candidate
    return candidates[0]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root: Path) -> dict[str, str]:
    files = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        is_benchmark_data = relative.parts[0] == "benchmark_v1"
        is_root_protocol_file = len(relative.parts) == 1 and not relative.name.startswith(
            "TABLEAU_"
        )
        if is_benchmark_data or is_root_protocol_file:
            files[str(relative).replace("\\", "/")] = digest(path)
    return files


def load_references(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["case_id"]: float(row["highs_objective"])
            for row in csv.DictReader(handle)
        }


def shared_status(status: str) -> str:
    return {
        "infeasible": "infeasible_claim",
        "unbounded": "unbounded_claim",
    }.get(status, status)


def max_available(rows: list[dict[str, object]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) not in ("", None)]
    return max(values) if values else None


def total_ram_gb() -> float | None:
    """Return installed RAM using the Windows API without another dependency."""
    if platform.system() != "Windows":
        return None
    try:
        import ctypes

        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return round(status.ullTotalPhys / (1024**3), 1)
    except (AttributeError, OSError):
        return None


def evaluate_result(
    result: dict, A: np.ndarray, b: np.ndarray, c: np.ndarray, reference: float
) -> tuple[dict[str, float], bool]:
    z, y, r = result["z"], result["y"], result["r"]
    if not all(np.isfinite(value).all() for value in (z, y, r)):
        return {}, False
    primal_objective = float(c @ z)
    dual_objective = float(b @ y)
    metrics = {
        "objective": primal_objective,
        "primal_residual_inf": float(np.linalg.norm(A @ z - b, np.inf)),
        "min_z": float(np.min(z)),
        "dual_residual_inf": float(np.linalg.norm(A.T @ y + r - c, np.inf)),
        "min_dual_slack": float(np.min(r)),
        "relative_duality_gap": abs(primal_objective - dual_objective)
        / max(1.0, abs(primal_objective), abs(dual_objective)),
        "relative_objective_error": abs(primal_objective - reference)
        / max(1.0, abs(reference)),
    }
    passed = result["status"] == "optimal"
    for field in (
        "primal_residual_inf",
        "dual_residual_inf",
        "relative_duality_gap",
        "relative_objective_error",
    ):
        passed = passed and metrics[field] <= ACCEPTANCE_LIMITS[field]
    for field in ("min_z", "min_dual_slack"):
        passed = passed and metrics[field] >= ACCEPTANCE_LIMITS[field]
    return metrics, bool(passed)


def blas_summary() -> str:
    """Report BLAS implementations without exposing installation paths."""
    try:
        from threadpoolctl import threadpool_info

        summaries = []
        for item in threadpool_info():
            if item.get("user_api") != "blas":
                continue
            name = item.get("internal_api") or item.get("prefix") or "BLAS"
            version = item.get("version") or "version unknown"
            threading = item.get("threading_layer")
            summary = f"{name} {version}"
            if threading:
                summary += f" ({threading})"
            summaries.append(summary)
        return "; ".join(summaries) or "BLAS details unavailable"
    except ImportError:
        return "BLAS details unavailable"


def environment_record(environment_id: str) -> dict:
    try:
        import scipy

        scipy_version = scipy.__version__
    except ImportError:
        scipy_version = None
    return {
        "environment_id": environment_id,
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.machine(),
        "physical_cores_logical": os.cpu_count(),
        "ram_total_gb": total_ram_gb(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "blas_summary": blas_summary(),
        "thread_env": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def save_solution(output: Path, case_id: str, result: dict) -> None:
    np.savez_compressed(
        output / "solutions" / f"{case_id}_solution.npz",
        z=result["z"],
        y=result["y"],
        r=result["r"],
        basis=np.asarray(result["basis"], dtype=np.int64),
    )
    if case_id not in HISTORY_CASES:
        return
    history_rows = []
    for phase, values in (
        ("phase1", result["phase1_objective_history"]),
        ("phase2", result["phase2_objective_history"]),
    ):
        history_rows.extend(
            {"phase": phase, "iteration": index, "objective": value}
            for index, value in enumerate(values)
        )
    with (output / "history" / f"{case_id}_history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("phase", "iteration", "objective"))
        writer.writeheader()
        writer.writerows(history_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=default_benchmark_root())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--small-only", action="store_true")
    parser.add_argument("--max-m", type=int, choices=(3, 5, 10), default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument(
        "--pivot-rule", choices=("most_negative", "bland"), default="most_negative"
    )
    parser.add_argument("--environment-id", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise RuntimeError("use a fresh output directory to preserve prior results")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    benchmark = args.benchmark.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True)
    (output / "solutions").mkdir()
    (output / "history").mkdir()

    before = snapshot(benchmark)
    references = load_references(benchmark / "VALIDATION_SUMMARY.csv")
    with (benchmark / "benchmark_v1" / "manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        manifest = list(csv.DictReader(handle))
    if args.small_only:
        args.case_id = ["M03_N05_a030_seed2026_base"]
    selected = [
        item
        for item in manifest
        if int(item["M"]) <= args.max_m
        and (args.case_id is None or item["case_id"] in args.case_id)
    ]
    if not selected:
        raise ValueError("no benchmark cases matched the selection")

    options = {
        "tol": args.tol,
        "max_iter": args.max_iter,
        "pivot_rule": args.pivot_rule,
    }
    environment_id = args.environment_id or (
        f"ENV_{platform.system()}_PY{sys.version_info.major}{sys.version_info.minor}"
        f"_NP{np.__version__.replace('.', '')}"
    )
    with (benchmark / "result_schema.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        schema = next(csv.reader(handle))
    rows: list[dict[str, object]] = []
    failures = 0
    warmup_failures = 0

    for item in selected:
        case_id = item["case_id"]
        path = benchmark / "benchmark_v1" / item["npz_file"]
        with np.load(path, allow_pickle=False) as data:
            A, b, c = (data[key].copy() for key in ("A", "b", "c_hat"))
        if not args.no_warmup:
            warmup = solve(A, b, c, options)
            _, passed = evaluate_result(warmup, A, b, c, references[case_id])
            warmup_failures += int(not passed)

        for repeat_id in range(1, args.repeats + 1):
            result = solve(A, b, c, options)
            metrics, passed = evaluate_result(result, A, b, c, references[case_id])
            failures += int(not passed)
            row: dict[str, object] = {field: "" for field in schema}
            row.update(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "case_id": case_id,
                    "algorithm": ALGORITHM,
                    "mode": "cold",
                    "repeat_id": repeat_id,
                    "status": shared_status(result["status"])
                    if passed
                    else "numerical_failure",
                    **metrics,
                    "iterations": result["iterations"],
                    "phase1_iterations": result["phase1_iterations"],
                    "phase2_iterations": result["phase2_iterations"],
                    "runtime_seconds": result["runtime"],
                    "environment_id": environment_id,
                    "options_id": (
                        f"tableau_v1_tol{args.tol:g}_{args.pivot_rule}_"
                        f"max{args.max_iter}"
                    ),
                    "message": result["message"],
                }
            )
            rows.append(row)
            if repeat_id == 1 and result["status"] == "optimal":
                save_solution(output, case_id, result)
        print(f"{case_id}: {args.repeats} runs completed", flush=True)

    with (output / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema)
        writer.writeheader()
        writer.writerows(rows)

    by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    summaries = []
    for case_id, case_rows in by_case.items():
        successful = [row for row in case_rows if row["status"] == "optimal"]
        summaries.append(
            {
                "case_id": case_id,
                "successful_runs": len(successful),
                "total_runs": len(case_rows),
                "median_runtime_seconds": statistics.median(
                    float(row["runtime_seconds"]) for row in successful
                )
                if successful
                else "",
                "iterations": case_rows[-1]["iterations"],
                "phase1_iterations": case_rows[-1]["phase1_iterations"],
                "phase2_iterations": case_rows[-1]["phase2_iterations"],
                "max_primal_residual_inf": max(
                    float(row["primal_residual_inf"]) for row in successful
                )
                if successful
                else "",
                "max_relative_objective_error": max(
                    float(row["relative_objective_error"]) for row in successful
                )
                if successful
                else "",
            }
        )
    with (output / "case_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    after = snapshot(benchmark)
    if before != after:
        raise RuntimeError("shared benchmark files changed during the run")
    report = {
        "algorithm": ALGORITHM,
        "cases": len(selected),
        "timed_runs": len(rows),
        "warmup_runs": 0 if args.no_warmup else len(selected),
        "failed": failures,
        "warmups_failed": warmup_failures,
        "all_pass": failures == 0 and warmup_failures == 0,
        "shared_files_unchanged": len(before),
        "max_primal_residual_inf": max_available(rows, "primal_residual_inf"),
        "max_relative_objective_error": max_available(
            rows, "relative_objective_error"
        ),
        "max_relative_duality_gap": max_available(rows, "relative_duality_gap"),
    }
    (output / "environment.json").write_text(
        json.dumps(environment_record(environment_id), indent=2),
        encoding="utf-8",
    )
    (output / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output / "input_sha256.json").write_text(
        json.dumps(before, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return int(not report["all_pass"])


if __name__ == "__main__":
    raise SystemExit(main())
