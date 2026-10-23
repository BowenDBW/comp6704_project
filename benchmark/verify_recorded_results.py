#!/usr/bin/env python3
"""Read-only checks of archived COMP6704 measurements; Python standard library only.

From the extracted repository root:
    python benchmark/verify_recorded_results.py --root .
    python benchmark/verify_recorded_results.py --root . --json

This checks hashes, recorded certificates, aggregation, and reported values.
It does not execute a solver or recompute certificates from solution arrays.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics
import sys


ALGORITHMS = ("Tableau", "Revised", "Dual", "IPM")
SIZES = ((3, 5), (5, 10), (10, 20))
LIMITS = {
    "primal_residual_inf": 1e-7,
    "min_z": -1e-8,
    "dual_residual_inf": 1e-7,
    "min_dual_slack": -1e-7,
    "relative_duality_gap": 1e-7,
    "relative_objective_error": 1e-7,
}
# Approved report table entries, transcribed at their displayed precision.
# Values are compared at the precision displayed in the report.
REPORTED_RUNTIME_MS = (
    ("1.28", "2.72", "0.80", "1.28"),
    ("6.89", "19.31", "3.79", "3.20"),
    ("240.28", "437.88", "183.79", "27.73"),
)
REPORTED_ITERATIONS = ((38.5, 22, 9, 8), (132, 66, 49, 9), (553.5, 287.5, 501, 10))
REPORTED_ACCURACY = ("5.89e-16", "5.87e-16", "4.71e-16", "2.35e-10")
REPORTED_SPEEDUP = ("4.36", "8.55", "102.23")


class CheckFailure(Exception):
    pass


def require(condition, message):
    if not condition:
        raise CheckFailure(message)


def close(actual, expected, description):
    """Allow only summation/interpolation rounding, even for tiny error metrics."""
    require(
        math.isfinite(actual)
        and math.isfinite(expected)
        and math.isclose(actual, expected, rel_tol=5e-12, abs_tol=1e-24),
        f"{description}: calculated {actual!r}, archived {expected!r}",
    )


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_path(root, relative):
    # The archived source-hash record was written on Windows.
    path = (root / relative.replace("\\", "/")).resolve()
    require(path.is_relative_to(root), f"Path outside repository root: {relative}")
    require(path.is_file(), f"Missing file: {relative}")
    return path


def verify_hashes(root, manifest, prefix=""):
    require(
        isinstance(manifest, dict) and bool(manifest), "Empty or invalid hash manifest"
    )
    for relative, expected in manifest.items():
        require(
            isinstance(relative, str)
            and isinstance(expected, str)
            and len(expected) == 64
            and set(expected.lower()) <= set("0123456789abcdef"),
            "Invalid manifest entry",
        )
        path = safe_path(root, prefix + relative)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(
            actual.lower() == expected.lower(), f"SHA256 mismatch: {prefix + relative}"
        )
    return len(manifest)


def linear_quantile(values, probability):
    """Hyndman-Fan type 7: the original NumPy quantile default ('linear')."""
    ordered = sorted(values)
    require(bool(ordered), "Cannot take a quantile of an empty sample")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def row_key(row):
    return (row["case_id"], row["algorithm"], row["mode"], int(row["repeat_id"]))


def indexed(rows, fields, label):
    result = {}
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        require(key not in result, f"Duplicate {label} key: {key}")
        result[key] = row
    return result


def compare_summary(path, calculated, keys):
    archived = indexed(read_csv(path), keys, path.name)
    computed = indexed(calculated, keys, f"computed {path.name}")
    require(archived.keys() == computed.keys(), f"Coverage differs: {path.name}")
    for key, actual in computed.items():
        expected = archived[key]
        require(actual.keys() == expected.keys(), f"Columns differ: {path.name}")
        for field, value in actual.items():
            label = f"{path.name} {key} {field}"
            if isinstance(value, (int, float)):
                close(float(value), float(expected[field]), label)
            else:
                require(value == expected[field], f"{label}: text differs")
    return len(calculated)


def verify(root):
    root = root.resolve()
    manifest_count = verify_hashes(root, read_json(root / "MANIFEST.json"))
    recorded = root / "recorded_results"
    source_hash_count = verify_hashes(
        root, read_json(recorded / "input_sha256.json"), ""
    )
    environment = read_json(recorded / "environment.json")
    require(
        environment["limits"] == LIMITS,
        "Recorded acceptance limits differ from protocol",
    )
    benchmark = read_csv(root / "shared/benchmark_v1/manifest.csv")
    cases = indexed(benchmark, ("case_id",), "benchmark")
    require(len(cases) == 54, "Expected 54 benchmark cases")
    expected_grid = {
        (m, n, alpha, seed, kind)
        for m, n in SIZES
        for alpha in (0.3, 0.6, 0.9)
        for seed in (2026, 2027, 2028)
        for kind in ("base", "demand_perturbed")
    }
    actual_grid = {
        (int(r["M"]), int(r["N"]), float(r["alpha"]), int(r["seed"]), r["kind"])
        for r in benchmark
    }
    require(actual_grid == expected_grid, "Benchmark grid is incomplete or unexpected")
    references = {
        r["case_id"]: float(r["highs_objective"])
        for r in read_csv(root / "shared/VALIDATION_SUMMARY.csv")
    }
    rows = read_csv(recorded / "measured_runs.csv")
    all_rows = read_csv(recorded / "runs_including_warmups.csv")
    require(len(rows) == 2430 and len(all_rows) == 2673, "Unexpected run counts")
    require(len({row_key(r) for r in all_rows}) == len(all_rows), "Duplicate run key")
    require(len({row_key(r) for r in rows}) == len(rows), "Duplicate measured run key")
    all_measured = {row_key(r): r for r in all_rows if int(r["repeat_id"]) > 0}
    require(
        {row_key(r): r for r in rows} == all_measured,
        "Measured CSV differs from full run log",
    )
    expected_keys = set()
    for case in benchmark:
        for algorithm in ALGORITHMS:
            expected_keys.update(
                (case["case_id"], algorithm, "cold", rep) for rep in range(11)
            )
        if case["kind"] == "demand_perturbed":
            expected_keys.update(
                (case["case_id"], "Dual", "warm", rep) for rep in range(11)
            )
    require(
        {row_key(r) for r in all_rows} == expected_keys,
        "Run coverage or repeat IDs differ",
    )
    for row in all_rows:
        key = row_key(row)
        case = cases[(row["case_id"],)]
        require(
            all(row[k] == case[k] for k in ("M", "N", "alpha")),
            f"Wrong case metadata: {key}",
        )
        require(
            row["status"] == "optimal" and row["validation_pass"] == "True",
            f"Failed run: {key}",
        )
        runtime = float(row["runtime_seconds"])
        require(math.isfinite(runtime) and runtime > 0, f"Invalid runtime: {key}")
        iterations = float(row["iterations"])
        require(
            math.isfinite(iterations) and iterations >= 0 and iterations.is_integer(),
            f"Invalid iteration count: {key}",
        )
        for metric, limit in LIMITS.items():
            value = float(row[metric])
            accepted = (
                value >= limit if metric.startswith("min_") else 0 <= value <= limit
            )
            require(
                math.isfinite(value) and accepted, f"Recorded {metric} fails: {key}"
            )
        objective = float(row["objective"])
        reference = references[row["case_id"]]
        error = abs(objective - reference) / max(1.0, abs(reference))
        close(
            error,
            float(row["relative_objective_error"]),
            f"Reference objective error {key}",
        )
    completion = read_json(recorded / "completion.json")
    require(
        completion["rows"] == len(all_rows)
        and completion["measured"] == len(rows)
        and completion["failed"] == 0,
        "Completion counts disagree",
    )

    groups = defaultdict(list)
    for row in rows:
        groups[(row["algorithm"], row["mode"], row["case_id"])].append(row)
    case_summary = []
    for (algorithm, mode, case_id), group in sorted(groups.items()):
        require(len(group) == 10, f"Expected ten repeats: {case_id} {algorithm} {mode}")
        require(
            len({r["iterations"] for r in group}) == 1,
            f"Iterations vary between repeats: {case_id}",
        )
        times = [1000 * float(r["runtime_seconds"]) for r in group]
        case_summary.append(
            dict(
                algorithm=algorithm,
                mode=mode,
                case_id=case_id,
                M=int(group[0]["M"]),
                N=int(group[0]["N"]),
                alpha=float(group[0]["alpha"]),
                runs=len(group),
                median_ms=statistics.median(times),
                q1_ms=linear_quantile(times, 0.25),
                q3_ms=linear_quantile(times, 0.75),
                median_iterations=statistics.median(
                    float(r["iterations"]) for r in group
                ),
            )
        )
    sizes, accuracy, warm = [], [], []
    for algorithm in ALGORITHMS:
        group = [r for r in rows if r["algorithm"] == algorithm and r["mode"] == "cold"]
        require(len(group) == 540, f"Expected 540 cold runs: {algorithm}")
        accuracy.append(
            dict(
                algorithm=algorithm,
                passed=len(group),
                total=len(group),
                **{
                    metric: (min if metric.startswith("min_") else max)(
                        float(r[metric]) for r in group
                    )
                    for metric in LIMITS
                },
            )
        )
        for m, n in SIZES:
            group = [
                r
                for r in case_summary
                if r["algorithm"] == algorithm
                and r["mode"] == "cold"
                and (r["M"], r["N"]) == (m, n)
            ]
            require(len(group) == 18, f"Expected eighteen cases: {algorithm} {m}x{n}")
            times = [r["median_ms"] for r in group]
            iterations = [r["median_iterations"] for r in group]
            sizes.append(
                dict(
                    algorithm=algorithm,
                    M=m,
                    N=n,
                    cases=len(group),
                    median_ms=statistics.median(times),
                    q1_case_ms=linear_quantile(times, 0.25),
                    q3_case_ms=linear_quantile(times, 0.75),
                    median_iterations=statistics.median(iterations),
                    min_iterations=min(iterations),
                    max_iterations=max(iterations),
                )
            )
    case_lookup = {(r["algorithm"], r["mode"], r["case_id"]): r for r in case_summary}
    for m, n in SIZES:
        w = [
            r
            for r in case_summary
            if r["mode"] == "warm" and (r["M"], r["N"]) == (m, n)
        ]
        require(len(w) == 9, f"Expected nine warm pairs: {m}x{n}")
        c = [case_lookup[("Dual", "cold", r["case_id"])] for r in w]
        cold_mean = statistics.mean(r["median_iterations"] for r in c)
        warm_mean = statistics.mean(r["median_iterations"] for r in w)
        warm.append(
            dict(
                M=m,
                N=n,
                pairs=len(w),
                cold_mean_pivots=cold_mean,
                warm_mean_pivots=warm_mean,
                pivot_reduction_pct=100 * (1 - warm_mean / cold_mean),
                cold_median_ms=statistics.median(r["median_ms"] for r in c),
                warm_median_ms=statistics.median(r["median_ms"] for r in w),
                median_paired_speedup=statistics.median(
                    a["median_ms"] / b["median_ms"] for a, b in zip(c, w)
                ),
                zero_pivot_cases=sum(r["median_iterations"] == 0 for r in w),
            )
        )
    checked_summaries = {}
    for filename, calculated, keys in (
        ("case_summary.csv", case_summary, ("algorithm", "mode", "case_id")),
        ("size_summary.csv", sizes, ("algorithm", "M", "N")),
        ("accuracy_summary.csv", accuracy, ("algorithm",)),
        ("warm_start_summary.csv", warm, ("M", "N")),
    ):
        checked_summaries[filename] = compare_summary(
            recorded / filename, calculated, keys
        )

    by_size = {(r["algorithm"], r["M"], r["N"]): r for r in sizes}
    runtime_table, iteration_table = [], []
    for i, (m, n) in enumerate(SIZES):
        runtime_row = [f"{by_size[(a, m, n)]['median_ms']:.2f}" for a in ALGORITHMS]
        iteration_row = [by_size[(a, m, n)]["median_iterations"] for a in ALGORITHMS]
        require(
            tuple(runtime_row) == REPORTED_RUNTIME_MS[i],
            f"Displayed runtime table differs: {m}x{n}",
        )
        require(
            tuple(iteration_row) == REPORTED_ITERATIONS[i],
            f"Displayed iteration table differs: {m}x{n}",
        )
        runtime_table.append({"size": f"{m}x{n}", **dict(zip(ALGORITHMS, runtime_row))})
        iteration_table.append(
            {"size": f"{m}x{n}", **dict(zip(ALGORITHMS, iteration_row))}
        )
    accuracy_table = [
        dict(
            algorithm=r["algorithm"],
            passed=r["passed"],
            total=r["total"],
            max_relative_objective_error=f"{r['relative_objective_error']:.2e}",
        )
        for r in accuracy
    ]
    require(
        tuple(r["max_relative_objective_error"] for r in accuracy_table)
        == REPORTED_ACCURACY,
        "Displayed accuracy table differs",
    )
    warm_table = [
        dict(size=f"{r['M']}x{r['N']}", speedup=f"{r['median_paired_speedup']:.2f}")
        for r in warm
    ]
    require(
        tuple(r["speedup"] for r in warm_table) == REPORTED_SPEEDUP,
        "Displayed warm-speedup table differs",
    )
    require(
        sum(r["zero_pivot_cases"] for r in warm) == 8,
        "Zero-pivot total differs from report",
    )
    return {
        "status": "PASS",
        "package_root": str(root),
        "scope": "Read-only verification of archived files and recorded metrics; no solver execution or solution-array replay.",
        "manifest_files_checked": manifest_count,
        "input_and_solver_hashes_checked": source_hash_count,
        "measured_runs": len(rows),
        "warmup_runs": len(all_rows) - len(rows),
        "cold_runs": 2160,
        "warm_runs": 270,
        "recorded_acceptance_failures": 0,
        "summary_rows_checked": checked_summaries,
        "report_tables": {
            "accuracy": accuracy_table,
            "iterations": iteration_table,
            "runtime_ms": runtime_table,
            "warm_speedup": warm_table,
        },
        "runtime_iqr_ms": [
            dict(
                algorithm=r["algorithm"],
                M=r["M"],
                N=r["N"],
                median=r["median_ms"],
                q1=r["q1_case_ms"],
                q3=r["q3_case_ms"],
                iqr=r["q3_case_ms"] - r["q1_case_ms"],
            )
            for r in sizes
        ],
        "warm_start_details": warm,
        "aggregation": {
            "runtime": "Median of ten runs per case, then median of eighteen case medians per size.",
            "runtime_iqr": "Q1 and Q3 across eighteen case medians; linear (type 7) interpolation; not a confidence interval.",
            "warm_speedup": "Median of nine paired cold/warm per-case median-time ratios.",
            "warm_pivots": "Arithmetic mean over nine paired perturbed cases; 100*(1-warm_mean/cold_mean).",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Extracted repository root (default: current directory)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print detailed machine-readable JSON to stdout",
    )
    args = parser.parse_args()
    try:
        result = verify(args.root)
    except (CheckFailure, OSError, ValueError, KeyError, TypeError) as error:
        result = {"status": "FAIL", "error": str(error)}
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print(
            f"PASS: {result['manifest_files_checked']} manifest hashes; "
            f"{result['measured_runs']} measured runs and {result['warmup_runs']} warmups; "
            "all archived summaries and four displayed report tables agree."
        )
        print(
            "Use --json for runtime quartiles and complete warm-start pivot statistics."
        )
        print(result["scope"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
