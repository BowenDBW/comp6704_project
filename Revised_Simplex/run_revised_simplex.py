"""Run the released benchmark and save results and comparison summaries"""
import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import statistics
import numpy as np
from revised_simplex import ACCEPTANCE, DEFAULT_OPTIONS, solve

HISTORY_CASE_IDS = {
    "M03_N05_a030_seed2026_base",
    "M10_N20_a090_seed2028_base",
    "M10_N20_a090_seed2028_demand_perturbed",
}

# Summarize the timestamp and reported environment settings in one CSV field.
def _environment_id():
    """Keep a compact environment record in each CSV row."""
    run_time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    configuration = np.show_config(mode="dicts")
    dependencies = configuration.get("Build Dependencies", {})
    blas = dependencies.get("blas", {})
    parts = []
    parts.append("run=" + run_time)
    parts.append("python=" + platform.python_version())
    parts.append("numpy=" + np.__version__)
    parts.append("os=" + platform.platform())
    parts.append("cpu=" + platform.machine())
    parts.append("logical_cpus=" + str(os.cpu_count()))
    parts.append("blas=" + blas.get("name", "unknown") + ":" + blas.get("version", "unknown"))
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        value = os.environ.get(name, "unset")
        parts.append(name + "=" + value)
    parts.append("actual_threads=unmeasured")
    return ";".join(parts)


# Check the solver output against the shared tolerances and reference objective.
def _evaluate(A, b, c, result, reference):
    """Recheck all six criteria using the returned arrays and the reference."""
    metrics = {"objective": None}
    for name in ACCEPTANCE:
        metrics[name] = None
    status = result["status"]
    message = result["message"]
    z = result["z"]
    y = result["y"]
    if z is None or y is None:
        if status == "optimal":
            status = "numerical_failure"
        return metrics, False, status, message
    if z.shape != c.shape or y.shape != b.shape:
        return metrics, False, "numerical_failure", "Invalid output shape."

    with np.errstate(over="ignore", invalid="ignore"):
        r = c - A.T @ y
        objective = float(c @ z)
        dual_objective = float(b @ y)
        gap_scale = max(1.0, abs(objective), abs(dual_objective))
        reference_scale = max(1.0, abs(reference))
        metrics["objective"] = objective
        metrics["primal_residual_inf"] = float(np.max(np.abs(A @ z - b)))
        metrics["min_z"] = float(np.min(z))
        metrics["dual_residual_inf"] = float(np.max(np.abs(A.T @ y + r - c)))
        metrics["min_dual_slack"] = float(np.min(r))
        metrics["relative_duality_gap"] = abs(objective - dual_objective) / gap_scale
        metrics["relative_objective_error"] = abs(objective - reference) / reference_scale

    finite = True
    for value in metrics.values():
        if not np.isfinite(value):
            finite = False
    primal_ok = metrics["primal_residual_inf"] <= ACCEPTANCE["primal_residual_inf"]
    variables_ok = metrics["min_z"] >= ACCEPTANCE["min_z"]
    dual_ok = metrics["dual_residual_inf"] <= ACCEPTANCE["dual_residual_inf"]
    reduced_costs_ok = metrics["min_dual_slack"] >= ACCEPTANCE["min_dual_slack"]
    gap_ok = metrics["relative_duality_gap"] <= ACCEPTANCE["relative_duality_gap"]
    reference_ok = metrics["relative_objective_error"] <= ACCEPTANCE["relative_objective_error"]
    valid = finite and primal_ok and variables_ok and dual_ok and reduced_costs_ok and gap_ok and reference_ok
    if not finite or (status == "optimal" and not valid):
        status = "numerical_failure"
        message = "External feasibility/optimality/reference check failed."
    for name in metrics:
        if not np.isfinite(metrics[name]):
            metrics[name] = None  # Unavailable CSV values stay blank, not zero.
    passed = status == "optimal" and valid
    return metrics, bool(passed), status, message


# Sort cases order.
def _case_order(row):
    """Order cases by size, capacity, seed; then base before perturbation."""
    return (int(row["M"]), int(row["N"]), float(row["alpha"]),
            int(row["seed"]), row["kind"] != "base")


# Reduce repeated runs to one comparison row per case.
def _case_summary(records):
    """Summarize successful timings and all available accuracy checks case by case."""
    groups = {}
    for record in records:
        groups.setdefault(record["case_id"], []).append(record)

    summaries = []
    for case_id, rows in groups.items():
        successful = [row for row in rows if row["status"] == "optimal"]
        times = [float(row["runtime_seconds"]) for row in successful]
        summary = {
            "case_id": case_id,
            "successful_runs": len(successful),
            "total_runs": len(rows),
            "median_runtime_seconds": statistics.median(times) if times else None,
            "runtime_iqr_seconds": None,
        }
        if len(times) >= 2:
            q1, _, q3 = statistics.quantiles(times, n=4, method="inclusive")
            summary["runtime_iqr_seconds"] = q3 - q1
        for name in ["iterations", "phase1_iterations", "phase2_iterations"]:
            values = [int(row[name]) for row in successful if row[name] not in (None, "")]
            summary[name] = statistics.median(values) if values else None
        for name in ["primal_residual_inf", "relative_objective_error"]:
            values = [float(row[name]) for row in rows if row[name] not in (None, "")]
            summary["max_" + name] = max(values) if values else None
        summaries.append(summary)
    return summaries


def _alpha_summary(records, cases):
    """Compare objective values for matched cases as capacity tightens."""
    runs = {}
    for record in records:
        runs.setdefault(record["case_id"], []).append(record)

    objectives = {}
    groups = {}
    matched = {}
    for case in cases:
        case_id = case["case_id"]
        good = [float(row["objective"]) for row in runs[case_id]
                if row["status"] == "optimal" and row["objective"] not in (None, "")]
        objectives[case_id] = statistics.median(good) if good else None
        size_alpha = (int(case["M"]), int(case["N"]), float(case["alpha"]))
        groups.setdefault(size_alpha, []).append(case)
        pair = (int(case["M"]), int(case["N"]), case["seed"], case["kind"], float(case["alpha"]))
        matched[pair] = case_id

    summaries = []
    for (M, N, alpha), group in sorted(groups.items()):
        values = [objectives[case["case_id"]] for case in group
                  if objectives[case["case_id"]] is not None]
        changes = []
        for case in group:
            baseline_id = matched.get((M, N, case["seed"], case["kind"], 0.3))
            current = objectives[case["case_id"]]
            baseline = objectives.get(baseline_id)
            if current is not None and baseline is not None:
                changes.append((current - baseline) / max(1.0, abs(baseline)))
        summaries.append({
            "M": M, "N": N, "alpha": alpha, "cases": len(group),
            "successful_cases": len(values), "paired_cases": len(changes),
            "median_objective": statistics.median(values) if values else None,
            "median_relative_change_from_alpha_0_3": statistics.median(changes) if changes else None,
        })
    return summaries


# Load cases, run repeated cold starts, validate results, and write result files.
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    folder = Path(__file__).resolve().parent
    # Short flags match the group runners; the original names still work.
    parser.add_argument("--benchmark", "--benchmark-dir", dest="benchmark_dir", type=Path,
                        default=folder.parent / "shared")
    parser.add_argument("--output", "--output-dir", dest="output_dir",
                        type=Path, default=folder / "results")
    parser.add_argument("--case-id")
    parser.add_argument("--small-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_OPTIONS["max_iter"])
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.max_iter < 0:
        parser.error("repeats must be positive; max-iter must be nonnegative.")
    if args.small_only:
        first_case = "M03_N05_a030_seed2026_base"
        if args.case_id is not None and args.case_id != first_case:
            parser.error("--small-only cannot select a different --case-id.")
        args.case_id = first_case
    given = args.benchmark_dir.resolve()
    if (given / "manifest.csv").is_file():
        data_dir = given
    else:
        data_dir = given / "benchmark_v1"
    release = data_dir.parent
    output_dir = args.output_dir.resolve()
    if output_dir == release or release in output_dir.parents:
        parser.error("Output must be outside the read-only release.")

    cases = []
    with (data_dir / "manifest.csv").open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if args.case_id is None or row["case_id"] == args.case_id:
                cases.append(row)
    if len(cases) == 0:
        parser.error("No matching case_id.")
    cases.sort(key=_case_order)
    references = {}
    with (release / "VALIDATION_SUMMARY.csv").open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            references[row["case_id"]] = float(row["highs_objective"])
    with (release / "result_schema.csv").open(newline="", encoding="utf-8-sig") as file:
        fields = next(csv.reader(file))

    options = DEFAULT_OPTIONS.copy()
    options["max_iter"] = args.max_iter
    options_id = (f"rs_basic_v3_rc{options['reduced_cost_tol']}_dir{options['direction_tol']}"
                  f"_p1{options['phase1_tol']}_maxiter{args.max_iter}")
    environment_id = _environment_id()
    print("Environment: " + environment_id, flush=True)
    print("Settings: " + options_id + "; protocol=v1 (proposed)", flush=True)
    records = []
    histories = {}
    passed_cases = 0
    measured_successes = 0
    warmup_successes = 0
    base_count = 0
    perturbed_count = 0

    for number, row in enumerate(cases, 1):
        case_id = row["case_id"]
        if row["kind"] == "base":
            base_count += 1
        elif row["kind"] == "demand_perturbed":
            perturbed_count += 1
        reference = references[case_id]
        if not np.isfinite(reference):
            raise ValueError(f"Nonfinite reference: {case_id}")
        with np.load(data_dir / row["npz_file"], allow_pickle=False) as data:
            A = data["A"]
            b = data["b"]
            c = data["c_hat"]
        case_options = options.copy()
        case_options["M"] = int(row["M"])
        case_options["N"] = int(row["N"])
        successes = 0
        successful_times = []
        warmup_passed = False
        warmup_status = "not_run"
        warmup_message = ""

        # Repeat 0 is the warm-up. Every call builds its own initial basis.
        for repeat in range(args.repeats + 1):
            result = solve(A, b, c, case_options)
            metrics, passed, status, message = _evaluate(A, b, c, result, reference)
            if repeat == 0:
                warmup_passed = passed
                warmup_status = status
                warmup_message = message
                if passed:
                    warmup_successes += 1
                else:
                    print(f"Warm-up failed: {case_id}: {status}: {message}", flush=True)
                continue
            if repeat == 1 and case_id in HISTORY_CASE_IDS:
                histories[case_id] = []
                for point in result["history"]:
                    if point["phase"] == 2 and point["event"] == "iterate":
                        objective = point["objective"]
                        histories[case_id].append({
                            "phase": "phase2", "iteration": point["iteration"],
                            "objective": objective,
                            "relative_objective_gap": abs(objective - reference) / max(1.0, abs(reference)),
                        })
            if passed:
                successes += 1
                measured_successes += 1
                successful_times.append(result["runtime_seconds"])

            message = message + "; warmup=" + warmup_status
            if not warmup_passed:
                message = message + "; warmup_message=" + warmup_message
            record = {
                "protocol_version": "v1", "case_id": case_id, "algorithm": "Revised Simplex",
                "mode": "cold", "repeat_id": repeat, "status": status,
                "iterations": result["iterations"], "phase1_iterations": result["phase1_iterations"],
                "phase2_iterations": result["phase2_iterations"], "runtime_seconds": result["runtime_seconds"],
                "base_runtime_seconds": None, "environment_id": environment_id,
                "options_id": options_id, "message": message,
            }
            for name in metrics:
                record[name] = metrics[name]
            if set(record) != set(fields):
                raise ValueError("Result columns do not match the supplied schema.")
            records.append(record)

        if successes == args.repeats and warmup_passed:
            passed_cases += 1
        timing = "unavailable"
        if len(successful_times) > 0:
            timing = str(statistics.median(successful_times))
        print(f"[{number}/{len(cases)}] {case_id}: {successes}/{args.repeats} passed; "
              f"warmup={warmup_status}; median_seconds_successful_runs={timing}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "revised_simplex_results.csv"
    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary_file = output_dir / "case_summary.csv"
    summary_fields = ["case_id", "successful_runs", "total_runs",
                      "median_runtime_seconds", "runtime_iqr_seconds", "iterations", "phase1_iterations",
                      "phase2_iterations", "max_primal_residual_inf",
                      "max_relative_objective_error"]
    with summary_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(_case_summary(records))
    alpha_file = output_dir / "alpha_summary.csv"
    alpha_fields = ["M", "N", "alpha", "cases", "successful_cases", "paired_cases",
                    "median_objective", "median_relative_change_from_alpha_0_3"]
    with alpha_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=alpha_fields)
        writer.writeheader()
        writer.writerows(_alpha_summary(records, cases))
    if histories:
        history_dir = output_dir / "history"
        history_dir.mkdir(exist_ok=True)
        for case_id, points in histories.items():
            with (history_dir / f"{case_id}_history.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=["phase", "iteration", "objective",
                                                          "relative_objective_gap"])
                writer.writeheader()
                writer.writerows(points)
    print(f"Summary: {passed_cases}/{len(cases)} cases passed. Results: {output_file}; {summary_file}; {alpha_file}; {len(histories)} histories", flush=True)
    print(f"Measured: {measured_successes}/{len(records)}; warm-ups: {warmup_successes}/{len(cases)}.")
    full_protocol = (args.case_id is None and base_count == 27 and perturbed_count == 27
                     and args.repeats == 10 and passed_cases == 54)
    print(f"Full 54-case, 10-repeat protocol passed: {full_protocol}")
    if passed_cases == len(cases):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
