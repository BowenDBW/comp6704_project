"""Full benchmark run of the Primal-Dual IPM over the shared COMP6704 v1 instances.

Usage:  python -X utf8 run_benchmark.py

Protocol compliance (PROPOSED_COMPARISON_PROTOCOL.md):
- Data: all 54 released instances (27 base + 27 demand-perturbed), keyed by case_id.
  Base cases are the four-algorithm comparison set; the paired perturbed cases are
  included here as cold-start solves for completeness (the paired warm-start
  reoptimization experiment belongs to the Dual Simplex owner).
- Timing: monotonic perf_counter, from solver entry (initialization included)
  to return; file loading, metric computation and reference look-up are outside
  the timed region. Per case: 1 untimed warm-up, then 10 independent timed runs;
  median reported. Cold start every run; no state reuse between runs.
- Output: one CSV row per run with the exact result_schema.csv header; empty
  fields stay empty (never 0). Solutions (z, y, r) of timed repeat 1 are saved
  per case under results/solutions/.
- HiGHS reference objectives are read from VALIDATION_SUMMARY.csv only AFTER
  solving, to compute relative_objective_error. Nothing else uses them.
"""

from __future__ import annotations

import os

# Pin BLAS threads BEFORE numpy is imported. The normal-equation matrices here
# are small (<= 230x230); multi-threaded OpenBLAS adds ~30x sync overhead on
# them. The pinned setting is recorded in environment_record.json per protocol
# section 4 (fixed thread configuration for the formal runtime comparison).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import contextlib
import csv
import ctypes
import io
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import scipy

from ipm_solver import IPMOptions, protocol_metrics, solve_ipm

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "COMP6704_Benchmark_v1_validated"
RESULTS = HERE / "results"

ALGORITHM = "primal_dual_ipm"
PROTOCOL_VERSION = "v1"
OPTIONS_ID = "ipm_default_v1"
REPEATS = 10
HISTORY_CASES = {"M03_N05_a030_seed2026_base",          # protocol calibration case
                 "M10_N20_a090_seed2028_base",          # largest + tightest capacity
                 "M10_N20_a090_seed2028_demand_perturbed"}

SCHEMA_FIELDS = ["protocol_version", "case_id", "algorithm", "mode", "repeat_id",
                 "status", "objective", "primal_residual_inf", "min_z",
                 "dual_residual_inf", "min_dual_slack", "relative_duality_gap",
                 "relative_objective_error", "iterations", "phase1_iterations",
                 "phase2_iterations", "runtime_seconds", "base_runtime_seconds",
                 "environment_id", "options_id", "message"]


def environment_record() -> tuple[str, dict]:
    """environment_id + machine/software snapshot for ENVIRONMENT_TEMPLATE.md."""
    total_gb = None
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            total_gb = round(stat.ullTotalPhys / 2**30, 1)

    blas_buf = io.StringIO()
    with contextlib.redirect_stdout(blas_buf):
        np.show_config()
    blas_text = " ".join(blas_buf.getvalue().split())

    env = {
        "os": platform.platform(),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "physical_cores_logical": os.cpu_count(),
        "ram_total_gb": total_gb,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas_summary": blas_text[:400],
        "thread_env": {k: os.environ.get(k, "<unset>")
                       for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                 "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }
    env_id = (f"ENV_{platform.system()}_PY{env['python'].replace('.', '')[:3]}"
              f"_NP{env['numpy'].replace('.', '')[:3]}")
    return env_id, env


def load_references() -> dict[str, float]:
    refs = {}
    with open(BENCH / "VALIDATION_SUMMARY.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            refs[row["case_id"]] = float(row["highs_objective"])
    return refs


def fmt(x: float) -> str:
    """Full-precision, round-trippable text for CSV cells."""
    return repr(float(x))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "solutions").mkdir(exist_ok=True)
    (RESULTS / "history").mkdir(exist_ok=True)

    env_id, env = environment_record()
    with open(RESULTS / "environment_record.json", "w", encoding="utf-8") as f:
        json.dump({"environment_id": env_id, **env}, f, indent=2)

    refs = load_references()
    case_ids = list(refs.keys())
    print(f"environment_id={env_id}   cases={len(case_ids)}   repeats={REPEATS}")

    all_rows: list[dict] = []
    per_case: dict[str, dict] = {}

    for idx, case_id in enumerate(case_ids, 1):
        with np.load(BENCH / "benchmark_v1" / "instances" / f"{case_id}.npz",
                     allow_pickle=False) as d:
            A = np.array(d["A"]); b = np.array(d["b"]).ravel(); c_hat = np.array(d["c_hat"]).ravel()
        f_ref = refs[case_id]

        # ---- warm-up (untimed, discarded from statistics; status still counted)
        warm = solve_ipm(A, b, c_hat, IPMOptions())

        runs = []
        for rep in range(1, REPEATS + 1):
            t0 = time.perf_counter()
            res = solve_ipm(A, b, c_hat, IPMOptions())
            dt = time.perf_counter() - t0

            m = protocol_metrics(A, b, c_hat, res.z, res.y, res.r)
            obj_err = abs(m["objective"] - f_ref) / max(1.0, abs(f_ref))
            row = {
                "protocol_version": PROTOCOL_VERSION,
                "case_id": case_id,
                "algorithm": ALGORITHM,
                "mode": "cold",
                "repeat_id": str(rep),
                "status": res.status,
                "objective": fmt(m["objective"]),
                "primal_residual_inf": fmt(m["primal_residual_inf"]),
                "min_z": fmt(m["min_z"]),
                "dual_residual_inf": fmt(m["dual_residual_inf"]),
                "min_dual_slack": fmt(m["min_dual_slack"]),
                "relative_duality_gap": fmt(m["relative_duality_gap"]),
                "relative_objective_error": fmt(obj_err),
                "iterations": str(res.iterations),
                "phase1_iterations": "",       # not applicable to IPM (protocol sec. 3)
                "phase2_iterations": "",
                "runtime_seconds": fmt(dt),
                "base_runtime_seconds": "",    # cold mode only
                "environment_id": env_id,
                "options_id": OPTIONS_ID,
                "message": res.message,
            }
            all_rows.append(row)
            runs.append({"repeat": rep, "status": res.status, "runtime": dt,
                         "iterations": res.iterations, "obj_err": obj_err,
                         "metrics": m})

            if rep == 1:
                np.savez(RESULTS / "solutions" / f"{case_id}_solution.npz",
                         z=res.z, y=res.y, r=res.r, status=res.status,
                         iterations=res.iterations)

        if case_id in HISTORY_CASES:
            res_h = solve_ipm(A, b, c_hat, IPMOptions())
            fieldnames: list[str] = []
            for rec in res_h.history:
                for k in rec:
                    if k not in fieldnames:
                        fieldnames.append(k)
            with open(RESULTS / "history" / f"{case_id}_history.csv", "w",
                      newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(res_h.history)

        oks = [r for r in runs if r["status"] == "optimal"]
        times = [r["runtime"] for r in runs]
        iters = [r["iterations"] for r in runs]
        per_case[case_id] = {
            "n_runs": len(runs),
            "n_optimal": len(oks),
            "status_counts": {s: sum(1 for r in runs if r["status"] == s)
                              for s in {r["status"] for r in runs}},
            "median_runtime_seconds": statistics.median(times),
            "min_runtime_seconds": min(times),
            "max_runtime_seconds": max(times),
            "median_iterations": statistics.median(iters),
            "max_primal_residual_inf": max(r["metrics"]["primal_residual_inf"] for r in runs),
            "min_min_z": min(r["metrics"]["min_z"] for r in runs),
            "max_dual_residual_inf": max(r["metrics"]["dual_residual_inf"] for r in runs),
            "min_min_dual_slack": min(r["metrics"]["min_dual_slack"] for r in runs),
            "max_relative_duality_gap": max(r["metrics"]["relative_duality_gap"] for r in runs),
            "max_relative_objective_error": max(r["obj_err"] for r in runs),
            "f_ref": f_ref,
        }
        pc = per_case[case_id]
        print(f"[{idx:2d}/{len(case_ids)}] {case_id:<38s} "
              f"opt {pc['n_optimal']}/{len(runs)}  iters {pc['median_iterations']:>3}  "
              f"t_med {pc['median_runtime_seconds']*1e3:7.2f} ms  "
              f"obj_err {pc['max_relative_objective_error']:.2e}")

    # ---------------- write outputs -----------------------------------------
    out_csv = RESULTS / "ipm_results_v1.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA_FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    with open(RESULTS / "case_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", *next(iter(per_case.values())).keys()])
        w.writeheader()
        for cid, agg in per_case.items():
            w.writerow({"case_id": cid, **agg})

    total_rows = len(all_rows)
    optimal_rows = sum(1 for r in all_rows if r["status"] == "optimal")
    max_pr = max(per_case[c]["max_primal_residual_inf"] for c in per_case)
    max_dr = max(per_case[c]["max_dual_residual_inf"] for c in per_case)
    max_gap = max(per_case[c]["max_relative_duality_gap"] for c in per_case)
    max_oe = max(per_case[c]["max_relative_objective_error"] for c in per_case)
    worst_mz = min(per_case[c]["min_min_z"] for c in per_case)
    worst_mds = min(per_case[c]["min_min_dual_slack"] for c in per_case)
    summary = {
        "algorithm": ALGORITHM, "protocol_version": PROTOCOL_VERSION,
        "options_id": OPTIONS_ID, "environment_id": env_id,
        "n_cases": len(per_case), "n_runs": total_rows, "n_optimal": optimal_rows,
        "protocol_thresholds": {"primal_residual_inf": 1e-7, "min_z": -1e-8,
                                "dual_residual_inf": 1e-7, "min_dual_slack": -1e-7,
                                "relative_duality_gap": 1e-7, "relative_objective_error": 1e-7},
        "observed_worst": {"max_primal_residual_inf": max_pr, "min_min_z": worst_mz,
                           "max_dual_residual_inf": max_dr, "min_min_dual_slack": worst_mds,
                           "max_relative_duality_gap": max_gap,
                           "max_relative_objective_error": max_oe},
        "all_thresholds_met": bool(max_pr <= 1e-7 and worst_mz >= -1e-8 and max_dr <= 1e-7
                                   and worst_mds >= -1e-7 and max_gap <= 1e-7 and max_oe <= 1e-7),
        "environment": env,
        "per_case": per_case,
    }
    with open(RESULTS / "ipm_results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 100)
    print(f"rows={total_rows}  optimal={optimal_rows}  ->  {out_csv.name}")
    print(f"worst primal_res={max_pr:.3e}  worst min_z={worst_mz:.3e}  "
          f"worst dual_res={max_dr:.3e}  worst min_dual_slack={worst_mds:.3e}")
    print(f"worst rel_gap={max_gap:.3e}  worst rel_obj_err={max_oe:.3e}")
    print(f"ALL PROTOCOL THRESHOLDS MET: {summary['all_thresholds_met']}")


if __name__ == "__main__":
    main()
