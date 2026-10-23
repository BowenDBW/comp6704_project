"""Protocol section 3, step 1: interface calibration on M03_N05_a030_seed2026_base.

Usage:  python -X utf8 run_calibration.py

Checks (protocol): A=(23,33), b length 23, c_hat length 33, then runs the
solver and prints the common acceptance metrics. Also writes the full
per-iteration convergence history to runs/ipm_calibration/history/.
"""

from __future__ import annotations

import csv
import pathlib
import time

import numpy as np

from ipm_solver import IPMOptions, protocol_metrics, solve_ipm

HERE = pathlib.Path(__file__).resolve().parent
BENCH = HERE.parent / "shared"
CASE = "M03_N05_a030_seed2026_base"


def main() -> None:
    print(f"Calibration case: {CASE}")
    with np.load(BENCH / "benchmark_v1" / "instances" / f"{CASE}.npz", allow_pickle=False) as d:
        A, b, c_hat = d["A"], d["b"], d["c_hat"]
    print(f"A={A.shape}, len(b)={b.shape[0]}, len(c_hat)={c_hat.shape[0]}  "
          f"(expected (23,33), 23, 33)")
    assert A.shape == (23, 33) and b.shape == (23,) and c_hat.shape == (33,)

    res = solve_ipm(A, b, c_hat, IPMOptions())
    m = protocol_metrics(A, b, c_hat, res.z, res.y, res.r)
    print(f"status                 : {res.status}")
    print(f"iterations             : {res.iterations}")
    print(f"objective              : {m['objective']!r}")
    print(f"primal_residual_inf    : {m['primal_residual_inf']:.6e}   (<=1e-7)")
    print(f"min_z                  : {m['min_z']:.6e}   (>=-1e-8)")
    print(f"dual_residual_inf      : {m['dual_residual_inf']:.6e}   (<=1e-7)")
    print(f"min_dual_slack         : {m['min_dual_slack']:.6e}   (>=-1e-7)")
    print(f"relative_duality_gap   : {m['relative_duality_gap']:.6e}   (<=1e-7)")

    # external reference check (never used inside the solver)
    with open(BENCH / "VALIDATION_SUMMARY.csv", newline="", encoding="utf-8") as f:
        row = next(r for r in csv.DictReader(f) if r["case_id"] == CASE)
    f_ref = float(row["highs_objective"])
    err = abs(m["objective"] - f_ref) / max(1.0, abs(f_ref))
    print(f"HiGHS f_ref            : {f_ref!r}")
    print(f"relative_objective_err : {err:.6e}   (<=1e-7)")

    hist_dir = HERE.parent / "runs" / "ipm_calibration" / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    out = hist_dir / f"{CASE}_history.csv"
    fieldnames: list[str] = []
    for rec in res.history:
        for k in rec:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(res.history)
    print(f"convergence history    : {out.relative_to(HERE.parent)}")

    ok = (res.status == "optimal"
          and m["primal_residual_inf"] <= 1e-7 and m["min_z"] >= -1e-8
          and m["dual_residual_inf"] <= 1e-7 and m["min_dual_slack"] >= -1e-7
          and m["relative_duality_gap"] <= 1e-7 and err <= 1e-7)
    print("CALIBRATION:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"(wall clock incl. I/O: {time.perf_counter() - t0:.3f}s)")
