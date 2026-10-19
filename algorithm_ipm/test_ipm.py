"""Self-checks for the IPM component (run BEFORE the benchmark).

Usage:  python test_ipm.py

Only these tests use scipy.optimize.linprog (HiGHS) as an INDEPENDENT
cross-check of a self-written solver on tiny hand-built LPs. Per the group
protocol, HiGHS is used for validation only and never inside the benchmark
pipeline in run_benchmark.py.

Run with:  python -X utf8 test_ipm.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from ipm_solver import IPMOptions, protocol_metrics, solve_ipm

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name} {detail}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {detail}")


def test_known_optimum_tiny_lp() -> None:
    print("[1] tiny LP with known analytic optimum")
    # min x1 + 2*x2  s.t.  x1 + x2 = 1, x >= 0  ->  x* = (1, 0), f* = 1
    A = np.array([[1.0, 1.0]])
    b = np.array([1.0])
    c = np.array([1.0, 2.0])
    res = solve_ipm(A, b, c)
    m = protocol_metrics(A, b, c, res.z, res.y, res.r)
    check("status", res.status == "optimal", res.message)
    check("objective", abs(m["objective"] - 1.0) < 1e-7, f"obj={m['objective']:.12f}")
    check("x* = (1,0)", abs(res.z[0] - 1.0) < 1e-6 and res.z[1] < 1e-6,
          f"z={np.round(res.z, 8)}")


def test_cross_check_vs_linprog() -> None:
    print("[2] cross-check vs HiGHS (scipy linprog) on random standard-form LPs")
    rng = np.random.default_rng(7)
    worst_obj_err = 0.0
    for trial in range(20):
        L, V = 12, 25
        A = rng.uniform(-1.0, 1.0, size=(L, V))
        z_star = rng.uniform(0.5, 2.0, size=V)      # planted feasible point
        b = A @ z_star                              # guarantees feasibility
        c = rng.uniform(1.0, 10.0, size=V)
        res = solve_ipm(A, b, c)
        if res.status != "optimal":
            check(f"trial {trial} status", False, res.message)
            continue
        m = protocol_metrics(A, b, c, res.z, res.y, res.r)
        ref = linprog(c, A_eq=A, b_eq=b, bounds=[(0, None)] * V, method="highs")
        err = abs(m["objective"] - ref.fun) / max(1.0, abs(ref.fun))
        worst_obj_err = max(worst_obj_err, err)
        ok = (m["primal_residual_inf"] <= 1e-7 and m["min_z"] >= -1e-8
              and m["dual_residual_inf"] <= 1e-7 and m["min_dual_slack"] >= -1e-7
              and m["relative_duality_gap"] <= 1e-7 and err <= 1e-7)
        if not ok:
            check(f"trial {trial} metrics", False,
                  f"pr={m['primal_residual_inf']:.2e} dr={m['dual_residual_inf']:.2e} "
                  f"gap={m['relative_duality_gap']:.2e} obj_err={err:.2e}")
    check("20/20 trials within protocol thresholds", worst_obj_err <= 1e-7,
          f"worst relative objective error = {worst_obj_err:.3e}")


def test_row_major_ordering() -> None:
    print("[3] row-major recovery on a real benchmark instance")
    import pathlib
    here = pathlib.Path(__file__).resolve().parent
    inst = here.parent / "COMP6704_Benchmark_v1_validated" / "benchmark_v1" / "instances"
    case = "M03_N05_a030_seed2026_base"
    with np.load(inst / f"{case}.npz", allow_pickle=False) as d:
        A, b, c_hat = d["A"], d["b"], d["c_hat"]
    check("calibration dimensions", A.shape == (23, 33) and b.shape == (23,) and c_hat.shape == (33,),
          f"A={A.shape}")
    M, N = 3, 5
    K, L = M * N, M + N + M * N
    res = solve_ipm(A, b, c_hat)
    m = protocol_metrics(A, b, c_hat, res.z, res.y, res.r)
    check("status", res.status == "optimal", f"iters={res.iterations}")
    check("primal_residual", m["primal_residual_inf"] <= 1e-7, f"{m['primal_residual_inf']:.3e}")
    check("min_z", m["min_z"] >= -1e-8, f"{m['min_z']:.3e}")
    check("dual_residual", m["dual_residual_inf"] <= 1e-7, f"{m['dual_residual_inf']:.3e}")
    check("min_dual_slack", m["min_dual_slack"] >= -1e-7, f"{m['min_dual_slack']:.3e}")
    check("duality_gap", m["relative_duality_gap"] <= 1e-7, f"{m['relative_duality_gap']:.3e}")
    f_ref = 776.2029706947461  # VALIDATION_SUMMARY.csv, full precision
    err = abs(m["objective"] - f_ref) / max(1.0, abs(f_ref))
    check("vs HiGHS reference", err <= 1e-7, f"rel_err={err:.3e}")

    # Physical-parameter recovery per SHARED_PROBLEM_SPECIFICATION.md sec. 3
    supplies = b[:M]
    demands = b[M:M + N]
    capacities = b[M + N:].reshape(M, N)
    costs = c_hat[:K].reshape(M, N)
    X = res.z[:K].reshape(M, N)          # row-major: x_ij at [i, j]
    p = res.z[K:K + M]
    q = res.z[K + M:].reshape(M, N)
    check("supply rows: sum_j x_ij + p_i == s_i",
          np.abs(X.sum(axis=1) + p - supplies).max() <= 1e-7)
    check("demand rows: sum_i x_ij == d_j",
          np.abs(X.sum(axis=0) - demands).max() <= 1e-7)
    check("capacity rows: x_ij + q_ij == u_ij",
          np.abs(X + q - capacities).max() <= 1e-7)
    check("costs c_ij positive in [1,20]", costs.min() >= 1.0 and costs.max() <= 20.0)
    check("objective == sum_ij c_ij x_ij",
          abs(m["objective"] - float((costs * X).sum())) <= 1e-9)


def test_options_documented_and_deterministic() -> None:
    print("[4] determinism and documented options")
    rng = np.random.default_rng(11)
    A = rng.uniform(0.1, 2.0, size=(15, 30))
    zp = rng.uniform(0.5, 3.0, 30)
    b = A @ zp
    c = rng.uniform(1.0, 5.0, 30)
    r1 = solve_ipm(A, b, c, IPMOptions())
    r2 = solve_ipm(A, b, c, IPMOptions())
    check("same input -> bit-identical output (no RNG in solver)",
          np.array_equal(r1.z, r2.z) and r1.iterations == r2.iterations)
    opts = IPMOptions()
    check("default options recorded", opts.primal_abs_tol == 1e-9 and opts.gap_rel_tol == 1e-9)


if __name__ == "__main__":
    test_known_optimum_tiny_lp()
    test_cross_check_vs_linprog()
    test_row_major_ordering()
    test_options_documented_and_deterministic()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
