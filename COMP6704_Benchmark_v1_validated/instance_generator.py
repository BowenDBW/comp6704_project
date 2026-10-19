#!/usr/bin/env python3
"""COMP6704 capacitated transportation LP benchmark generator.

Generates fixed, reproducible standard-form LP instances shared by all four
solvers, including RHS-only demand perturbations for Dual Simplex warm starts.

Requires: numpy, scipy.
Usage: python instance_generator.py --output instances --validate-highs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.optimize import linprog

SIZES = ((3, 5), (5, 10), (10, 20))
ALPHAS = (0.3, 0.6, 0.9)
SEEDS = (2026, 2027, 2028)
FEAS_TOL = 1e-8


def base_parameters(m: int, n: int, seed: int):
    """Draw c, d, and a feasible reference allocation using a reproducible RNG.

    Draws do not depend on alpha, enabling controlled capacity comparisons.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, m, n, 0]))
    costs = rng.uniform(1.0, 20.0, size=(m, n))
    demands = rng.uniform(10.0, 30.0, size=n)
    weights = rng.uniform(0.5, 1.5, size=(m, n))
    witness_x = weights / weights.sum(axis=0, keepdims=True) * demands[None, :]
    supplies = 1.2 * witness_x.sum(axis=1)
    return costs, demands, supplies, witness_x


def perturbed_demands(demands: np.ndarray, m: int, n: int, seed: int):
    """Draw one deterministic +/-5% demand perturbation for each (m,n,seed)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, m, n, 1]))
    deltas = rng.uniform(-0.05, 0.05, size=n)
    return demands * (1.0 + deltas), deltas


def standard_form(costs: np.ndarray, demands: np.ndarray,
                  supplies: np.ndarray, capacities: np.ndarray):
    """Create min c_hat.T z, A z=b, z>=0, with z=[x(row-major),p,q(row-major)].

    Constraint ordering: supply rows, demand rows, capacity rows.
    The matrix contains all equality constraints including slack variables.
    """
    m, n = costs.shape
    mn = m * n
    nr = m + n + mn
    nc = 2 * mn + m
    A = np.zeros((nr, nc), dtype=np.float64)
    # Transport variable x[i,j] occupies column k=i*n+j.
    for i in range(m):
        for j in range(n):
            k = i * n + j
            A[i, k] = 1.0              # supply row for warehouse i
            A[m + j, k] = 1.0          # demand row for customer j
            A[m + n + k, k] = 1.0      # route-capacity row for (i,j)
            A[m + n + k, mn + m + k] = 1.0  # q[i,j]
        A[i, mn + i] = 1.0             # supply slack p[i]

    b = np.concatenate([supplies, demands, capacities.ravel(order="C")])
    c_hat = np.concatenate([costs.ravel(order="C"), np.zeros(m + mn)])
    return A, b, c_hat


def check_witness(A: np.ndarray, b: np.ndarray, costs: np.ndarray,
                  supplies: np.ndarray, capacities: np.ndarray,
                  transport_x: np.ndarray, tolerance: float = FEAS_TOL):
    """Verify feasibility analytically using a nonoptimal witness allocation."""
    m, n = costs.shape
    p = supplies - transport_x.sum(axis=1)
    q = capacities - transport_x
    z = np.concatenate([transport_x.ravel(order="C"), p, q.ravel(order="C")])
    assert A.shape == (m + n + m*n, 2*m*n + m)
    assert b.shape == (m + n + m*n,)
    assert z.shape == (2*m*n + m,)
    assert np.min(z) >= -tolerance, "Generated instance violates nonnegativity"
    residual = float(np.max(np.abs(A @ z - b)))
    assert residual <= tolerance, f"Generated instance is infeasible: {residual}"
    return residual


def verify_highs(A: np.ndarray, b: np.ndarray, c_hat: np.ndarray):
    """Independent LP optimality check (never supplies results to other solvers)."""
    result = linprog(c_hat, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    if not result.success:
        raise AssertionError(f"HiGHS failed: {result.message}")
    residual = float(np.max(np.abs(A @ result.x - b)))
    assert residual < 1e-7, f"Unexpected HiGHS primal residual {residual}"
    return float(result.fun)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def export_case(output: Path, case_id: str, A: np.ndarray,
                b: np.ndarray, c_hat: np.ndarray):
    npz_path = output / f"{case_id}.npz"
    mat_path = output / f"{case_id}.mat"
    np.savez_compressed(npz_path, A=A, b=b, c_hat=c_hat)
    savemat(mat_path, {"A": A, "b": b.reshape(-1, 1),
                       "c_hat": c_hat.reshape(-1, 1)}, do_compression=True)
    return str(npz_path.name), str(mat_path.name), sha256(npz_path)


def generate_all(output: Path, validate_highs: bool):
    output.mkdir(parents=True, exist_ok=True)
    data_dir = output / "instances"
    data_dir.mkdir(exist_ok=True)
    rows = []
    optima = {}
    for m, n in SIZES:
        for seed in SEEDS:
            costs, d, s, witness = base_parameters(m, n, seed)
            d_perturbed, deltas = perturbed_demands(d, m, n, seed)
            x_perturbed = witness * (d_perturbed / d)[None, :]
            for alpha in ALPHAS:
                u = (1.0 + 2.0 * (1.0 - alpha)) * witness
                A, b, c_hat = standard_form(costs, d, s, u)
                assert np.array_equal(b[m:m+n], d)
                assert np.array_equal(c_hat[:m*n], costs.ravel())
                check_witness(A, b, costs, s, u, witness)
                base_id = f"M{m:02d}_N{n:02d}_a{round(alpha*100):03d}_seed{seed}"
                for kind in ("base", "demand_perturbed"):
                    this_b = b.copy()
                    if kind == "demand_perturbed":
                        # RHS-only update: must NOT alter A or c_hat.
                        this_b[m:m+n] = d_perturbed
                        check_witness(A, this_b, costs, s, u, x_perturbed)
                    file_id = f"{base_id}_{kind}"
                    npz_name, mat_name, digest = export_case(
                        data_dir, file_id, A, this_b, c_hat
                    )
                    optimum = verify_highs(A, this_b, c_hat) if validate_highs else None
                    if optimum is not None:
                        optima[file_id] = optimum
                    rows.append({
                        "case_id": file_id,
                        "kind": kind,
                        "M": m, "N": n,
                        "alpha": alpha, "seed": seed,
                        "rows_A": A.shape[0], "columns_A": A.shape[1],
                        "npz_file": "instances/" + npz_name,
                        "mat_file": "instances/" + mat_name,
                        "sha256_npz": digest,
                        "highs_objective": "" if optimum is None else f"{optimum:.12g}"
                    })
            # These deltas depend on (M,N,seed), not on alpha.
            assert np.max(np.abs(deltas)) <= 0.05

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "benchmark_name": "COMP6704 Capacitated Transportation LP Benchmark v1.0",
        "sizes_M_N": [list(k) for k in SIZES],
        "alphas": list(ALPHAS),
        "seeds": list(SEEDS),
        "num_base_instances": len(rows)//2,
        "num_demand_perturbations": len(rows)//2,
        "tolerance": FEAS_TOL,
        "lp": "min c_hat.T@z subject to A@z=b, z>=0",
        "z_order": "[x row-major (M*N); supply_slack p (M); capacity_slack q row-major (M*N)]",
        "A_rows": "[supply (M); demand (N); capacity (M*N)]",
        "c_ij": "Independent U[1,20]",
        "d_j": "Independent U[10,30]",
        "weight_ij": "Independent U[0.5,1.5]",
        "x_ref_ij": "d_j * weight_ij / sum_i weight_ij",
        "s_i": "1.2 * sum_j x_ref_ij",
        "u_ij": "(1+2*(1-alpha)) * x_ref_ij",
        "demand_perturbation": "d'_j=d_j*(1+delta_j); independent delta_j~U[-0.05,0.05]",
        "random_generator": "numpy.random.default_rng(numpy.random.SeedSequence([seed,M,N,stream]))",
        "stream_0": "base cost, demand, weight generation",
        "stream_1": "demand perturbation; same across alpha",
        "highs_validated": bool(validate_highs),
        "files": "Each .npz/.mat has A, b, c_hat. Companion manifest.csv stores metadata."
    }
    with (output / "benchmark_config.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        f.write("\n")

    # Defensive regression checks on files, not just in-memory matrices.
    for m, n in SIZES:
        for seed in SEEDS:
            previous = None
            for alpha in ALPHAS:
                name = f"M{m:02d}_N{n:02d}_a{round(alpha*100):03d}_seed{seed}"
                with np.load(data_dir / (name + "_base.npz")) as p, \
                     np.load(data_dir / (name + "_demand_perturbed.npz")) as r:
                    assert set(p.files) == {"A", "b", "c_hat"}
                    assert np.array_equal(p["A"], r["A"])
                    assert np.array_equal(p["c_hat"], r["c_hat"])
                    assert np.array_equal(p["b"][:m], r["b"][:m])
                    assert np.array_equal(p["b"][m+n:], r["b"][m+n:])
                    if previous is not None:
                        assert np.array_equal(p["A"], previous[0])
                        assert np.array_equal(p["c_hat"], previous[1])
                        assert np.array_equal(p["b"][:m+n], previous[2][:m+n])
                        assert np.all(p["b"][m+n:] <= previous[2][m+n:] + 1e-12)
                    previous = (p["A"].copy(), p["c_hat"].copy(), p["b"].copy())
    print(f"Wrote {len(rows)} verified instances ({len(rows)//2} base + "
          f"{len(rows)//2} demand-perturbed) to {output}")
    if optima:
        vals = list(optima.values())
        print(f"HiGHS reference optimal objectives: min={min(vals):.6f}, "
              f"max={max(vals):.6f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path("benchmark_v1"))
    ap.add_argument("--validate-highs", action="store_true",
                    help="also solve each instance with scipy.optimize.linprog(method='highs')")
    args = ap.parse_args()
    generate_all(args.output, args.validate_highs)


if __name__ == "__main__":
    main()
